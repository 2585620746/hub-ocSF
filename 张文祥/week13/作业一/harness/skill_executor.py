"""
---
name: skill-executor
description: >-
  渐进式 skill 执行引擎。按 Step 顺序逐步执行，每步完成后自动进入下一步，
  支持在步骤间暂停、等待用户输入、或调用 LLM 决策。
---

# skill_executor — 渐进式执行引擎

## 触发/使用场景

- 用户选中一个 skill 后开始执行
- 每步执行后自动推进到下一步
- 遇到 ask_user 动作时等待用户输入确认

## 执行流程

1. 接收 Skill 对象 → 创建 ExecutionContext（含当前步骤索引、状态、变量存储）
2. 每步依次执行：
   - describe → 打印步骤说明，不执行动作
   - identify → 从用户输入中提取信息，结果存入 context.vars
   - generate → 创建文件/目录，路径存 context.vars
   - run_python → 执行 Python 命令
   - run_command → 执行任意 shell 命令
   - open_file → 打开文件/URL
   - ask_user → 暂停执行，等待用户输入（progressive 关键点）
3. 每步完成后自动推进 next()
4. 全部完成后状态变为 completed

## 使用方式

```python
from skill_executor import ProgressiveExecutor
executor = ProgressiveExecutor(skill)
executor.start(context={"user_input": "给我做张 crazy 的闪卡"})
while executor.status() == "running":
    result = executor.step()
    print(result)
    # 如果 result.action == "ask_user": input = await user_input()
```
"""

import os
import subprocess
import shlex
import json
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI

from skill_loader import Skill, Step

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    step: int
    action: str
    description: str
    status: str  # "ok" | "error" | "awaiting_input"
    output: str = ""
    prompt: str = ""


@dataclass
class ExecutionContext:
    skill_name: str
    vars: dict = field(default_factory=dict)
    step_index: int = 0
    status: str = "idle"  # idle | running | paused | completed | cancelled | error


class ActionHandler:
    def __init__(self, executor: "ProgressiveExecutor"):
        self.executor = executor

    def handle(self, step: Step, ctx: ExecutionContext) -> StepResult:
        handler = getattr(self, f"do_{step.action}", self.do_describe)
        return handler(step, ctx)

    def do_describe(self, step: Step, ctx: ExecutionContext) -> StepResult:
        return StepResult(step=step.order, action=step.action, description=step.description, status="ok", output=f"[说明] {step.description}")

    def do_identify(self, step: Step, ctx: ExecutionContext) -> StepResult:
        user_input = ctx.vars.get("user_input", "")
        words = [w for w in user_input.split() if w.isascii() and w.isalpha()]
        word = words[-1].lower() if words else ""
        if word:
            ctx.vars["target_word"] = word
            return StepResult(step=step.order, action=step.action, description=step.description, status="ok", output=f"提取到单词: {word}")
        return StepResult(step=step.order, action=step.action, description=step.description, status="awaiting_input", output="", prompt="请输入目标英语单词:")

    def do_generate(self, step: Step, ctx: ExecutionContext) -> StepResult:
        word = ctx.vars.get("target_word", "unknown")
        data_dir = self.executor.skill_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        json_path = data_dir / f"{word}.json"
        data = self._generate_word_data(word)
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx.vars["target_word"] = word
        ctx.vars["json_path"] = str(json_path).replace("\\", "/")
        return StepResult(step=step.order, action=step.action, description=step.description, status="ok", output=f"数据已保存: {json_path}")

    def do_run_python(self, step: Step, ctx: ExecutionContext) -> StepResult:
        command = self._resolve_command(step.command, ctx)
        if not command:
            return StepResult(step=step.order, action=step.action, description=step.description, status="error", output="无可用命令")
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        output = (result.stdout or "")[:500] + (("\n" + result.stderr[:200]) if result.returncode != 0 else "")
        status = "ok" if result.returncode == 0 else "error"
        return StepResult(step=step.order, action=step.action, description=step.description, status=status, output=output.strip() or "执行完成")

    def do_run_command(self, step: Step, ctx: ExecutionContext) -> StepResult:
        return self.do_run_python(step, ctx)

    def do_open_file(self, step: Step, ctx: ExecutionContext) -> StepResult:
        command = self._resolve_command(step.command, ctx)
        if command:
            try:
                if os.name == "nt":
                    os.startfile(command.rsplit(" ", 1)[-1].strip("\"") if " " in command else command)
                else:
                    subprocess.Popen(["xdg-open", command.rsplit(" ", 1)[-1].strip("\"")], start_new_session=True)
            except Exception as e:
                return StepResult(step=step.order, action=step.action, description=step.description, status="ok", output=f"无法自动打开: {e}，文件路径: {command}")
        return StepResult(step=step.order, action=step.action, description=step.description, status="ok", output="文件已打开")

    def do_ask_user(self, step: Step, ctx: ExecutionContext) -> StepResult:
        return StepResult(step=step.order, action=step.action, description=step.description, status="awaiting_input", prompt=step.params.get("prompt", step.description))

    def _resolve_command(self, command: str | None, ctx: ExecutionContext) -> str | None:
        if not command:
            return None
        word = ctx.vars.get("target_word", "")
        json_path = ctx.vars.get("json_path", "")

        # Replace variable placeholders
        resolved = command.replace("<word>", word).replace("<json_path>", json_path)

        # Map .cursor/skills/<name>/ to the actual skill directory
        if ".cursor/skills" in resolved:
            skill_dir_str = str(self.executor.skill_dir).replace("\\", "/")
            resolved = resolved.replace(".cursor/skills/flash-card", skill_dir_str)
            resolved = resolved.replace(".cursor/skills", str(self.executor.skill_dir.parent).replace("\\", "/"))

        return resolved

    def _generate_word_data(self, word: str, _retry: int = 0) -> dict:
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return self._placeholder_word_data(word, "需要设置 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY")
        try:
            base_url = "https://api.deepseek.com" if os.getenv("DEEPSEEK_API_KEY") else "https://dashscope.aliyuncs.com/compatible-mode/v1"
            client = OpenAI(api_key=api_key, base_url=base_url)
            model = "deepseek-v4-flash" if os.getenv("DEEPSEEK_API_KEY") else "qwen-plus"
            prompt = f"""为英语单词 "{word}" 生成学习数据。必须返回**完整JSON对象**（不要代码块包裹，不要省略任何字段）：

{{
  "phonetic": "填写真实国际音标，用 /.../ 包裹，不要空着",
  "pos": "填写词性缩写如 adj./v./n./adv.，不要空着",
  "definition": "填写简洁准确的中文释义，不要空着",
  "examples": [
    {{"en": "地道的英文例句，体现该词典型用法", "zh": "对应中文翻译"}},
    {{"en": "第二条例句，不同用法", "zh": "对应中文翻译"}},
    {{"en": "第三条例句，不同语境", "zh": "对应中文翻译"}}
  ],
  "synonyms": ["近义词1", "近义词2", "近义词3", "近义词4"]
}}

要求：
- phonetic、pos、definition **不允许为空**
- 例句恰好 3 条，地道且长度适中
- 近义词 4-6 个
- 只输出 JSON，不要任何其他文字"""
            resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.3, timeout=20)
            text = resp.choices[0].message.content.strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
            data = json.loads(text)
            data["word"] = word
            # Validate and fill missing required fields
            if not data.get("phonetic"):
                data["phonetic"] = f"/{word}/"
            if not data.get("pos"):
                data["pos"] = "n."
            if not data.get("definition"):
                data["definition"] = f"单词 '{word}' 的含义"
            if not isinstance(data.get("examples"), list) or len(data["examples"]) != 3:
                data["examples"] = self._placeholder_word_data(word, "")["examples"]
            if not isinstance(data.get("synonyms"), list):
                data["synonyms"] = []
            # Retry once if response is garbage
            if _retry < 1 and (not data.get("phonetic", "").strip("/ ") or not data.get("pos", "").strip()):
                return self._generate_word_data(word, _retry=1)
            return data
        except Exception as e:
            if _retry < 1:
                return self._generate_word_data(word, _retry=1)
            return self._placeholder_word_data(word, str(e))

    def _placeholder_word_data(self, word: str, reason: str = "") -> dict:
        return {
            "word": word,
            "phonetic": "/" + word + "/",
            "pos": "",
            "definition": reason or f"单词 '{word}' 的释义" ,
            "examples": [
                {"en": f"'{word}' is commonly used in everyday English.", "zh": f"'{word}' 在日常英语中很常用。"},
                {"en": f"Learning how to use '{word}' correctly improves your fluency.", "zh": f"学会正确使用 '{word}' 能提升流利度。"},
                {"en": f"Can you write a sentence with '{word}'?", "zh": f"你能用 '{word}' 造个句吗？"},
            ],
            "synonyms": [],
        }


class ProgressiveExecutor:
    def __init__(self, skill: Skill, skill_dir: Path | None = None):
        self.skill = skill
        self.skill_dir = skill_dir or skill.skill_dir
        self.handler = ActionHandler(self)
        self.ctx = ExecutionContext(skill_name=skill.name)

    def start(self, context: dict | None = None):
        if context:
            self.ctx.vars.update(context)
        self.ctx.status = "running"
        self.ctx.step_index = 0

    def step(self) -> Optional[StepResult]:
        if self.ctx.status != "running":
            return None
        if self.ctx.step_index >= len(self.skill.steps):
            self.ctx.status = "completed"
            return StepResult(step=-1, action="", description="", status="completed", output="全部步骤已完成")
        step = self.skill.steps[self.ctx.step_index]
        result = self.handler.handle(step, self.ctx)
        if result.status == "awaiting_input":
            self.ctx.status = "paused"
        elif result.status == "error":
            self.ctx.status = "error"
        else:
            self.ctx.step_index += 1
        return result

    def resume(self, user_input: str = ""):
        if user_input:
            self.ctx.vars["user_input"] = user_input
            step = self.skill.steps[self.ctx.step_index]
            if step.action == "identify" and not self.ctx.vars.get("target_word"):
                words = [w for w in user_input.split() if w.isascii() and w.isalpha()]
                word = words[-1].lower() if words else user_input.strip().lower()
                self.ctx.vars["target_word"] = word
            elif step.action == "ask_user":
                self.ctx.vars["last_answer"] = user_input
            self.ctx.step_index += 1
        self.ctx.status = "running"

    def status(self) -> str:
        return self.ctx.status

    def cancel(self):
        self.ctx.status = "cancelled"

    def history(self) -> list[StepResult]:
        return []
