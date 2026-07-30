"""
---
name: skill-loader
description: >-
  解析 SKILL.md 文件，提取 YAML frontmatter（name/description）+ 执行步骤。
  支持自由格式 markdown 步骤抽取 + 结构化 YAML steps 扩展字段。
---

# skill_loader — SKILL.md 解析器

## 触发/使用场景

- 系统启动时扫描 skill 目录，加载所有 SKILL.md
- 用户请求执行某个 skill 时按需解析

## 执行流程

1. 读取 SKILL.md → 分离 YAML frontmatter 和正文
2. 提取 name / description / trigger_patterns / steps 字段
3. 从 markdown 正文中按 `### Step N:` 抽取步骤定义
4. 返回 Skill 结构化对象

## 数据格式

```yaml
---
name: skill-name
description: >-
  Use when the user says "..."
---
```

## 使用方式

```python
from skill_loader import SkillRegistry
registry = SkillRegistry()
registry.scan("K:/week13/skills")
skill = registry.find_by_name("flash-card")
skill = registry.match_trigger("给我做张 crazy 的闪卡")
```
"""

import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Step:
    order: int
    action: str
    description: str
    command: Optional[str] = None
    params: dict = field(default_factory=dict)


@dataclass
class Skill:
    name: str
    description: str
    trigger_patterns: list[str]
    steps: list[Step]
    skill_dir: Path
    raw_body: str = ""


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_STEP_HDR_RE = re.compile(r"^###\s+Step\s+(\d+)\s*[：:]\s*(.+?)$", re.MULTILINE)
_STEP_NUM_RE = re.compile(r"^\s*(\d+)\s*[.、．]\s*(\*{0,2})(.+?)\2\s*(?:[：:：]|$)", re.MULTILINE)
_COMMAND_BLOCK_RE = re.compile(r"```(?:bash)?\s*\n(.+?)```", re.DOTALL)


def parse_skill_md(path: Path) -> Optional[Skill]:
    content = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return None

    frontmatter = yaml.safe_load(m.group(1))
    body = content[m.end():]

    name = frontmatter.get("name", path.parent.name)
    desc = frontmatter.get("description", "")
    steps = []

    # Split body by step headers into step-specific sections
    step_matches = list(_STEP_HDR_RE.finditer(body))
    if not step_matches:
        step_matches = list(_STEP_NUM_RE.finditer(body))

    for i, sm in enumerate(step_matches):
        order = int(sm.group(1))
        desc_text = sm.group(3).strip() if sm.lastindex >= 3 else sm.group(2).strip()
        desc_text = desc_text.split("：", 1)[0].split(":", 1)[0].strip()  # strip follow-up text after colon

        # Only search command block in the section between this step and next step
        section_end = step_matches[i + 1].start() if i + 1 < len(step_matches) else len(body)
        cm = _COMMAND_BLOCK_RE.search(body, sm.end(), section_end)
        command = cm.group(1).strip() if cm else None

        action = _infer_action(desc_text, command)
        steps.append(Step(order=order, action=action, description=desc_text, command=command))

    return Skill(
        name=name,
        description=desc,
        trigger_patterns=_extract_triggers(desc, body),
        steps=steps,
        skill_dir=path.parent,
        raw_body=body,
    )


def _infer_action(desc: str, command: str | None) -> str:
    if command:
        if command.startswith("python"):
            return "run_python"
        if any(cmd in command for cmd in ("start ", "open ", "xdg-open")):
            return "open_file"
        return "run_command"
    if "识别" in desc or "提取" in desc:
        return "identify"
    if "生成" in desc or "创建" in desc or "保存" in desc:
        return "generate"
    if "预览" in desc or "打开" in desc:
        return "open_file"
    if "询问" in desc or "确认" in desc:
        return "ask_user"
    return "describe"


def _extract_triggers(desc: str, body: str) -> list[str]:
    triggers = []
    trigger_section = re.search(r"##\s*触发场景(.*?)(?=^##)", body, re.DOTALL | re.MULTILINE)
    if trigger_section:
        triggers += re.findall(r'"(.+?)"', trigger_section.group(1))
    if not triggers and desc:
        triggers.append(desc)
    return triggers


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def scan(self, directory: str | Path) -> list[Skill]:
        found = []
        for md_path in Path(directory).rglob("SKILL.md"):
            skill = parse_skill_md(md_path)
            if skill:
                self._skills[skill.name] = skill
                found.append(skill)
        return found

    def find_by_name(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def match_trigger(self, query: str) -> list[Skill]:
        results = []
        for skill in self._skills.values():
            for pattern in skill.trigger_patterns:
                if any(kw in query for kw in _keywords_from(pattern)):
                    results.append(skill)
                    break
        return results

    @property
    def skills(self) -> list[Skill]:
        return list(self._skills.values())


def _keywords_from(pattern: str) -> list[str]:
    import re
    return re.findall(r"[\w\u4e00-\u9fff]+", pattern)
