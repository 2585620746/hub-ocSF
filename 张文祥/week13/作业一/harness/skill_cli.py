"""
---
name: skill-cli
description: >-
  Harness CLI 入口。扫描 skills 目录、匹配用户输入、选择并渐进式执行 skill。
  Use when the user wants to run any registered skill interactively,
  e.g. "帮我做张闪卡"、"运行 skill harness"。
---

# skill_cli — Harness CLI 入口

## 触发/使用场景

- 终端中运行 `python skill_cli.py`
- 自动扫描上级 skills 目录，列出可用 skill
- 用户选择 → 输入触发语句 → 渐进式执行

## 执行流程

1. 扫描 skills 目录加载所有 SKILL.md
2. 列出可用 skills 供用户选择
3. 用户输入触发语句 → registry.match_trigger() 匹配 skill
4. 选中后创建 ProgressiveExecutor → 逐步执行
5. 每步完成后打印结果，需要输入时暂停等待
6. 全部完成后输出摘要

## 使用方式

```bash
cd K:/week13/skills/harness
python skill_cli.py
```
"""

import sys
import logging
from pathlib import Path

HERE = Path(__file__).parent
SKILLS_DIR = HERE.parent

try:
    from skill_loader import SkillRegistry
    from skill_executor import ProgressiveExecutor
except ImportError:
    sys.path.insert(0, str(HERE))
    from skill_loader import SkillRegistry
    from skill_executor import ProgressiveExecutor

logging.basicConfig(level=logging.WARNING)
COLOR = {"cyan": "\033[36m", "green": "\033[32m", "yellow": "\033[33m", "magenta": "\033[35m", "red": "\033[31m", "dim": "\033[2m", "reset": "\033[0m"}


def c(name, text):
    return f"{COLOR[name]}{text}{COLOR['reset']}"


def main():
    registry = SkillRegistry()
    skills = registry.scan(SKILLS_DIR)

    print(f"\n{c('cyan', '═' * 60)}")
    print(f"{c('cyan', '  Skill Harness — 渐进式技能执行器')}")
    print(f"{c('cyan', '═' * 60)}")

    if not skills:
        print(f"{c('yellow', '  未找到 SKILL.md 文件')}")
        return

    print(f"\n  可用 skills ({len(skills)}):\n")
    for i, skill in enumerate(skills, 1):
        trigger_preview = (skill.trigger_patterns[0][:60] + "…") if skill.trigger_patterns else "—"
        print(f"  {c('green', str(i))}. {c('cyan', skill.name)}")
        print(f"     {c('dim', trigger_preview)}")
        steps_count = len(skill.steps)
        print(f"     {steps_count} 个步骤\n")

    try:
        choice = input(f"\n{c('yellow', '选择 skill (输入编号或名称): ')}").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(skills):
            skill = skills[idx]
        else:
            print(f"{c('red', '无效编号')}")
            return
    else:
        skill = registry.find_by_name(choice)
        if not skill:
            print(f"{c('red', f'未找到 skill: {choice}')}")
            return

    print(f"\n{c('cyan', f'已选择: {skill.name}')}")
    print(f"{c('dim', skill.description[:100])}\n")

    user_input = input(f"{c('yellow', '输入触发语句: ')}").strip()
    if not user_input:
        user_input = skill.trigger_patterns[0] if skill.trigger_patterns else ""

    executor = ProgressiveExecutor(skill)
    executor.start(context={"user_input": user_input, "work_dir": str(SKILLS_DIR)})

    print(f"\n{c('cyan', '─' * 60)}")
    print(f"{c('cyan', f'  执行: {skill.name}')}")
    print(f"{c('cyan', '─' * 60)}\n")

    while True:
        result = executor.step()
        if result is None:
            break

        if result.status == "completed":
            print(f"\n{c('green', '✓ 全部步骤完成!')}")
            break

        if result.status == "error":
            print(f"\n{c('red', f'✗ 步骤 {result.step} 出错: {result.output}')}")
            break

        print(f"  {c('green', f'[{result.step}]')} {result.description}")
        if result.output:
            print(f"    {c('dim', result.output[:200])}")

        if result.status == "awaiting_input":
            prompt = result.prompt or "请输入: "
            try:
                inp = input(f"\n  {c('yellow', prompt)}").strip()
                executor.resume(inp)
                print()
            except (KeyboardInterrupt, EOFError):
                executor.cancel()
                print(f"\n{c('yellow', '已取消')}")
                break

    print(f"\n{c('cyan', '═' * 60)}")
    print(f"{c('cyan', '  执行结束')}")
    print(f"{c('cyan', '═' * 60)}\n")


if __name__ == "__main__":
    main()
