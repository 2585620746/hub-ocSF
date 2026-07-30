---
name: skill-harness
description: >-
  渐进式 Skill 加载与执行系统。解析符合 SKILL.md 格式的技能定义文件，按步骤逐步执行。
  Use when the user wants to discover and run registered skills from the skills directory,
  e.g. "运行技能"、"执行 skill"、"帮我做张闪卡"。
---

# Skill Harness — 渐进式技能执行器

从 `SKILL.md` 文件中加载技能定义，按步骤逐步执行，支持在步骤间暂停等待用户输入。

## 触发场景

当用户说出类似下面的话时触发本 harness：
- "运行技能 harness"
- "帮我执行一个 skill"
- "有哪些 skill 可以用"
- 或任何与已注册 skill 触发模式匹配的语句

## 文件结构

```
harness/
├── SKILL.md              # Harness 自描述
├── skill_loader.py       # 解析 SKILL.md → Skill 对象
├── skill_executor.py     # 渐进式执行引擎
├── skill_cli.py          # CLI 交互入口
└── data/                 # 运行时数据
```

## 执行流程

### Step 1: 发现技能
`SkillRegistry.scan(directory)` → 递归扫描所有 `SKILL.md` → 解析 YAML frontmatter + markdown 步骤 → 注册到内存

### Step 2: 匹配触发
用户输入 → `registry.match_trigger(query)` → 按 trigger_patterns 关键词匹配 → 返回候选 skill 列表

### Step 3: 选择技能
用户从候选列表中确认要执行的 skill

### Step 4: 渐进式执行
`ProgressiveExecutor.start()` → 创建 ExecutionContext → 逐步骤执行：

| 步骤动作 | 说明 | 渐进特性 |
|---------|------|---------|
| `describe` | 打印步骤说明，不做操作 | 自动推进 |
| `identify` | 从输入提取信息（如单词） | 缺信息时暂停等待 |
| `generate` | 创建数据文件 | 自动推进 |
| `run_python` | 执行 Python 脚本 | 自动推进，异常时暂停 |
| `run_command` | 执行 shell 命令 | 自动推进 |
| `open_file` | 在浏览器/系统中打开文件 | 自动推进 |
| `ask_user` | 等待用户确认或输入 | **暂停等待** |

### Step 5: 完成
全部步骤执行完毕 → 状态设为 `completed` → 打印摘要

## SKILL.md 格式要求

```yaml
---
name: skill-name
description: >-
  Use when the user says "..."
  e.g. "给我做张闪卡"、"..."
---
```

### 步骤定义

正文中用 `### Step N: 动作描述` 标记每一步，可选紧跟 ` ```bash ` 代码块指定要执行的命令。

```markdown
### Step 1: 识别单词
从用户话语中提取目标英语单词。

### Step 2: 生成 JSON 数据
```bash
python scripts/make_data.py data/<word>.json
```

### Step 3: 打开预览
在浏览器中打开生成的 HTML 文件。
```

## 注意事项

- 每个 skill 目录需要包含 `SKILL.md` 文件
- 步骤中的 `<word>`、`<json_path>` 等变量由执行器自动替换
- `identify` 动作默认从输入中提取最后一个 ASCII 单词
- `ask_user` 动作用于需要用户确认的步骤，执行器会暂停等待
