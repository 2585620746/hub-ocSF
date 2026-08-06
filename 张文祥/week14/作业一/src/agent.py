import os
from openai import OpenAI
from skill_manager import SkillManager


SYSTEM_TEMPLATE = """你是电商平台的智能客服助手。

你的所有知识来源于以下技能文档，严格基于文档内容回答，不要自行推断或编造政策。

## 回答规则（严格遵守）
- 【能回答】如果技能文档覆盖了用户问题：直接给出完整具体的答案（含具体天数/金额/
  工作日数等政策细节）。**不要在答案中加"建议联系人工客服"之类的推脱话**。
- 【不能回答】如果技能文档确实不覆盖：**仅回答一句** "需要联系人工客服"，
  不要编造答案，也不要列举可能的情况。

{skills_section}
"""

SKILLS_SECTION_TEMPLATE = """## 当前知识库（共{count}个技能）

{skills_content}
"""


class CustomerServiceAgent:
    def __init__(
        self,
        skill_manager: SkillManager,
        nudge_interval: int = 20,
        model: str = "deepseek-chat",
    ):
        self.skill_manager = skill_manager
        self.nudge_interval = nudge_interval
        self.model = model
        self._iters_since_nudge = 0
        self.conversation_history: list[dict] = []

        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    def answer(self, question: str) -> str:
        """
        回答单个问题。每次调用都会重新加载最新 Skills（保证 Nudge 后立即生效），
        且 messages 里只含系统提示 + 当前问题（不携带 conversation_history），
        这样每次评估互不干扰。
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=400,
        )
        answer_text = response.choices[0].message.content.strip()

        self.conversation_history.append({
            "question": question,
            "answer": answer_text,
            "skills_used": list(self.skill_manager.load_all().keys()),
        })
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]

        self._iters_since_nudge += 1
        return answer_text

    def should_trigger_nudge(self) -> bool:
        """判断是否应该触发后台回顾。触发后计数器归零。"""
        if self.nudge_interval > 0 and self._iters_since_nudge >= self.nudge_interval:
            self._iters_since_nudge = 0
            return True
        return False

    def reset_nudge_counter(self):
        """Agent 主动调用 skill_manage 时手动归零，避免重复触发。"""
        self._iters_since_nudge = 0

    def _build_system_prompt(self) -> str:
        skills = self.skill_manager.load_all()
        if not skills:
            skills_section = "（暂无技能文档，请依据通用客服原则回答）"
        else:
            parts = []
            for name, content in sorted(skills.items()):
                parts.append(f"### 技能：{name}\n{content}")
            skills_content = "\n\n---\n\n".join(parts)
            skills_section = SKILLS_SECTION_TEMPLATE.format(
                count=len(skills),
                skills_content=skills_content,
            )
        return SYSTEM_TEMPLATE.format(skills_section=skills_section)
