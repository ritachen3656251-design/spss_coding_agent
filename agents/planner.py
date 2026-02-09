from agents.base_agent import BaseAgent
from agents.memory import AgentMemory


# LLM 输出的工具名 → 执行层识别的 key
PLAN_TOOL_ALIAS = {
    "context_enhanced": "use_context",
    "dual_model": "prioritize_dual_model",
}


class PlanningAgent(BaseAgent):
    """
    规划Agent：由 LLM 动态推理制定处理计划
    """

    def __init__(self, memory: AgentMemory):
        super().__init__(name="PlanningAgent")
        self.memory = memory

    def process(self, text: str, preliminary_analysis: dict) -> dict:
        """
        由 LLM 制定处理计划，失败时回退到规则
        """
        result = self._llm_plan(text, preliminary_analysis)
        if result is not None:
            return result
        return self._rule_based_plan(text, preliminary_analysis)

    def _llm_plan(self, text: str, preliminary_analysis: dict) -> dict | None:
        """LLM 动态规划"""
        system_prompt = """你是规划专家，负责为问卷编码系统制定处理计划。

可选工具：
- standard_scoring: 标准打分，适用于信息充足、表述清晰地回答
- context_enhanced: 加入情景描述辅助，适用于文本过短或信息不足时
- dual_model: 调用第二模型验证，适用于表述模糊、模棱两可的回答
- keyword_fallback: 先用关键词匹配辅助判断，适用于含特定关键词的回答

请根据回答内容选择合适的工具组合，输出 JSON：
{
    "plan": ["tool1", "tool2"],
    "reasoning": "为什么这样安排"
}

只返回 JSON，不要其他文字。plan 中工具按优先级排列，至少包含一个。"""

        obs_summary = preliminary_analysis.get("summary", "")
        user_message = f"""当前观察：{obs_summary}

问卷回答："{text[:200]}{'...' if len(text) > 200 else ''}"

请制定处理计划。"""

        out = self.call_llm_with_json(system_prompt, user_message)
        if not out["success"] or not out.get("data"):
            return None

        data = out["data"]
        plan = data.get("plan")
        reasoning = data.get("reasoning", "")

        if not isinstance(plan, list) or not plan:
            return None

        # 规范化为执行层可识别的 key
        normalized = []
        for tool in plan:
            normalized.append(PLAN_TOOL_ALIAS.get(tool, tool))

        return {"plan": normalized, "reasoning": reasoning}

    def _rule_based_plan(self, text: str, preliminary_analysis: dict) -> dict:
        """规则兜底：LLM 失败时使用"""
        plan = []
        reasoning = []

        if len(text) < 10:
            plan.append("use_context")
            reasoning.append("文本过短，需要情景信息")

        ambiguous_words = ["想", "可能", "好像", "应该"]
        if any(word in text for word in ambiguous_words):
            plan.append("prioritize_dual_model")
            reasoning.append("包含模糊表述，可能需要多模型验证")

        strategy_advice = self.memory.get_strategy_advice()
        if strategy_advice == "dual_model_recommended":
            plan.append("prioritize_dual_model")
            reasoning.append("历史数据显示双模型策略效果好")

        if preliminary_analysis.get("is_boundary"):
            plan.append("record_to_memory")
            reasoning.append("边界case，需记录以改进未来决策")

        if not plan:
            plan = ["standard_scoring"]
            reasoning = ["常规打分流程"]

        return {
            "plan": plan,
            "reasoning": " → ".join(reasoning),
        }
