from agents.base_agent import BaseAgent


class ContextAgent(BaseAgent):
    """
    情景提供Agent：
    当回答信息不足时，提供原始视频情景描述，帮助LLM判断
    """

    def __init__(self, scenario_description: str):
        super().__init__(name="ContextAgent")
        self.scenario = scenario_description

    def process(self, answer_text: str) -> dict:
        """
        判断是否需要补充情景信息

        返回：
        {
            "needs_context": True/False,
            "reason": "为什么需要情景",
            "enhanced_prompt": "增强后的提示词（如果需要）"
        }
        """
        # 检测"信息不足"的特征
        low_info_signals = [
            len(answer_text) < 10,  # 过短
            answer_text.count("，") == 0 and answer_text.count("。") == 0,  # 没有标点
            not any(
                keyword in answer_text
                for keyword in ["以为", "想", "没发现", "拿", "看"]
            ),  # 没有关键动词
        ]

        if any(low_info_signals):
            enhanced_prompt = f"""
# 原始情景（帮助你理解背景）
{self.scenario}

# 被试的回答
"{answer_text}"

# 评分提示
这个回答虽然简短，但请结合原始情景判断：
1. 如果回答暗示了"误以为猫是围巾"这个错误信念 → 2分
2. 如果只是提到了想法或意图，但没有明确错误信念 → 1分
3. 如果只是描述物理动作 → 0分

即使回答很短，也要尽量从情景中推断其含义，而不是轻易标记为"信息不足"。
"""
            return {
                "needs_context": True,
                "reason": "回答信息量较少，需要情景辅助",
                "enhanced_prompt": enhanced_prompt,
            }

        return {
            "needs_context": False,
            "reason": None,
            "enhanced_prompt": None,
        }
