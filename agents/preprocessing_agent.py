"""
预处理 Agent：在打分前判断是否为 999（完全空白或无效关键词）
"""
from agents.base_agent import BaseAgent
import config


class PreprocessingAgent(BaseAgent):
    """
    预处理 Agent：判定 999 后直接走 auto_999，否则交给 ScoringAgent。
    """

    def __init__(self):
        super().__init__(name="PreprocessingAgent")

    def process(self, text: str) -> dict:
        """
        返回：{"is_999": bool, "text": str, "reason": str}
        - is_999 为 True 时不再调用打分，流水线直接写 999
        """
        text = (text or "").strip()

        if len(text) == 0:
            return {"is_999": True, "text": text, "reason": "空文本"}

        for keyword in config.AUTO_999_KEYWORDS:
            if keyword in text:
                return {"is_999": True, "text": text, "reason": keyword}

        return {"is_999": False, "text": text, "reason": ""}
