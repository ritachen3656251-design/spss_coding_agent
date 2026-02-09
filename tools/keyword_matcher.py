import re
import config


class KeywordMatcher:
    """
    当LLM不确定时，使用关键词匹配作为回退策略
    """

    def __init__(self):
        # 各分数对应的关键词模式
        self.patterns = {
            2: [
                r"以为",
                r"没发现",
                r"拿错",
                r"看错",
                r"误以为",
                r"当成.*围巾",
                r"把.*猫.*当.*围巾",
            ],
            1: [
                r"想",
                r"打算",
                r"为了还",
                r"颜色.*像",
                r"位置.*近",
            ],
            0: [
                r"捡起.*猫",
                r"骗人",
                r"吓唬",
                r"偷",
            ],
        }

    def match(self, text: str) -> dict:
        """
        基于关键词匹配打分

        返回：
        {
            "score": 0/1/2/None,
            "confidence": 0.0-1.0,
            "matched_patterns": ["匹配到的关键词"]
        }
        """
        text = text.strip()

        # 999 仅两种情形：①完全空白（字数严格为0）②包含配置中的无效关键词。
        # 标点、数字、符号、无效关键词等均不算 999。
        if len(text) == 0:
            return {"score": 999, "confidence": 1.0, "matched_patterns": ["空文本"]}

        for keyword in config.AUTO_999_KEYWORDS:
            if keyword in text:
                return {"score": 999, "confidence": 1.0, "matched_patterns": [keyword]}

        # 按优先级匹配（2分 > 1分 > 0分）
        for score in [2, 1, 0]:
            matched = []
            for pattern in self.patterns[score]:
                if re.search(pattern, text):
                    matched.append(pattern)

            if matched:
                # 匹配到的模式越多，置信度越高
                confidence = min(0.6 + len(matched) * 0.1, 0.9)
                return {
                    "score": score,
                    "confidence": confidence,
                    "matched_patterns": matched,
                }

        # 无法匹配
        return {"score": None, "confidence": 0.0, "matched_patterns": []}
