"""
核心打分 Agent：主模型 Qwen，低置信时用 DeepSeek 与关键词回退验证
"""
import os

from agents.base_agent import BaseAgent
from tools.llm_client import LLMClient
from tools.keyword_matcher import KeywordMatcher
import config


class ScoringAgent(BaseAgent):
    """
    核心打分 Agent：
    1. 主模型（Qwen）打分
    2. 评估置信度
    3. 低置信度时启用多策略验证（DeepSeek + 关键词回退）
    """

    def __init__(self, rubric: str, scenario: str = None):
        super().__init__(name="ScoringAgent", llm_client=LLMClient(model_type="qwen"))
        self.rubric = rubric
        self.scenario = scenario
        has_deepseek_key = config.DEEPSEEK_API_KEY or os.getenv("OPENAI_API_KEY")
        self.secondary_client = (
            LLMClient(model_type="deepseek")
            if (config.ENABLE_DUAL_MODEL and has_deepseek_key)
            else None
        )
        self.keyword_matcher = (
            KeywordMatcher() if config.ENABLE_KEYWORD_FALLBACK else None
        )
        if scenario:
            from agents.context_agent import ContextAgent

            self.context_agent = ContextAgent(scenario)
        else:
            self.context_agent = None

    def process(self, text: str) -> dict:
        """
        智能打分流程

        返回：
        {
            "score": 0/1/2,
            "confidence": 0.0-1.0,
            "reasoning": "判分理由（高置信时为空）",
            "strategy": "primary/dual_model/keyword_fallback",
            "details": {...}
        }
        """
        primary_result = self._score_with_primary_model(text)

        if primary_result["confidence"] >= config.CONFIDENCE_THRESHOLD:
            self.log(f"主模型高置信度 ({primary_result['confidence']:.2f})，直接采用")
            return {
                "score": primary_result["score"],
                "confidence": primary_result["confidence"],
                "reasoning": "",  # 高置信不输出判断理由，只给分数
                "strategy": "primary",
                "details": {"primary": primary_result},
            }

        self.log(f"主模型置信度低 ({primary_result['confidence']:.2f})，启用多策略验证")
        return self._handle_boundary_case(text, primary_result)

    def _score_with_primary_model(self, text: str) -> dict:
        """
        使用主模型打分，必要时提供情景
        """
        if self.context_agent:
            context_check = self.context_agent.process(text)
            if context_check["needs_context"]:
                self.log(f"检测到信息不足，提供情景辅助")
                user_message = (
                    context_check["enhanced_prompt"] + f"\n\n{self.rubric}"
                )
            else:
                user_message = f"""# 评分标准
{self.rubric}

# 待评分回答
"{text}"

请评分并说明你的置信度。"""
        else:
            user_message = f"""# 评分标准
{self.rubric}

# 待评分回答
"{text}"

请评分并说明你的置信度。"""

        system_prompt = """你是严格的心理学问卷编码专家。

你必须输出JSON格式：
{
    "score": 0/1/2,
    "confidence": 0.0-1.0,
    "reasoning": "判分理由",
    "key_evidence": ["支持该分数的关键证据"]
}

confidence说明：
- 0.9-1.0: 非常确定，有明确关键词
- 0.7-0.89: 比较确定，逻辑清晰
- 0.5-0.69: 有一定依据，但存在模糊性
- 0.3-0.49: 倾向某个分数，但不太确定
- 0.0-0.29: 完全无法判断

重要：即使信息较少，也要基于情景和上下文给出最合理的判断，不要轻易说"信息不足"。

只返回JSON，不要其他文字。"""

        result = self.call_llm_with_json(system_prompt, user_message)

        if not result["success"]:
            return {
                "score": None,
                "confidence": 0.0,
                "reasoning": "LLM调用失败",
                "key_evidence": []
            }

        return result["data"]

    def _handle_boundary_case(self, text: str, primary_result: dict) -> dict:
        """处理边界 case：多策略验证"""
        strategies_used = ["primary"]
        all_results = {"primary": primary_result}

        if config.ENABLE_DUAL_MODEL and self.secondary_client:
            self.log("调用第二模型（DeepSeek）验证...")
            secondary_result = self._score_with_secondary_model(text)
            all_results["secondary"] = secondary_result
            strategies_used.append("dual_model")

            if secondary_result["score"] == primary_result["score"]:
                self.log("双模型结果一致，提升置信度")
                return {
                    "score": primary_result["score"],
                    "confidence": min(primary_result["confidence"] + 0.2, 0.95),
                    "reasoning": f"双模型一致判定：{primary_result.get('reasoning', '')}",
                    "strategy": "dual_model_consensus",
                    "details": all_results,
                }

            self.log(
                f"双模型不一致：Qwen={primary_result['score']}, DeepSeek={secondary_result['score']}"
            )

        if config.ENABLE_KEYWORD_FALLBACK and self.keyword_matcher:
            self.log("使用关键词匹配作为回退策略...")
            keyword_result = self.keyword_matcher.match(text)
            all_results["keyword"] = keyword_result
            strategies_used.append("keyword_fallback")

            if keyword_result["score"] == primary_result["score"]:
                self.log("关键词匹配与主模型一致")
                return {
                    "score": primary_result["score"],
                    "confidence": max(
                        primary_result["confidence"], keyword_result["confidence"]
                    ),
                    "reasoning": f"{primary_result.get('reasoning', '')} (关键词验证: {keyword_result['matched_patterns']})",
                    "strategy": "keyword_consensus",
                    "details": all_results,
                }

        self.log("多策略验证后仍不确定，标记为需人工复核")
        return {
            "score": primary_result["score"],
            "confidence": primary_result["confidence"],
            "reasoning": primary_result.get("reasoning", "") + " [需人工复核]",
            "strategy": "uncertain",
            "details": all_results,
            "needs_human_review": True,
        }

    def _score_with_secondary_model(self, text: str) -> dict:
        """使用第二模型（DeepSeek）打分"""
        original_client = self.llm_client
        self.llm_client = self.secondary_client
        result = self._score_with_primary_model(text)
        self.llm_client = original_client
        return result


def load_rubric(file_path: str = "prompts/scoring_rubric.txt") -> str:
    """
    读取评分标准文件
    返回：评分标准的字符串内容
    """
    encodings = ("utf-8", "gbk")
    last_error = None
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError as e:
            last_error = e
    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        f"无法使用 {encodings} 解码文件 {file_path}",
    )
