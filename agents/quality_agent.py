from agents.base_agent import BaseAgent


class QualityAgent(BaseAgent):
    """
    质量检查 Agent：
    1. 检查打分结果的合理性
    2. 标记异常 case
    3. 生成质量报告
    """

    def __init__(self):
        super().__init__(name="QualityAgent")
        self.low_confidence_cases = []
        self.conflict_cases = []

    def process(self, scoring_result: dict, text: str, row_id: str) -> dict:
        """
        质量检查单条结果

        返回：增强的结果字典
        """
        result = scoring_result.copy()
        result["quality_flags"] = []

        # 检查1：置信度过低
        if scoring_result.get("confidence", 1.0) < 0.5:
            result["quality_flags"].append("LOW_CONFIDENCE")
            self.low_confidence_cases.append({
                "id": row_id,
                "text": text,
                "result": scoring_result,
            })

        # 检查2：双模型冲突
        if "details" in scoring_result and "secondary" in scoring_result["details"]:
            primary_score = scoring_result["details"]["primary"]["score"]
            secondary_score = scoring_result["details"]["secondary"]["score"]
            if primary_score != secondary_score:
                result["quality_flags"].append("MODEL_CONFLICT")
                self.conflict_cases.append({
                    "id": row_id,
                    "text": text,
                    "primary": primary_score,
                    "secondary": secondary_score,
                })

        # 检查3：需要人工复核
        if scoring_result.get("needs_human_review"):
            result["quality_flags"].append("NEEDS_REVIEW")

        return result

    def generate_quality_report(self) -> dict:
        """
        生成质量报告
        """
        return {
            "low_confidence_count": len(self.low_confidence_cases),
            "conflict_count": len(self.conflict_cases),
            "low_confidence_cases": self.low_confidence_cases[:10],  # 只返回前10个
            "conflict_cases": self.conflict_cases,
        }
