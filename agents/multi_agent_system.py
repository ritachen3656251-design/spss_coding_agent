"""
多Agent协作系统：专家分工 + 协调者分配
"""
from agents.base_agent import BaseAgent
from agents.scoring_agent import ScoringAgent


class SpecialistAgent(BaseAgent):
    """
    专家Agent基类
    """

    def __init__(self, name: str, expertise: str, rubric: str):
        super().__init__(name=name)
        self.expertise = expertise
        self.rubric = rubric
        self.performance_stats = {"total": 0, "success": 0}

    def can_handle(self, text: str, context: dict) -> float:
        """
        判断是否能处理这个case（返回置信度）

        子类需要重写
        """
        raise NotImplementedError

    def score(self, text: str) -> dict:
        """
        打分（带专业知识）
        """
        specialized_prompt = f"""你是{self.expertise}专家。

评分标准：
{self.rubric}

你的专长：{self.expertise}

回答：{text}

请根据你的专业经验打分。
输出JSON：{{"score": X, "confidence": 0.X, "reasoning": "..."}}"""

        result = self.call_llm_with_json(
            "你是专家评分者，专注于特定类型的case。",
            specialized_prompt,
        )

        self.performance_stats["total"] += 1

        data = result.get("data")
        if data and result.get("success"):
            self.performance_stats["success"] += 1
        if data and data.get("score") is not None:
            try:
                data["score"] = int(float(data["score"]))
            except (ValueError, TypeError):
                pass
        return data if data else {"score": None, "confidence": 0.0}


class ShortTextExpert(SpecialistAgent):
    """短文本专家：擅长处理信息量少的回答"""

    def __init__(self, rubric: str, scenario: str = None):
        super().__init__(
            name="ShortTextExpert",
            expertise="短文本分析，擅长从有限信息推断意图",
            rubric=rubric,
        )
        self.scenario = scenario

    def can_handle(self, text: str, context: dict) -> float:
        if len(text) < 10:
            return 0.9  # 高置信度处理短文本
        elif len(text) < 15:
            return 0.6
        else:
            return 0.2


class AmbiguousTextExpert(SpecialistAgent):
    """模糊文本专家：擅长处理含模糊词汇的回答"""

    def __init__(self, rubric: str):
        super().__init__(
            name="AmbiguousTextExpert",
            expertise="模糊语义分析，擅长从上下文判断真实意图",
            rubric=rubric,
        )
        self.ambiguous_words = ["想", "可能", "好像", "应该", "似乎"]

    def can_handle(self, text: str, context: dict) -> float:
        ambiguous_count = sum(
            1 for word in self.ambiguous_words if word in text
        )

        if ambiguous_count >= 2:
            return 0.9
        elif ambiguous_count == 1:
            return 0.7
        else:
            return 0.3


class BoundaryExpert(SpecialistAgent):
    """边界判断专家：擅长区分1分和2分"""

    def __init__(self, rubric: str):
        super().__init__(
            name="BoundaryExpert",
            expertise="边界case判断，擅长区分心理状态（1分）和错误信念（2分）",
            rubric=rubric,
        )

    def can_handle(self, text: str, context: dict) -> float:
        # 包含"想"但不包含"以为"的case
        has_want = "想" in text or "打算" in text
        has_belief = "以为" in text or "认为" in text

        if has_want and not has_belief:
            return 0.85  # 典型的边界case
        elif has_want and has_belief:
            return 0.75  # 也可能是边界
        else:
            return 0.3


class CoordinatorAgent(BaseAgent):
    """
    协调者Agent：分配任务给最合适的专家
    """

    def __init__(self, rubric: str, scenario: str = None):
        super().__init__(name="CoordinatorAgent")

        # 初始化专家团队
        self.specialists = [
            ShortTextExpert(rubric, scenario),
            AmbiguousTextExpert(rubric),
            BoundaryExpert(rubric),
        ]

        # 通用Agent（兜底）
        self._scoring_agent = ScoringAgent(rubric=rubric, scenario=scenario)

    def process(self, text: str) -> dict:
        """
        协调处理流程
        """
        self.log(f"协调处理：'{text[:30]}...'")

        # 1. 询问所有专家是否能处理
        expert_confidence = []
        for specialist in self.specialists:
            confidence = specialist.can_handle(text, {})
            expert_confidence.append((specialist, confidence))
            self.log(f"  {specialist.name}: 置信度{confidence:.2f}")

        # 2. 选择最有信心的专家
        best_expert, best_confidence = max(
            expert_confidence, key=lambda x: x[1]
        )

        # 3. 如果没有专家有信心，用通用方案
        if best_confidence < 0.5:
            self.log("  → 无专家有信心，使用通用方案")
            return self._general_scoring(text)

        # 4. 让专家处理
        self.log(f"  → 分配给 {best_expert.name}")
        result = best_expert.score(text)
        result["assigned_expert"] = best_expert.name
        result["expert_confidence"] = best_confidence
        result.setdefault("strategy", "expert")

        # 5. 如果专家的结果置信度仍然低，请求第二意见
        if result.get("confidence", 0) < 0.7:
            self.log("  → 专家结果不确定，请求第二意见")
            second_result = self._get_second_opinion(text, best_expert)

            # 如果两个专家一致，提升置信度
            if self._scores_equal(
                result.get("score"), second_result.get("score")
            ):
                result["confidence"] = min(
                    0.9,
                    (result.get("confidence", 0) + second_result.get("confidence", 0))
                    / 2
                    + 0.1,
                )
                result["strategy"] = "expert_consensus"

        return result

    def _scores_equal(self, a, b) -> bool:
        """统一比较分数（支持 int/float/str）"""
        if a is None or b is None:
            return a == b
        try:
            return int(float(a)) == int(float(b))
        except (ValueError, TypeError):
            return False

    def _general_scoring(self, text: str) -> dict:
        """通用打分（兜底方案）"""
        return self._scoring_agent.process(text)

    def _get_second_opinion(self, text: str, first_expert) -> dict:
        """请求第二意见（选择不同的专家）"""
        other_experts = [e for e in self.specialists if e != first_expert]

        if not other_experts:
            return {"score": None, "confidence": 0}

        # 选择第二合适的专家
        second_expert = other_experts[0]
        return second_expert.score(text)

    def get_team_report(self) -> str:
        """生成团队协作报告"""
        report = "## 多Agent协作统计\n\n"

        for specialist in self.specialists:
            stats = specialist.performance_stats
            total = stats["total"]

            if total > 0:
                report += f"### {specialist.name}\n"
                report += f"- 处理案例数：{total}\n"
                report += f"- 专长：{specialist.expertise}\n\n"

        return report
