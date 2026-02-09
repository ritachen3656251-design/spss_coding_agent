import json
import os
from collections import Counter
from difflib import SequenceMatcher

# 历史策略 → 执行层 plan key（用于相似经验检索后的决策）
STRATEGY_TO_PLAN_KEY = {
    "primary": "standard_scoring",
    "dual_model_consensus": "prioritize_dual_model",
    "secondary_higher_confidence": "prioritize_dual_model",
    "context_enhanced": "use_context",
    "keyword_consensus": "standard_scoring",
    "primary_higher_confidence": "standard_scoring",
    "uncertain": None,
}

# ScoringAgent 返回的 strategy → strategy_performance 的 key
STRATEGY_TO_PERF_KEY = {
    "primary": "primary_only",
    "dual_model_consensus": "dual_model",
    "keyword_consensus": "keyword_fallback",
    "uncertain": "dual_model",
    "auto_999": None,
    "context_enhanced": "primary_only",
    "primary_higher_confidence": "primary_only",
    "secondary_higher_confidence": "dual_model",
}


class AgentMemory:
    """
    Agent记忆系统：
    - 短期记忆：当前会话中的关键决策
    - 长期记忆：历史处理经验
    """

    def __init__(self, memory_file="data/cache/agent_memory.json"):
        self.memory_file = memory_file
        self.short_term = {
            "difficult_cases": [],
            "strategy_performance": {
                "primary_only": {"count": 0, "success": 0},
                "dual_model": {"count": 0, "success": 0},
                "keyword_fallback": {"count": 0, "success": 0},
            },
        }
        self.long_term = self._load_long_term()

    def _load_long_term(self) -> dict:
        """加载长期记忆"""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("experience_cases", [])
            return data
        return {
            "total_processed": 0,
            "common_patterns": {},
            "edge_cases": [],
            "experience_cases": [],
        }

    def save_long_term(self):
        """保存长期记忆"""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.long_term, f, ensure_ascii=False, indent=2)

    def record_decision(
        self, text: str, strategy: str, score: int, confidence: float
    ):
        """记录每次决策。strategy 为 ScoringAgent 返回的原始值，内部会自动映射"""
        perf_key = STRATEGY_TO_PERF_KEY.get(strategy, strategy)
        if perf_key and perf_key in self.short_term["strategy_performance"]:
            self.short_term["strategy_performance"][perf_key]["count"] += 1

        if confidence < 0.6:
            self.short_term["difficult_cases"].append(
                {
                    "text": text[:50],
                    "strategy": strategy,
                    "score": score,
                    "confidence": confidence,
                }
            )

    def add_case(self, case: dict):
        """存储一条经验，用于相似检索"""
        cases = self.long_term.setdefault("experience_cases", [])
        cases.append({
            "text": case.get("text", "")[:100],
            "strategy": case.get("strategy", ""),
            "score": case.get("score"),
            "confidence": case.get("confidence", 0),
        })
        self.long_term["experience_cases"] = cases[-500:]

    def retrieve_similar(self, text: str, top_k: int = 5) -> list:
        """检索与当前文本相似的历史经验"""
        cases = self.long_term.get("experience_cases", [])
        if not cases:
            return []

        def similarity(a: str, b: str) -> float:
            return SequenceMatcher(None, a, b).ratio()

        scored = [(c, similarity(text[:100], c.get("text", ""))) for c in cases]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, s in scored[:top_k] if s > 0.3]

    def get_best_strategy_from_cases(self, cases: list) -> str | None:
        """从相似案例中选出最常见的有效策略，返回执行层 plan key"""
        if not cases:
            return None
        strategies = [
            STRATEGY_TO_PLAN_KEY.get(c["strategy"], "standard_scoring")
            for c in cases
            if c.get("strategy") and STRATEGY_TO_PLAN_KEY.get(c["strategy"])
        ]
        if not strategies:
            return None
        return Counter(strategies).most_common(1)[0][0]

    def get_strategy_advice(self) -> str:
        """基于记忆给出策略建议"""
        perf = self.short_term["strategy_performance"]

        if perf["dual_model"]["count"] > 5:
            success_rate = perf["dual_model"]["success"] / perf["dual_model"][
                "count"
            ]
            if success_rate > 0.8:
                return "dual_model_recommended"

        return "default"
