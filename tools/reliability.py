import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


class ReliabilityAnalyzer:
    """
    信度分析工具：
    - Cohen's Kappa（评分者间一致性）
    - 混淆矩阵
    - 一致性百分比
    """

    def __init__(self):
        pass

    def calculate_kappa(self, human_scores: list, ai_scores: list) -> dict:
        """
        计算Cohen's Kappa系数

        参数：
        - human_scores: 人工打分列表
        - ai_scores: AI打分列表

        返回：
        {
            "kappa": 0.0-1.0,
            "interpretation": "一致性解释",
            "agreement_rate": 百分比
        }
        """
        valid_pairs = [
            (h, a)
            for h, a in zip(human_scores, ai_scores)
            if h != 999 and a != 999
        ]

        if len(valid_pairs) < 10:
            return {"error": "有效样本太少，无法计算信度"}

        h_valid, a_valid = zip(*valid_pairs)

        kappa = cohen_kappa_score(h_valid, a_valid)
        agreement = sum(1 for h, a in zip(h_valid, a_valid) if h == a) / len(
            valid_pairs
        )

        if kappa > 0.8:
            interp = "几乎完全一致（Excellent）"
        elif kappa > 0.6:
            interp = "高度一致（Substantial）"
        elif kappa > 0.4:
            interp = "中度一致（Moderate）"
        elif kappa > 0.2:
            interp = "一般一致（Fair）"
        else:
            interp = "一致性较低（Poor）"

        return {
            "kappa": round(kappa, 3),
            "interpretation": interp,
            "agreement_rate": round(agreement * 100, 2),
        }

    def generate_confusion_matrix(
        self, human_scores: list, ai_scores: list
    ) -> pd.DataFrame:
        """生成混淆矩阵"""
        valid_pairs = [
            (h, a)
            for h, a in zip(human_scores, ai_scores)
            if h != 999 and a != 999
        ]
        h_valid, a_valid = zip(*valid_pairs)

        cm = confusion_matrix(h_valid, a_valid, labels=[0, 1, 2])

        return pd.DataFrame(
            cm,
            index=[f"人工{i}分" for i in [0, 1, 2]],
            columns=[f"AI{i}分" for i in [0, 1, 2]],
        )
