from agents.base_agent import BaseAgent
import pandas as pd


class AnalysisAgent(BaseAgent):
    """
    分析 Agent：
    1. 统计分数分布
    2. 分析高分/低分案例的共性
    3. 生成可读性报告
    """

    def __init__(self):
        super().__init__(name="AnalysisAgent")

    def process(self, scored_df: pd.DataFrame) -> dict:
        """
        分析整个数据集

        返回：分析报告字典
        """
        self.log("开始数据分析...")

        stats = self._basic_stats(scored_df)
        semantic_analysis = self._semantic_analysis(scored_df)
        report = self._generate_report(stats, semantic_analysis)

        return {
            "stats": stats,
            "semantic_analysis": semantic_analysis,
            "report": report,
        }

    def _basic_stats(self, df: pd.DataFrame) -> dict:
        """基础统计"""
        score_col = "编码分数"

        low_confidence_count = 0
        if "质量标记" in df.columns:
            low_confidence_count = int(
                df["质量标记"].astype(str).str.contains("LOW_CONFIDENCE", na=False).sum()
            )

        valid_subset = df[df[score_col].isin([0, 1, 2])]
        non_999 = df[df[score_col] != 999]
        mean_score = non_999[score_col].mean() if len(non_999) > 0 else None

        return {
            "total_count": len(df),
            "score_distribution": df[score_col].value_counts().to_dict(),
            "valid_count": len(valid_subset),
            "missing_count": len(df[df[score_col] == 999]),
            "mean_score": mean_score,
            "low_confidence_count": low_confidence_count,
        }

    def _semantic_analysis(self, df: pd.DataFrame) -> dict:
        """
        语义分析：找出各分数段的典型特征
        """
        analysis = {}

        if "编码分数" not in df.columns or "回答内容" not in df.columns:
            return analysis

        for score in [0, 1, 2]:
            subset = df[df["编码分数"] == score]["回答内容"].dropna().astype(str).tolist()
            if not subset:
                continue

            sample_texts = subset[:20]
            display_texts = sample_texts[:10]

            system_prompt = f"""你是数据分析专家。请分析以下被打{score}分的回答，总结它们的共同特征。

输出JSON格式：
{{
    "common_patterns": ["特征1", "特征2"],
    "typical_examples": ["典型例子1", "典型例子2"]
}}"""

            user_message = (
                f"这些回答都被打了{score}分：\n"
                + "\n".join([f"- {t[:200]}" for t in display_texts])
            )

            result = self.call_llm_with_json(system_prompt, user_message)

            if result["success"] and result.get("data"):
                analysis[f"score_{score}"] = result["data"]

        return analysis

    def _generate_report(self, stats: dict, semantic: dict) -> str:
        """
        生成 Markdown 格式的报告
        """
        total = stats["total_count"]
        if total == 0:
            return "# 问卷编码分析报告\n\n无数据。"

        valid_count = stats["valid_count"]
        missing_count = stats["missing_count"]
        mean_score = stats["mean_score"]
        mean_str = f"{mean_score:.2f}" if mean_score is not None and pd.notna(mean_score) else "N/A"

        report = f"""# 问卷编码分析报告

## 一、基础统计

- 总样本数：{total}
- 有效样本：{valid_count} ({valid_count / total * 100:.1f}%)
- 缺失值(999)：{missing_count} ({missing_count / total * 100:.1f}%)
- 平均分：{mean_str}

### 分数分布
"""
        for score, count in sorted(stats["score_distribution"].items()):
            pct = count / total * 100
            report += f"- {score}分：{count}个 ({pct:.1f}%)\n"

        report += f"\n## 二、质量控制\n\n"
        report += f"- 低置信度样本：{stats['low_confidence_count']}个\n"

        report += f"\n## 三、语义分析\n\n"
        for score_key, analysis in semantic.items():
            score = score_key.split("_")[1]
            report += f"### {score}分样本特征\n"
            if isinstance(analysis, dict) and "common_patterns" in analysis:
                for pattern in analysis["common_patterns"]:
                    report += f"- {pattern}\n"
            report += "\n"

        return report
