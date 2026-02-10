from datetime import datetime

import pandas as pd

import config
from agents.base_agent import BaseAgent
from agents.case_repository import CaseRepository


class AnalysisAgent(BaseAgent):
    """
    分析 Agent：
    1. 统计分数分布
    2. 分析高分/低分案例的共性
    3. 生成可读性报告（含 Agent 学习内容）
    """

    def __init__(self, case_repo: CaseRepository = None):
        super().__init__(name="AnalysisAgent")
        self.case_repo = case_repo or CaseRepository()

    def process(self, scored_df: pd.DataFrame, score_col: str = None) -> dict:
        """
        分析整个数据集，包含学习报告

        score_col: 分数列名，默认 "编码分数"，评估流程可能传入 "AI评分"
        """
        self.log("开始数据分析...")
        score_col = score_col or "编码分数"
        if score_col not in scored_df.columns and "AI评分" in scored_df.columns:
            score_col = "AI评分"

        stats = self._basic_stats(scored_df, score_col)
        semantic_analysis = self._semantic_analysis(scored_df, score_col)

        id_col = config.ID_COL if config.ID_COL in scored_df.columns else (
            "_row_id" if "_row_id" in scored_df.columns else scored_df.columns[0]
        )
        answer_col = config.ANSWER_COL if config.ANSWER_COL in scored_df.columns else (
            "VAR00001" if "VAR00001" in scored_df.columns else scored_df.columns[1]
        )

        all_responses = [
            {
                "id": str(row[id_col]),
                "text": str(row[answer_col]) if pd.notna(row.get(answer_col)) else "",
                "score": row[score_col],
            }
            for _, row in scored_df.iterrows()
            if row[score_col] != 999
        ]

        report = self._generate_enhanced_report(
            stats, semantic_analysis, all_responses
        )

        return {
            "stats": stats,
            "semantic_analysis": semantic_analysis,
            "learned_knowledge": self._get_learned_knowledge(),
            "report": report,
        }

    def _get_learned_knowledge(self) -> dict:
        """获取 Agent 学到的知识"""
        return {
            "difficult_cases_count": len(
                self.case_repo.cases.get("difficult_cases", [])
            ),
            "ambiguous_terms": len(
                [
                    k
                    for k, v in self.case_repo.cases.get(
                        "ambiguous_terms", {}
                    ).items()
                    if v.get("is_ambiguous")
                ]
            ),
            "learned_rules": self.case_repo.generate_learned_rules(),
        }

    def _basic_stats(self, df: pd.DataFrame, score_col: str = "编码分数") -> dict:
        """基础统计"""
        if score_col not in df.columns:
            score_col = "AI评分" if "AI评分" in df.columns else "编码分数"

        low_confidence_count = 0
        needs_review_count = 0
        if "质量标记" in df.columns:
            low_confidence_count = int(
                df["质量标记"].astype(str).str.contains("LOW_CONFIDENCE", na=False).sum()
            )
            needs_review_count = int(
                df["质量标记"].astype(str).str.contains("NEEDS_REVIEW", na=False).sum()
            )

        score_vals = pd.to_numeric(df[score_col], errors="coerce").fillna(999).astype(int)
        valid_subset = df[score_vals.isin([0, 1, 2])]
        non_999_vals = score_vals[score_vals != 999]
        mean_score = non_999_vals.mean() if len(non_999_vals) > 0 else None
        std_score = non_999_vals.std() if len(non_999_vals) > 1 else 0.0

        return {
            "total_count": len(df),
            "score_distribution": score_vals.value_counts().to_dict(),
            "valid_count": len(valid_subset),
            "missing_count": int((score_vals == 999).sum()),
            "mean_score": mean_score,
            "std_score": float(std_score) if std_score is not None else 0.0,
            "low_confidence_count": low_confidence_count,
            "needs_review_count": needs_review_count,
        }

    def _semantic_analysis(self, df: pd.DataFrame, score_col: str = "编码分数") -> dict:
        """
        语义分析：找出各分数段的典型特征
        """
        analysis = {}
        if score_col not in df.columns:
            score_col = "AI评分" if "AI评分" in df.columns else "编码分数"
        if score_col not in df.columns or "回答内容" not in df.columns:
            return analysis

        for score in [0, 1, 2]:
            subset = df[df[score_col] == score]["回答内容"].dropna().astype(str).tolist()
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
        生成科研级分析报告
        """
        total = stats["total_count"]
        if total == 0:
            return "# 问卷编码分析报告\n\n无数据。"

        mean_score = stats["mean_score"]
        mean_str = (
            f"{mean_score:.2f}"
            if mean_score is not None and pd.notna(mean_score)
            else "N/A"
        )
        std_str = f"{stats.get('std_score', 0):.2f}"

        report = f"""# 问卷编码分析报告

## 元数据
- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 使用模型：{config.PRIMARY_MODEL}
- 双模型验证：{'启用' if config.ENABLE_DUAL_MODEL else '未启用'}

## 一、基础统计

- 总样本数：{stats['total_count']}
- 有效样本：{stats['valid_count']} ({stats['valid_count']/total*100:.1f}%)
- 缺失值(999)：{stats['missing_count']} ({stats['missing_count']/total*100:.1f}%)
- 平均分：{mean_str}
- 标准差：{std_str}

### 分数分布
"""
        for score, count in sorted(stats["score_distribution"].items()):
            pct = count / total * 100
            report += f"- {score}分：{count}个 ({pct:.1f}%)\n"

        report += f"""
## 二、质量控制

- 低置信度样本：{stats['low_confidence_count']}个
- 需人工复核：{stats.get('needs_review_count', 0)}个

### 建议
"""
        if stats["low_confidence_count"] > total * 0.2:
            report += "⚠️ 超过20%的样本置信度较低，建议：\n"
            report += "1. 人工验证分层抽样的100个样本\n"
            report += "2. 计算Cohen's Kappa评估信度\n"
            report += "3. 根据验证结果调整评分标准\n"
        else:
            report += "✓ 置信度分布合理，建议人工验证30-50个样本确认质量\n"

        report += f"""
## 三、语义分析

### 各分数段特征
"""
        for score_key, analysis in semantic.items():
            score = score_key.split("_")[1]
            report += f"\n#### {score}分样本\n"
            if isinstance(analysis, dict) and "common_patterns" in analysis:
                report += "共同特征：\n"
                for pattern in analysis["common_patterns"]:
                    report += f"- {pattern}\n"

        report += f"""
## 四、信度检验（待完成）

为确保编码质量，请完成以下步骤：

1. **分层抽样验证**
   - 运行：`python tools/create_validation_sheet.py`
   - 人工评分抽样样本
   - 计算Cohen's Kappa

2. **可疑样本复核**
   - 检查 `*_quality.json` 中的低置信度案例
   - 人工判断是否需要调整

3. **结果报告**
   - 在论文方法部分报告：使用的模型、置信度阈值、Kappa系数
   - 附录中提供评分标准和抽样验证结果

## 五、引用建议

如果在论文中使用本系统，建议这样描述方法：

> 问卷数据编码采用大语言模型辅助方法。首先由{config.PRIMARY_MODEL}模型根据预设评分标准进行自动编码。对于模型置信度低于{config.CONFIDENCE_THRESHOLD}的样本，启用{config.SECONDARY_MODEL if config.ENABLE_DUAL_MODEL else '关键词匹配'}策略进行二次验证。为确保编码质量，从每个分数段随机抽取[N]个样本由两名独立评分者进行人工评分，Cohen's Kappa系数为[待计算]，表明编码具有良好的信度。
"""
        return report

    def _generate_enhanced_report(
        self, stats: dict, semantic: dict, all_responses: list
    ) -> str:
        """生成增强版报告（包含 Agent 学习内容）"""
        base_report = self._generate_report(stats, semantic)

        report = base_report.replace(
            "# 问卷编码分析报告",
            "# 问卷编码分析报告（Agent 学习版）",
            1,
        )

        report += "\n---\n\n"
        report += self.case_repo.get_ambiguous_terms_report()

        report += "\n---\n\n"
        report += self.case_repo.get_boundary_patterns_report()

        report += "\n---\n\n"
        report += self.case_repo.get_respondent_report(all_responses)

        report += "\n---\n\n"
        report += self.case_repo.get_learned_rules_report()

        report += "\n---\n\n"
        report += self._generate_improvement_suggestions()

        return report

    def _generate_improvement_suggestions(self) -> str:
        """基于学习结果生成改进建议"""
        rules = self.case_repo.generate_learned_rules()

        report = "## Agent 改进建议\n\n"

        if not rules:
            report += "当前数据量不足，暂无改进建议。建议处理更多数据后再查看此部分。\n"
            return report

        report += "### 建议1：更新评分标准\n"
        report += "根据 Agent 学习到的规则，建议在评分标准中增加：\n\n"

        for rule in rules[:5]:
            report += f"- {rule['rule_text']}\n"

        report += "\n### 建议2：优化关键词匹配\n"
        report += "可以将以下词汇加入关键词匹配工具：\n\n"

        word_rules = [r for r in rules if r["type"] == "word_tendency"]
        for rule in word_rules[:3]:
            report += f"- '{rule['word']}' → 倾向{rule['suggested_score']}分\n"

        report += "\n### 建议3：人工复核重点\n"
        report += "建议重点复核以下类型的回答：\n"
        report += "- 包含模糊词汇的回答\n"
        report += "- 1分案例（容易与2分混淆）\n"
        difficult_count = len(
            self.case_repo.cases.get("difficult_cases", [])
        )
        report += f"- 当前有{difficult_count}个疑难案例需要关注\n"

        return report
