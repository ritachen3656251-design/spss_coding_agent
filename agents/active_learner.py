"""
主动学习 Agent：识别最有价值的样本、优先级排序、生成人工标注任务
"""
import pandas as pd

from agents.base_agent import BaseAgent


class ActiveLearner(BaseAgent):
    """
    主动学习Agent：
    1. 实时识别最需要人工标注的样本
    2. 优先级排序（哪些样本最有学习价值）
    3. 生成人工标注任务清单
    """

    def __init__(self):
        super().__init__(name="ActiveLearner")
        self.unlabeled_pool = []
        self.high_priority_queue = []

    def evaluate_sample_value(
        self, text: str, scoring_result: dict
    ) -> float:
        """
        评估样本的学习价值

        价值维度：
        1. 不确定性：置信度越低，价值越高
        2. 代表性：能代表一类样本的价值更高
        3. 多样性：与已有样本差异大的价值高

        返回：价值分数（0-1，越高越有价值）
        """
        value_score = 0.0

        uncertainty = 1 - scoring_result.get("confidence", 0)
        value_score += 0.5 * min(max(uncertainty, 0), 1)

        if scoring_result.get("score") == 1:
            value_score += 0.3

        if "conflict_details" in scoring_result:
            value_score += 0.2

        return min(value_score, 1.0)

    def should_request_human_label(
        self, value_score: float, threshold: float = 0.6
    ) -> bool:
        """判断是否应该请求人工标注"""
        return value_score >= threshold

    def add_to_labeling_queue(self, sample: dict):
        """添加到标注队列"""
        self.high_priority_queue.append(sample)
        self.high_priority_queue.sort(
            key=lambda x: x.get("value_score", 0), reverse=True
        )

    def generate_labeling_batch(self, batch_size: int = 50) -> pd.DataFrame:
        """
        生成一批需要人工标注的样本

        策略：结合不确定性和多样性
        """
        if len(self.high_priority_queue) < batch_size:
            return pd.DataFrame(self.high_priority_queue)

        top_candidates = self.high_priority_queue[:100]
        df = pd.DataFrame(top_candidates)

        if df.empty or "text" not in df.columns:
            return df.head(batch_size)

        df["text"] = df["text"].fillna("").astype(str)
        df["length_bin"] = pd.cut(
            df["text"].str.len(),
            bins=5,
            labels=False,
        ).fillna(0).astype(int)

        sort_col = "value_score" if "value_score" in df.columns else "length_bin"
        n_per_bin = max(1, batch_size // 5)
        selected = (
            df.groupby("length_bin", group_keys=False)
            .apply(lambda x: x.nlargest(n_per_bin, sort_col) if sort_col in x.columns else x.head(n_per_bin))
            .head(batch_size)
        )

        return selected.reset_index(drop=True)

    def create_labeling_sheet(
        self,
        output_path: str = "data/output/human_labeling_batch.xlsx",
    ):
        """创建人工标注表格"""
        import os

        batch = self.generate_labeling_batch(50)

        if batch.empty:
            self.log("没有需要标注的样本")
            return None

        id_col = next((c for c in ["id", "编号"] if c in batch.columns), batch.index.astype(str))
        text_col = next((c for c in ["text", "回答内容"] if c in batch.columns), batch.columns[0])
        score_col = next((c for c in ["ai_score", "AI评分", "编码分数"] if c in batch.columns), None)
        conf_col = next((c for c in ["confidence", "置信度"] if c in batch.columns), None)

        n = len(batch)
        labeling_df = pd.DataFrame({
            "编号": batch[id_col] if isinstance(id_col, str) else id_col,
            "回答内容": batch[text_col],
            "AI预测分数": batch[score_col] if score_col else [""] * n,
            "AI置信度": batch[conf_col] if conf_col else [""] * n,
            "价值分数": batch["value_score"] if "value_score" in batch.columns else [""] * n,
            "人工评分": [""] * n,
            "备注": [""] * n,
        })

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        labeling_df.to_excel(output_path, index=False, engine="openpyxl")
        self.log(f"✓ 标注任务已生成：{output_path}")

        return output_path

    def incorporate_human_labels(self, labeled_file: str):
        """
        将人工标注结果融入系统

        用途：
        1. 更新案例库
        2. 微调置信度阈值
        3. 改进评分标准
        """
        df = pd.read_excel(labeled_file, engine="openpyxl")

        agreements = []
        disagreements = []

        ai_col = "AI预测分数" if "AI预测分数" in df.columns else "AI评分"
        human_col = "人工评分"

        if human_col not in df.columns:
            self.log("标注文件缺少「人工评分」列")
            return {
                "agreement_rate": 0.0,
                "agreements": 0,
                "disagreements": 0,
                "disagreement_cases": [],
            }

        for _, row in df.iterrows():
            ai_score = row.get(ai_col)
            human_score = row.get(human_col)

            if pd.notna(human_score) and human_score != "":
                try:
                    human_val = int(float(human_score))
                except (ValueError, TypeError):
                    continue
                ai_val = int(float(ai_score)) if pd.notna(ai_score) else None
                if ai_val == human_val:
                    agreements.append(row.to_dict())
                else:
                    disagreements.append(row.to_dict())

        total = len(agreements) + len(disagreements)
        agreement_rate = len(agreements) / total if total > 0 else 0.0

        self.log(f"人工标注结果：一致率 {agreement_rate:.1%}")

        return {
            "agreement_rate": agreement_rate,
            "agreements": len(agreements),
            "disagreements": len(disagreements),
            "disagreement_cases": disagreements,
        }
