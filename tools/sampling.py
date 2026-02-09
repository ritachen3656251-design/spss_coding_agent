import pandas as pd


class SamplingTool:
    """
    抽样工具：
    - 随机抽样：验证整体质量
    - 分层抽样：确保各分数段都被验证
    - 可疑样本定向抽样
    """

    def __init__(self):
        pass

    def stratified_sample(
        self, df: pd.DataFrame, n_per_score: int = 20
    ) -> pd.DataFrame:
        """
        分层抽样：每个分数段抽取n条

        返回：抽样后的DataFrame，用于人工验证
        """
        sampled = []

        for score in [0, 1, 2]:
            score_subset = df[df["编码分数"] == score]
            if len(score_subset) >= n_per_score:
                sampled.append(
                    score_subset.sample(n=n_per_score, random_state=42)
                )
            else:
                sampled.append(score_subset)

        result = pd.concat(sampled).reset_index(drop=True)
        result["验证状态"] = "待人工验证"
        result["人工评分"] = None

        return result

    def sample_low_confidence(
        self, df: pd.DataFrame, threshold: float = 0.6, n: int = 50
    ) -> pd.DataFrame:
        """抽取低置信度样本用于重点验证"""
        low_conf = df[df["置信度"] < threshold].copy()

        if len(low_conf) > n:
            return low_conf.sample(n=n, random_state=42)
        else:
            return low_conf

    def create_validation_sheet(
        self, sampled_df: pd.DataFrame, output_path: str
    ):
        """创建人工验证表格"""
        id_col = (
            "编号"
            if "编号" in sampled_df.columns
            else ("_row_id" if "_row_id" in sampled_df.columns else sampled_df.columns[0])
        )
        answer_col = (
            "回答内容"
            if "回答内容" in sampled_df.columns
            else ("VAR00001" if "VAR00001" in sampled_df.columns else sampled_df.columns[1] if len(sampled_df.columns) > 1 else sampled_df.columns[0])
        )
        score_col = "编码分数"

        validation_df = sampled_df[[id_col, answer_col, score_col]].copy()
        validation_df = validation_df.rename(columns={score_col: "AI评分"})
        validation_df["人工评分"] = ""
        validation_df["备注"] = ""

        validation_df.to_excel(output_path, index=False, engine="openpyxl")
        print(f"✓ 验证表格已生成：{output_path}")
