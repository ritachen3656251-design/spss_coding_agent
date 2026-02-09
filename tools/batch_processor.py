"""
Agent 流水线：整合预处理、打分、质量检查、分析，批量处理 CSV
"""
import json
import os

import pandas as pd
from tqdm import tqdm

from agents.preprocessing_agent import PreprocessingAgent
from agents.scoring_agent import ScoringAgent
from agents.quality_agent import QualityAgent
from agents.analysis_agent import AnalysisAgent
import config


class AgentPipeline:
    """
    Agent 流水线：整合所有 Agent 的工作流
    高置信度与 999 的题目不输出判分理由，只输出分数。
    """

    def __init__(self, rubric: str, scenario: str = None):
        self.preprocessing_agent = PreprocessingAgent()
        self.scoring_agent = ScoringAgent(rubric, scenario)
        self.quality_agent = QualityAgent()
        self.analysis_agent = AnalysisAgent()

    def process_csv(
        self,
        input_csv: str,
        output_csv: str,
        id_col: str | None = None,
        answer_col: str | None = None,
        score_col: str = "编码分数",
    ) -> pd.DataFrame:
        """
        批量处理 CSV 文件
        id_col/answer_col 未指定时从 config 读取；若列不存在则自动推断（第一列作为回答，行号作为编号）
        """
        id_col = id_col or config.ID_COL
        answer_col = answer_col or config.ANSWER_COL

        print("📂 正在读取数据...")
        try:
            df = pd.read_csv(input_csv, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(input_csv, encoding="gbk")

        # 列名自动适配：若期望列不存在，用第一列作为回答、行号作为编号
        if answer_col not in df.columns:
            answer_col = df.columns[0]
            print(f"   使用列「{answer_col}」作为回答内容\n")
        if id_col not in df.columns:
            df["_row_id"] = range(1, len(df) + 1)
            id_col = "_row_id"
            print(f"   未找到编号列，使用行号作为编号\n")

        print(f"✓ 读取成功：{len(df)} 条数据\n")

        df[score_col] = None
        df["置信度"] = None
        df["判分理由"] = None
        df["使用策略"] = None
        df["质量标记"] = None

        cache_dir = config.CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        print("🤖 开始 Agent 流水线处理...\n")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理进度"):
            row_id = str(row[id_col])
            text = str(row[answer_col]) if pd.notna(row[answer_col]) else ""

            preprocess_result = self.preprocessing_agent.process(text)

            if preprocess_result["is_999"]:
                df.at[idx, score_col] = 999
                df.at[idx, "判分理由"] = ""  # 999 不输出理由，只输出分数
                df.at[idx, "使用策略"] = "auto_999"
                df.at[idx, "置信度"] = 1.0
                df.at[idx, "质量标记"] = ""
                continue

            scoring_result = self.scoring_agent.process(preprocess_result["text"])
            quality_result = self.quality_agent.process(
                scoring_result, text, row_id
            )

            df.at[idx, score_col] = quality_result["score"]
            df.at[idx, "置信度"] = quality_result["confidence"]
            # 高置信度时 scoring 已返回 reasoning=""，此处直接写入
            df.at[idx, "判分理由"] = quality_result.get("reasoning") or ""
            df.at[idx, "使用策略"] = quality_result["strategy"]
            df.at[idx, "质量标记"] = ", ".join(
                quality_result.get("quality_flags", [])
            )

            if idx % 10 == 0:
                df.to_csv(output_csv, index=False, encoding="utf-8-sig")

        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n✓ 处理完成，结果已保存：{output_csv}\n")

        print("📊 正在生成分析报告...")
        analysis_result = self.analysis_agent.process(df)
        report_path = output_csv.replace(".csv", "_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(analysis_result["report"])
        print(f"✓ 分析报告已保存：{report_path}\n")

        quality_report = self.quality_agent.generate_quality_report()
        quality_path = output_csv.replace(".csv", "_quality.json")
        with open(quality_path, "w", encoding="utf-8") as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)
        print(f"✓ 质量报告已保存：{quality_path}\n")

        return df
