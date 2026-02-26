"""
Agent 流水线：整合预处理、打分、质量检查、分析，批量处理 CSV 的流水线
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

_BATCH_SIZE = 10  # 每 N 条保存一次


class AgentPipeline:
    """
    Agent 流水线：整合所有 Agent 的工作流
    高置信度与 999 的题目不输出判分理由，只输出分数。
    """

    def __init__(self, rubric: str):
        self.preprocessing_agent = PreprocessingAgent()
        self.scoring_agent = ScoringAgent(rubric)
        self.quality_agent = QualityAgent()
        self.analysis_agent = AnalysisAgent()

    def _read_csv(self, path: str) -> pd.DataFrame:
        """读取 CSV，自动尝试 utf-8-sig 与 gbk 编码。"""
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="gbk")

    def _ensure_output_dir(self, filepath: str) -> None:
        """确保输出文件所在目录存在。"""
        out_dir = os.path.dirname(os.path.abspath(filepath))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def process_csv(
        self,
        input_csv: str,
        output_csv: str,
        id_col: str = "编号",
        answer_col: str = "回答内容",
        score_col: str = "编码分数",
    ) -> pd.DataFrame:
        """
        批量处理 CSV 文件

        Raises:
            ValueError: 当必需列不存在或数据为空时
        """
        print("正在读取数据...")
        df = self._read_csv(input_csv)

        if df.empty:
            raise ValueError(f"输入文件为空：{input_csv}")

        missing = [c for c in (id_col, answer_col) if c not in df.columns]
        if missing:
            raise ValueError(f"输入文件缺少必需列：{missing}，现有列：{list(df.columns)}")

        print(f"✓ 读取成功：{len(df)} 条数据\n")

        df[score_col] = None
        df["置信度"] = None
        df["判分理由"] = None
        df["使用策略"] = None
        df["质量标记"] = None

        cache_dir = config.CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        self._ensure_output_dir(output_csv)

        error_count = 0
        print("🤖 开始 Agent 流水线处理...\n")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理进度"):
            try:
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
                df.at[idx, "判分理由"] = quality_result.get("reasoning") or ""
                df.at[idx, "使用策略"] = quality_result["strategy"]
                df.at[idx, "质量标记"] = ", ".join(
                    quality_result.get("quality_flags", [])
                )
            except Exception as e:
                error_count += 1
                df.at[idx, "质量标记"] = f"ERROR: {str(e)[:50]}"
                print(f"\n⚠️ 第 {idx + 1} 行处理失败 (ID={row.get(id_col, '?')}): {e}")

            if (idx + 1) % _BATCH_SIZE == 0:
                df.to_csv(output_csv, index=False, encoding="utf-8-sig")

        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n✓ 处理完成，结果已保存：{output_csv}")
        if error_count > 0:
            print(f"⚠️ 共 {error_count} 条处理失败，已标记在「质量标记」列\n")
        else:
            print()

        print("正在生成分析报告...")
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
