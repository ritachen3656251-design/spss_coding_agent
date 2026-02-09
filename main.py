"""
SPSS Agent System - 智能问卷编码系统
完整流程：SAV → CSV → 预处理 → 智能打分 → 质量检查 → 分析报告
"""
import argparse
import os

import config
from agents.scoring_agent import load_rubric
from tools.batch_processor import AgentPipeline
from tools.data_converter import find_and_convert_sav_to_questionnaire


def main():
    parser = argparse.ArgumentParser(description="SPSS Agent System")
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="使用异步处理模式",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="从断点继续（默认开启）",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="忽略断点，重新开始",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🤖 SPSS Agent System - 智能问卷编码系统")
    print("=" * 60)
    print()

    output_csv = "data/output/scored_results.csv"

    # 步骤1：确定输入文件（优先已有 CSV，否则从 SAV 转换）
    if os.path.exists(config.QUESTIONNAIRE_CSV):
        print("📄 步骤1：使用已有 questionnaire.csv")
        input_file = config.QUESTIONNAIRE_CSV
    else:
        print("📄 步骤1：转换 SAV 文件为 questionnaire.csv...")
        input_file = find_and_convert_sav_to_questionnaire(
            input_dir=config.INPUT_DIR,
            output_csv_path=config.QUESTIONNAIRE_CSV,
        )
        if input_file is None:
            print("⚠️  data/input 中未找到 .sav 文件，且无 questionnaire.csv")
            print("   请将 SAV 文件放入 data/input/ 或手动提供 questionnaire.csv")
            return

    # 步骤2：加载评分标准和情景
    print("\n📋 步骤2：加载评分标准和情景...")
    rubric = load_rubric(config.RUBRIC_PATH)

    # 加载情景描述（如果存在）
    scenario_path = "prompts/scenario.txt"
    scenario = None
    if os.path.exists(scenario_path):
        with open(scenario_path, "r", encoding="utf-8") as f:
            scenario = f.read()
        print(f"✓ 评分标准已加载（{len(rubric)} 字符）")
        print(f"✓ 情景描述已加载（{len(scenario)} 字符）")
    else:
        print(f"✓ 评分标准已加载（{len(rubric)} 字符）")
        print("⚠️  未找到情景描述文件，将不使用情景辅助")

    # 步骤3：初始化 Agent 流水线
    print("\n🤖 步骤3：初始化 Agent 流水线...")
    pipeline = AgentPipeline(rubric, scenario)
    print("✓ 已启动以下 Agent：")
    print("  - PreprocessingAgent（预处理）")
    print("  - ScoringAgent（智能打分）")
    if config.ENABLE_DUAL_MODEL:
        print("  - 双模型验证已启用")
    if config.ENABLE_KEYWORD_FALLBACK:
        print("  - 关键词回退已启用")
    print("  - QualityAgent（质量检查）")
    print("  - AnalysisAgent（数据分析）")

    print("\n📊 步骤4：开始批量处理...")
    print(f"置信度阈值：{config.CONFIDENCE_THRESHOLD}")
    print()

    result_df = pipeline.process_csv(
        input_csv=input_file,
        output_csv=output_csv,
        use_async=args.use_async,
        resume=args.resume,
    )

    print("\n" + "=" * 60)
    print("✅ 全部完成！")
    print("=" * 60)
    print("\n生成文件：")
    print(f"  1. 打分结果：{output_csv}")
    print(f"  2. 分析报告：{output_csv.replace('.csv', '_report.md')}")
    print(f"  3. 质量报告：{output_csv.replace('.csv', '_quality.json')}")
    print()

    score_col = "编码分数"
    quality_col = "质量标记"
    score_dist = result_df[score_col].value_counts().to_dict()
    print("快速统计：")
    for score, count in sorted(score_dist.items()):
        print(f"  {score} 分：{count} 个")

    if quality_col in result_df.columns:
        needs_review = int(
            result_df[quality_col]
            .astype(str)
            .str.contains("NEEDS_REVIEW", na=False)
            .sum()
        )
        if needs_review > 0:
            print(f"\n⚠️  有 {needs_review} 条数据需要人工复核")

    print("\n💾 保存 Agent 记忆与知识库...")
    pipeline.scoring_agent.memory.save_long_term()
    pipeline.scoring_agent.save_knowledge()
    print("✓ 记忆与知识库已保存，将用于优化未来处理")


if __name__ == "__main__":
    main()
