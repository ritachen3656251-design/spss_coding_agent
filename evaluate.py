"""
SPSS Agent 评估系统 - 便捷入口
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from agents.scoring_agent import load_rubric, load_scenario
from tools.batch_processor import AgentPipeline
from tools.evaluation.data_manager import EvaluationDataManager
from tools.evaluation.evaluator import AgentEvaluator


def main():
    """
    评估系统便捷入口

    用法：
    1. 首次使用（从SAV初始化）：
       python evaluate.py --setup --sav-file path/to/data.sav

    2. 日常评估：
       python evaluate.py --version v1.0-baseline

    3. 对比所有版本：
       python evaluate.py --compare
    """
    parser = argparse.ArgumentParser(
        description="SPSS Agent System - 评估环境"
    )

    parser.add_argument(
        "--setup",
        action="store_true",
        help="初始化评估系统（从SAV文件）",
    )
    parser.add_argument(
        "--sav-file",
        help="SAV文件路径（包含text和label列）",
    )

    parser.add_argument(
        "--version",
        help="版本名称（如：v1.0-baseline）",
    )
    parser.add_argument(
        "--description",
        default="",
        help="版本描述",
    )
    parser.add_argument(
        "--dataset",
        default="dev",
        choices=["dev", "test", "holdout"],
        help="使用哪个数据集（默认：dev）",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比所有版本",
    )

    args = parser.parse_args()

    print("""
╔════════════════════════════════════════════════════════╗
║        SPSS Agent 评估系统                             ║
╚════════════════════════════════════════════════════════╝
    """)

    if args.setup:
        if not args.sav_file:
            print("❌ 请提供SAV文件：--sav-file path/to/file.sav")
            return

        print("✓ 模式：初始化评估系统\n")
        setup_evaluation_system(args.sav_file)
        return

    if args.compare:
        print("✓ 模式：版本对比\n")
        compare_all_versions()
        return

    if args.version:
        print(f"✓ 模式：评估版本 {args.version}\n")
        evaluate_version(args.version, args.description, args.dataset)
        return

    parser.print_help()
    print("\n✓ 提示：")
    print("  首次使用：python evaluate.py --setup --sav-file your_data.sav")
    print("  日常评估：python evaluate.py --version v1.0")
    print("  对比版本：python evaluate.py --compare")


def setup_evaluation_system(sav_file: str):
    """初始化评估系统（从SAV文件）"""
    try:
        data_mgr = EvaluationDataManager()
        data_mgr.prepare_datasets_from_sav(sav_file)

        print("""
╔════════════════════════════════════════════════════════╗
║  ✅ 初始化完成！                                        ║
║                                                        ║
║  生成的数据：                                           ║
║  ✓ 评估数据：dev/test/holdout → data/evaluation/splits/ ║
║  ✓ 生产数据：data/production/input/questionnaire.csv   ║
║                                                        ║
║  下一步：                                               ║
║  1. 评估baseline: python evaluate.py --version v1.0    ║
║  2. 处理生产数据：将 questionnaire.csv 复制到            ║
║     data/input/ 后运行 python main.py                  ║
╚════════════════════════════════════════════════════════╝
        """)

    except Exception as e:
        print(f"❌ 初始化失败：{e}")
        import traceback

        traceback.print_exc()


def evaluate_version(version: str, description: str, dataset: str):
    """评估指定版本"""
    try:
        data_mgr = EvaluationDataManager()

        if dataset == "dev":
            ground_truth_df = data_mgr.get_dev_set()
            print(f"✓ 使用开发集：{len(ground_truth_df)}条")
        elif dataset == "test":
            ground_truth_df = data_mgr.get_test_set()
            print(f"✓ 使用测试集：{len(ground_truth_df)}条")
        else:
            ground_truth_df = data_mgr.get_holdout_set()
            print(f"✓ 使用保留集：{len(ground_truth_df)}条")

        input_df = ground_truth_df[["编号", "回答内容"]].copy()

        temp_dir = Path("data/evaluation/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_input = temp_dir / "input.csv"
        temp_output = temp_dir / f"output_{version}.csv"

        input_df.to_csv(temp_input, index=False, encoding="utf-8-sig")

        print(f"\n✓ 运行Agent打分（{version}）...\n")

        rubric = load_rubric(config.RUBRIC_PATH)
        scenario = load_scenario()

        pipeline = AgentPipeline(rubric, scenario)

        result_df = pipeline.process_csv(
            input_csv=str(temp_input),
            output_csv=str(temp_output),
            id_col="编号",
            answer_col="回答内容",
            score_col="AI评分",
        )

        print(f"\n✓ 评估性能...\n")
        evaluator = AgentEvaluator()

        evaluator.evaluate(
            predictions_df=result_df,
            ground_truth_df=ground_truth_df,
            version_name=version,
            description=description,
        )

        print("\n✅ 评估完成！\n")
        print(f"详细结果：data/evaluation/results/{version}_detailed.csv")
        print(f"错误案例：data/evaluation/results/{version}_errors.csv")

    except FileNotFoundError as e:
        print(f"\n❌ 错误：{e}")
        print("\n✓ 可能原因：")
        print("  1. 尚未初始化评估系统")
        print(
            "  2. 解决方法：python evaluate.py --setup --sav-file your_data.sav"
        )

    except Exception as e:
        print(f"\n❌ 评估失败：{e}")
        import traceback

        traceback.print_exc()


def compare_all_versions():
    """对比所有版本"""
    try:
        evaluator = AgentEvaluator()
        report = evaluator.compare_versions()

        print("\n" + "=" * 60)
        print(report)
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"❌ 对比失败：{e}")


if __name__ == "__main__":
    main()
