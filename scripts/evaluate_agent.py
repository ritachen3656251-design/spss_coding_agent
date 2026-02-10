"""
评估 Agent 的标准流程脚本
"""
import argparse
import os
import sys

sys.path.insert(0, ".")  # 添加项目根目录到路径

import config
from agents.scoring_agent import load_rubric
from tools.batch_processor import AgentPipeline
from tools.evaluation.data_manager import EvaluationDataManager
from tools.evaluation.evaluator import AgentEvaluator


def main():
    """
    评估Agent的标准流程

    使用方法：
    python scripts/evaluate_agent.py --version v1.0-baseline --description "初始版本"
    """
    parser = argparse.ArgumentParser(description="评估Agent性能")
    parser.add_argument(
        "--version",
        required=True,
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
        help="使用哪个数据集",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比所有版本",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Agent评估系统")
    print(f"{'='*60}\n")

    # 1. 初始化数据管理器
    data_mgr = EvaluationDataManager()

    # 2. 获取数据集
    if args.dataset == "dev":
        ground_truth_df = data_mgr.get_dev_set()
        print(f"✓ 使用开发集（Dev Set）：{len(ground_truth_df)}条")
    elif args.dataset == "test":
        ground_truth_df = data_mgr.get_test_set()
        print(f"✓ 使用测试集（Test Set）：{len(ground_truth_df)}条")
    else:
        ground_truth_df = data_mgr.get_holdout_set()
        print(f"✓ 使用保留集（Holdout Set）：{len(ground_truth_df)}条")

    # 3. 运行Agent打分（使用当前系统）
    print(f"\n✓ 运行Agent打分（版本：{args.version}）...")

    # 提取待打分数据
    input_df = ground_truth_df[["编号", "回答内容"]].copy()

    # 保存为临时文件
    os.makedirs("data/evaluation", exist_ok=True)
    temp_input = "data/evaluation/temp_input.csv"
    temp_output = f"data/evaluation/temp_output_{args.version}.csv"
    input_df.to_csv(temp_input, index=False, encoding="utf-8-sig")

    # 加载评分标准和情景
    rubric = load_rubric(config.RUBRIC_PATH)
    scenario = None
    scenario_path = "prompts/scenario.txt"
    if os.path.exists(scenario_path):
        with open(scenario_path, "r", encoding="utf-8") as f:
            scenario = f.read()

    # 运行Agent
    pipeline = AgentPipeline(rubric, scenario)

    result_df = pipeline.process_csv(
        input_csv=temp_input,
        output_csv=temp_output,
        id_col="编号",
        answer_col="回答内容",
        score_col="AI评分",
    )

    # 4. 评估性能
    print(f"\n✓ 开始评估...")
    evaluator = AgentEvaluator()

    metrics = evaluator.evaluate(
        predictions_df=result_df,
        ground_truth_df=ground_truth_df,
        version_name=args.version,
        description=args.description,
    )

    # 5. 对比版本（如果需要）
    if args.compare:
        print(f"\n✓ 版本对比...")
        evaluator.compare_versions()

    print("\n✅ 评估完成！\n")
    print(f"详细结果：data/evaluation/results/{args.version}_detailed.csv")
    print(f"错误案例：data/evaluation/results/{args.version}_errors.csv")
    print(f"评估历史：data/evaluation/results/evaluation_history.json")


if __name__ == "__main__":
    main()
