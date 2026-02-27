"""
A/B 实验：置信度阈值 0.6–0.8 对比
测试不同 CONFIDENCE_THRESHOLD 对 Kappa、准确率、双模型触发率的影响
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from agents.scoring_agent import load_rubric, load_scenario
from tools.batch_processor import AgentPipeline
from tools.evaluation.data_manager import EvaluationDataManager
from tools.evaluation.evaluator import AgentEvaluator


THRESHOLDS = [0.6, 0.65, 0.7, 0.75, 0.8]


def run_single_threshold(threshold: float, dataset: str = "dev") -> dict:
    """在指定置信度阈值下运行评估，返回指标"""
    config.CONFIDENCE_THRESHOLD = threshold

    data_mgr = EvaluationDataManager()
    if dataset == "dev":
        ground_truth_df = data_mgr.get_dev_set()
    elif dataset == "test":
        ground_truth_df = data_mgr.get_test_set()
    else:
        ground_truth_df = data_mgr.get_holdout_set()

    input_df = ground_truth_df[["编号", "回答内容"]].copy()
    temp_dir = Path("data/evaluation/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_input = temp_dir / "input.csv"
    temp_output = temp_dir / f"ab_conf{threshold}.csv"
    input_df.to_csv(temp_input, index=False, encoding="utf-8-sig")

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

    dual_model_strategies = [
        "dual_model_consensus",
        "primary_higher_confidence",
        "secondary_higher_confidence",
    ]
    strategy_col = "使用策略"
    dual_model_count = 0
    if strategy_col in result_df.columns:
        dual_model_count = int(
            result_df[strategy_col]
            .astype(str)
            .isin(dual_model_strategies)
            .sum()
        )

    evaluator = AgentEvaluator()
    metrics = evaluator.evaluate(
        predictions_df=result_df,
        ground_truth_df=ground_truth_df,
        version_name=f"ab_conf{threshold}",
        description=f"阈值={threshold}",
    )

    n = len(ground_truth_df)
    return {
        "threshold": threshold,
        "kappa": metrics["kappa"],
        "accuracy": metrics["accuracy"]["overall"],
        "dual_model_count": dual_model_count,
        "dual_model_rate": dual_model_count / n if n > 0 else 0,
        "mae": metrics.get("error_analysis", {}).get("mean_absolute_error", 0),
        "n_samples": n,
    }


def main():
    import argparse
    # A/B 测试期间关闭语义分析以节省 LLM 成本
    _orig_semantic = getattr(config, "ENABLE_SEMANTIC_ANALYSIS", True)
    config.ENABLE_SEMANTIC_ANALYSIS = False

    parser = argparse.ArgumentParser(description="置信度阈值 A/B 实验")
    parser.add_argument(
        "--dataset",
        default="dev",
        choices=["dev", "test", "holdout"],
        help="评估数据集",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="逗号分隔的阈值列表，如 0.6,0.65,0.7",
    )
    args = parser.parse_args()

    thresholds = (
        [float(t.strip()) for t in args.thresholds.split(",")]
        if args.thresholds
        else THRESHOLDS
    )

    print("""
╔════════════════════════════════════════════════════════╗
║     置信度阈值 A/B 实验 (0.6–0.8)                      ║
╚════════════════════════════════════════════════════════╝
    """)
    print(f"数据集: {args.dataset}")
    print(f"阈值: {thresholds}\n")

    results = []
    for th in thresholds:
        print(f"--- 运行阈值 {th} ---")
        try:
            r = run_single_threshold(th, args.dataset)
            results.append(r)
            print(f"  Kappa={r['kappa']:.3f}, Acc={r['accuracy']:.3f}, "
                  f"双模型触发={r['dual_model_count']} ({r['dual_model_rate']*100:.1f}%)\n")
        except Exception as e:
            print(f"  ❌ 失败: {e}\n")
            import traceback
            traceback.print_exc()

    if not results:
        print("无有效结果")
        return

    print("=" * 70)
    print("汇总对比")
    print("=" * 70)
    print(f"{'阈值':<8} {'Kappa':<10} {'准确率':<10} {'双模型数':<10} {'双模型%':<10} {'MAE':<8}")
    print("-" * 70)
    for r in results:
        print(f"{r['threshold']:<8.2f} {r['kappa']:<10.3f} {r['accuracy']:<10.3f} "
              f"{r['dual_model_count']:<10} {r['dual_model_rate']*100:<10.1f} {r['mae']:<8.3f}")

    best_kappa = max(results, key=lambda x: x["kappa"])
    best_acc = max(results, key=lambda x: x["accuracy"])
    print("-" * 70)
    print(f"最高 Kappa: 阈值 {best_kappa['threshold']} → {best_kappa['kappa']:.3f}")
    print(f"最高准确率: 阈值 {best_acc['threshold']} → {best_acc['accuracy']:.3f}")
    print()

    config.ENABLE_SEMANTIC_ANALYSIS = _orig_semantic


if __name__ == "__main__":
    main()
