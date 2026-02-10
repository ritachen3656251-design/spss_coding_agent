"""
Agent 评估器：计算完整评估指标、对比版本、生成可视化报告
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


class AgentEvaluator:
    """
    Agent评估器：
    1. 计算所有相关指标
    2. 对比不同版本
    3. 生成可视化报告

    完全独立，不修改Agent代码
    """

    def __init__(self, output_dir="data/evaluation/results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 所有评估历史
        self.history_file = os.path.join(output_dir, "evaluation_history.json")
        self.history = self._load_history()

    def _load_history(self) -> dict:
        """加载评估历史"""
        if os.path.exists(self.history_file):
            with open(self.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"evaluations": []}

    def evaluate(
        self,
        predictions_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        version_name: str,
        description: str = "",
    ) -> dict:
        """
        评估Agent性能

        参数：
        - predictions_df: Agent的预测结果（必须包含：编号、AI评分、置信度）
        - ground_truth_df: 人工标注真值（必须包含：编号、人工评分）
        - version_name: 版本名称（如："v1.0-baseline", "v1.1-with-context"）
        - description: 版本描述

        返回：完整的评估指标
        """
        print(f"\n{'='*60}")
        print(f"开始评估：{version_name}")
        print(f"{'='*60}\n")

        # 1. 数据对齐
        merged_df = self._align_data(predictions_df, ground_truth_df)

        if len(merged_df) == 0:
            raise ValueError("无法对齐预测和真值数据，请检查编号列")

        print(f"✓ 数据对齐完成：{len(merged_df)}条有效样本")

        # 2. 提取预测和真值（转为数值，处理 NaN/字符串）
        y_true = pd.to_numeric(merged_df["人工评分"], errors="coerce").fillna(999).astype(int)
        y_pred = pd.to_numeric(merged_df["AI评分"], errors="coerce").fillna(999).astype(int)
        confidences = (
            pd.to_numeric(merged_df["置信度"], errors="coerce").values
            if "置信度" in merged_df.columns
            else None
        )

        # 3. 过滤999及非0/1/2样本
        valid_mask = (
            y_true.isin([0, 1, 2]).values
            & y_pred.isin([0, 1, 2]).values
        )
        y_true_valid = y_true.values[valid_mask].astype(int)
        y_pred_valid = y_pred.values[valid_mask].astype(int)
        confidences_valid = (
            np.asarray(confidences, dtype=float)[valid_mask]
            if confidences is not None and len(confidences) > 0
            else None
        )

        print(f"✓ 过滤后有效样本：{len(y_true_valid)}条")

        if len(y_true_valid) == 0:
            raise ValueError(
                "过滤后无有效样本（0/1/2分），请检查预测和真值数据"
            )

        # 4. 计算所有指标
        metrics = self._calculate_all_metrics(
            y_true_valid,
            y_pred_valid,
            confidences_valid,
        )

        # 5. 添加元信息
        metrics["meta"] = {
            "version_name": version_name,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(y_true_valid),
            "n_total": len(merged_df),
        }

        # 6. 保存到历史
        self.history["evaluations"].append(metrics)

        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        # 7. 保存详细结果（带错误case）
        self._save_detailed_results(merged_df, valid_mask, metrics, version_name)

        # 8. 打印摘要
        self._print_summary(metrics)

        return metrics

    def _align_data(
        self, pred_df: pd.DataFrame, truth_df: pd.DataFrame
    ) -> pd.DataFrame:
        """对齐预测和真值数据"""
        pred_df = pred_df.copy()

        # 统一列名
        if "编码分数" in pred_df.columns and "AI评分" not in pred_df.columns:
            pred_df = pred_df.rename(columns={"编码分数": "AI评分"})

        pred_cols = ["编号", "AI评分"]
        if "置信度" in pred_df.columns:
            pred_cols.append("置信度")
        if "回答内容" in pred_df.columns:
            pred_cols.append("回答内容")

        pred_cols = [c for c in pred_cols if c in pred_df.columns]
        truth_cols = [
            c for c in ["编号", "人工评分", "回答内容"] if c in truth_df.columns
        ]

        if "编号" not in pred_cols or "编号" not in truth_cols:
            raise ValueError(
                "预测和真值数据必须包含「编号」列用于对齐"
            )
        if "AI评分" not in pred_cols:
            raise ValueError(
                "预测数据必须包含「AI评分」或「编码分数」列"
            )
        if "人工评分" not in truth_cols:
            raise ValueError(
                "真值数据必须包含「人工评分」列"
            )

        # 统一编号为字符串，避免 int/str 导致 merge 失败
        pred_merge = pred_df[pred_cols].copy()
        truth_merge = truth_df[truth_cols].copy()
        pred_merge["编号"] = pred_merge["编号"].astype(str).str.strip()
        truth_merge["编号"] = truth_merge["编号"].astype(str).str.strip()

        merged = pd.merge(
            pred_merge,
            truth_merge,
            on="编号",
            how="inner",
            suffixes=("_pred", "_truth"),
        )

        # 处理重复列名
        if "回答内容_truth" in merged.columns:
            merged["回答内容"] = merged["回答内容_truth"]
            drop_cols = [c for c in ["回答内容_pred", "回答内容_truth"] if c in merged.columns]
            merged = merged.drop(columns=drop_cols)
        elif "回答内容_pred" in merged.columns:
            merged["回答内容"] = merged["回答内容_pred"]
            merged = merged.drop(columns=["回答内容_pred"])

        return merged

    def _calculate_all_metrics(
        self, y_true, y_pred, confidences=None
    ) -> dict:
        """计算所有评估指标"""
        metrics = {}
        n = len(y_true)
        if n == 0:
            return metrics

        # ===== 1. 基础准确性指标 =====
        metrics["accuracy"] = {
            "overall": float(accuracy_score(y_true, y_pred)),
            "by_score": {},
        }

        for score in [0, 1, 2]:
            mask = y_true == score
            if mask.sum() > 0:
                acc = accuracy_score(y_true[mask], y_pred[mask])
                metrics["accuracy"]["by_score"][score] = float(acc)

        # ===== 2. Cohen's Kappa =====
        metrics["kappa"] = float(cohen_kappa_score(y_true, y_pred))

        # ===== 3. Precision, Recall, F1 =====
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], zero_division=0
        )

        metrics["per_class"] = {
            "0": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "1": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
            "2": {
                "precision": float(precision[2]),
                "recall": float(recall[2]),
                "f1": float(f1[2]),
                "support": int(support[2]),
            },
        }

        # ===== 4. 混淆矩阵 =====
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        metrics["confusion_matrix"] = cm.tolist()

        # ===== 5. 误差分析 =====
        abs_errors = np.abs(y_true - y_pred)
        over_estimate = (y_pred > y_true).sum()
        under_estimate = (y_pred < y_true).sum()

        metrics["error_analysis"] = {
            "mean_absolute_error": float(np.mean(abs_errors)),
            "error_distribution": {
                "0": int((abs_errors == 0).sum()),
                "1": int((abs_errors == 1).sum()),
                "2": int((abs_errors == 2).sum()),
            },
            "over_estimate_count": int(over_estimate),
            "under_estimate_count": int(under_estimate),
            "over_estimate_rate": float(over_estimate / n),
            "under_estimate_rate": float(under_estimate / n),
        }

        # ===== 6. 置信度分析 =====
        if confidences is not None and len(confidences) > 0:
            conf_arr = np.asarray(confidences, dtype=float)
            conf_valid = conf_arr[~np.isnan(conf_arr)]
            metrics["confidence_analysis"] = {
                "mean": float(np.nanmean(conf_arr)),
                "median": float(np.nanmedian(conf_arr)),
                "std": float(np.nanstd(conf_arr)) if len(conf_valid) > 1 else 0.0,
            }

            high_conf_mask = confidences >= 0.8
            if high_conf_mask.sum() > 0:
                high_conf_acc = accuracy_score(
                    y_true[high_conf_mask], y_pred[high_conf_mask]
                )
                metrics["confidence_analysis"]["high_confidence_accuracy"] = (
                    float(high_conf_acc)
                )
                metrics["confidence_analysis"]["high_confidence_count"] = int(
                    high_conf_mask.sum()
                )

            low_conf_mask = confidences < 0.6
            if low_conf_mask.sum() > 0:
                low_conf_acc = accuracy_score(
                    y_true[low_conf_mask], y_pred[low_conf_mask]
                )
                metrics["confidence_analysis"]["low_confidence_accuracy"] = (
                    float(low_conf_acc)
                )
                metrics["confidence_analysis"]["low_confidence_count"] = int(
                    low_conf_mask.sum()
                )

        # ===== 7. 边界问题分析 =====
        one_score_mask = y_true == 1
        if one_score_mask.sum() > 0:
            one_score_acc = accuracy_score(
                y_true[one_score_mask], y_pred[one_score_mask]
            )
            one_to_zero = ((y_true == 1) & (y_pred == 0)).sum()
            one_to_two = ((y_true == 1) & (y_pred == 2)).sum()

            metrics["boundary_analysis"] = {
                "score_1_accuracy": float(one_score_acc),
                "score_1_to_0_count": int(one_to_zero),
                "score_1_to_2_count": int(one_to_two),
                "score_1_total": int(one_score_mask.sum()),
            }

        return metrics

    def _save_detailed_results(
        self, merged_df, valid_mask, metrics, version_name
    ):
        """保存详细结果（包含所有错误case）"""
        merged_df = merged_df.copy()
        y_true = pd.to_numeric(merged_df["人工评分"], errors="coerce").fillna(999).astype(int)
        y_pred = pd.to_numeric(merged_df["AI评分"], errors="coerce").fillna(999).astype(int)
        merged_df["预测正确"] = (y_true.values == y_pred.values)
        merged_df["绝对误差"] = np.abs(y_true.astype(float).values - y_pred.astype(float).values)

        output_file = os.path.join(
            self.output_dir, f"{version_name}_detailed.csv"
        )
        merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        error_df = merged_df[~merged_df["预测正确"]].copy()
        error_df = error_df.sort_values("绝对误差", ascending=False)

        error_file = os.path.join(
            self.output_dir, f"{version_name}_errors.csv"
        )
        error_df.to_csv(error_file, index=False, encoding="utf-8-sig")

        print(f"✓ 详细结果已保存：{output_file}")
        print(f"✓ 错误案例已保存：{error_file}（{len(error_df)}个错误）")

    def _print_summary(self, metrics):
        """打印评估摘要"""
        print("\n" + "=" * 60)
        print("评估结果摘要")
        print("=" * 60)

        print(f"\n✓ 整体表现")
        print(f"  准确率 (Accuracy): {metrics['accuracy']['overall']:.1%}")
        print(
            f"  Kappa系数: {metrics['kappa']:.3f} "
            f"({self._interpret_kappa(metrics['kappa'])})"
        )

        print(f"\n✓ 各分数表现")
        for score in ["0", "1", "2"]:
            m = metrics["per_class"][score]
            print(
                f"  {score}分: F1={m['f1']:.1%}, "
                f"Precision={m['precision']:.1%}, "
                f"Recall={m['recall']:.1%} (n={m['support']})"
            )

        print(f"\n❌ 误差分析")
        err = metrics["error_analysis"]
        print(f"  平均绝对误差: {err['mean_absolute_error']:.2f}分")
        print(f"  完全正确: {err['error_distribution']['0']}个")
        print(f"  差1分: {err['error_distribution']['1']}个")
        print(f"  差2分: {err['error_distribution']['2']}个")
        print(f"  高估倾向: {err['over_estimate_rate']:.1%}")
        print(f"  低估倾向: {err['under_estimate_rate']:.1%}")

        if "confidence_analysis" in metrics:
            conf = metrics["confidence_analysis"]
            print(f"\n✓ 置信度分析")
            print(f"  平均置信度: {conf['mean']:.2f}")
            if "high_confidence_accuracy" in conf:
                print(
                    f"  高置信度(≥0.8)准确率: "
                    f"{conf['high_confidence_accuracy']:.1%} "
                    f"(n={conf['high_confidence_count']})"
                )
            if "low_confidence_accuracy" in conf:
                print(
                    f"  低置信度(<0.6)准确率: "
                    f"{conf['low_confidence_accuracy']:.1%} "
                    f"(n={conf['low_confidence_count']})"
                )

        if "boundary_analysis" in metrics:
            bd = metrics["boundary_analysis"]
            print(f"\n⚖️  边界分数(1分)分析")
            print(f"  1分准确率: {bd['score_1_accuracy']:.1%}")
            print(f"  误判为0分: {bd['score_1_to_0_count']}次")
            print(f"  误判为2分: {bd['score_1_to_2_count']}次")

        print("\n" + "=" * 60 + "\n")

    def _interpret_kappa(self, kappa: float) -> str:
        """解释Kappa系数"""
        if kappa > 0.8:
            return "几乎完全一致"
        elif kappa > 0.6:
            return "高度一致"
        elif kappa > 0.4:
            return "中度一致"
        elif kappa > 0.2:
            return "一般一致"
        else:
            return "一致性较低"

    def compare_versions(self, version_names: list = None) -> str:
        """
        对比多个版本的性能

        参数：
        - version_names: 要对比的版本列表，None表示对比所有版本

        返回：对比报告（markdown格式）
        """
        if not self.history["evaluations"]:
            return "暂无评估历史"

        evals = self.history["evaluations"]

        if version_names:
            evals = [
                e
                for e in evals
                if e["meta"]["version_name"] in version_names
            ]

        if len(evals) < 2:
            return "至少需要2个版本才能对比"

        report = "# Agent版本对比报告\n\n"
        report += f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        report += "## 性能对比\n\n"
        report += (
            "| 版本 | 准确率 | Kappa | 平均误差 | 高估率 | 低估率 | 平均置信度 |\n"
        )
        report += "|------|--------|-------|----------|--------|--------|------------|\n"

        for e in evals:
            version = e["meta"]["version_name"]
            acc = e["accuracy"]["overall"]
            kappa = e["kappa"]
            mae = e["error_analysis"]["mean_absolute_error"]
            over = e["error_analysis"]["over_estimate_rate"]
            under = e["error_analysis"]["under_estimate_rate"]
            conf = e.get("confidence_analysis", {}).get("mean", 0)

            report += f"| {version} | {acc:.1%} | {kappa:.3f} | {mae:.2f} | {over:.1%} | {under:.1%} | {conf:.2f} |\n"

        best_acc = max(evals, key=lambda x: x["accuracy"]["overall"])
        best_kappa = max(evals, key=lambda x: x["kappa"])

        report += f"\n## 最佳性能\n\n"
        report += f"- **最高准确率**: {best_acc['meta']['version_name']} ({best_acc['accuracy']['overall']:.1%})\n"
        report += f"- **最高Kappa**: {best_kappa['meta']['version_name']} ({best_kappa['kappa']:.3f})\n"

        if len(evals) >= 2:
            baseline = evals[0]
            latest = evals[-1]
            base_acc = baseline["accuracy"]["overall"]

            acc_improve = (
                (latest["accuracy"]["overall"] - base_acc) / base_acc * 100
                if base_acc > 0
                else 0
            )
            kappa_improve = latest["kappa"] - baseline["kappa"]

            report += f"\n## 改进幅度（相对于初始版本）\n\n"
            report += f"- 准确率提升: {acc_improve:+.1f}%\n"
            report += f"- Kappa提升: {kappa_improve:+.3f}\n"

        report_file = os.path.join(
            self.output_dir, "version_comparison.md"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(report)
        return report
