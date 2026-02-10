"""
评估数据管理器：管理人工标注的黄金数据，智能分割数据集，避免数据泄露
支持从单个 SAV 文件自动分离已标注和未标注数据
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


class EvaluationDataManager:
    """
    评估数据管理器（SAV文件专用版）：
    1. 从单个SAV文件中分离已标注和未标注数据
    2. 智能分割数据集
    3. 避免数据泄露
    """

    def __init__(self, data_dir="data/evaluation"):
        self.data_dir = data_dir
        self.labeled_dir = os.path.join(data_dir, "labeled")
        self.splits_dir = os.path.join(data_dir, "splits")

        for d in [self.data_dir, self.labeled_dir, self.splits_dir]:
            os.makedirs(d, exist_ok=True)

        # 数据集配置文件
        self.config_file = os.path.join(data_dir, "dataset_config.json")
        self.config = self._load_or_create_config()

    def _load_or_create_config(self) -> dict:
        """加载或创建数据集配置"""
        if os.path.exists(self.config_file):
            with open(self.config_file, "r", encoding="utf-8") as f:
                return json.load(f)

        # 默认配置
        return {
            "created_at": datetime.now().isoformat(),
            "splits": {
                "dev": 50,  # 开发集：快速迭代测试
                "test": 100,  # 测试集：最终评估
                "holdout": 250,  # 保留集：训练/优化用
            },
            "split_seed": 42,
            "split_done": False,
            "source_file": None,
        }

    def read_sav_file(self, sav_file_path: str) -> pd.DataFrame:
        """
        读取SPSS文件

        参数：
        - sav_file_path: SAV文件路径

        返回：DataFrame
        """
        try:
            import pyreadstat
        except ImportError:
            raise ImportError(
                "需要安装 pyreadstat 库来读取SAV文件\n"
                "请运行：pip install pyreadstat"
            )

        print(f"\n✓ 正在读取SPSS文件：{sav_file_path}")

        df, meta = pyreadstat.read_sav(sav_file_path)

        print(f"✓ 读取成功：{len(df)}条数据，{len(df.columns)}列\n")

        print("✓ SPSS文件信息：")
        print(f"  变量数：{len(meta.column_names)}")
        print(f"  观测数：{len(df)}")

        print(f"\n变量列表：")
        for i, (col, label) in enumerate(
            zip(meta.column_names, meta.column_labels), 1
        ):
            print(f"  {i}. {col}")
            if label:
                print(f"     标签：{label}")

        return df

    def detect_columns(self, df: pd.DataFrame) -> dict:
        """
        自动检测text和label列

        返回：
        {
            "text_col": "列名",
            "label_col": "列名",
            "id_col": "列名" or None
        }
        """
        print(f"\n✓ 自动检测列名...")

        # 检测text列
        text_col = None
        for col in df.columns:
            if col.lower() in [
                "text",
                "answer",
                "回答",
                "回答内容",
                "文本",
                "response",
            ]:
                text_col = col
                break

        if text_col is None:
            for col in df.columns:
                if df[col].dtype == "object":
                    avg_len = df[col].astype(str).str.len().mean()
                    if avg_len > 5:
                        text_col = col
                        break

        # 检测label列
        label_col = None
        for col in df.columns:
            if col.lower() in [
                "label",
                "score",
                "标注",
                "评分",
                "人工评分",
                "coding",
            ]:
                label_col = col
                break

        if label_col is None:
            for col in df.columns:
                if col != text_col and df[col].dtype in ["int64", "float64"]:
                    unique_vals = df[col].dropna().unique()
                    valid_vals = [
                        v
                        for v in unique_vals
                        if not (pd.isna(v) or isinstance(v, (str, bool)))
                    ]
                    try:
                        if len(valid_vals) <= 10 and all(
                            0 <= float(v) <= 999 for v in valid_vals
                        ):
                            label_col = col
                            break
                    except (ValueError, TypeError):
                        pass

        # 检测ID列
        id_col = None
        first_col = df.columns[0]
        if first_col != text_col and first_col != label_col:
            id_col = first_col

        # 自动检测失败时进入交互模式
        if text_col is None or label_col is None:
            print("\n⚠️  自动检测失败，请手动指定：\n")
            print(f"可用列：{list(df.columns)}\n")

            if text_col is None:
                print("前3行数据预览：")
                print(df.head(3))
                text_col = input(
                    "\n请输入文本列的列名（如：text）："
                ).strip()

            if label_col is None:
                label_col = input(
                    "请输入标注列的列名（如：label）："
                ).strip()

        if not text_col or not label_col:
            raise ValueError(
                "文本列和标注列不能为空，请重新运行并正确指定列名"
            )
        if text_col not in df.columns:
            raise ValueError(f"文本列「{text_col}」不存在于数据中")
        if label_col not in df.columns:
            raise ValueError(f"标注列「{label_col}」不存在于数据中")

        result = {
            "text_col": text_col,
            "label_col": label_col,
            "id_col": id_col,
        }

        print(f"\n✓ 检测结果：")
        print(f"  - 文本列：{text_col}")
        print(f"  - 标注列：{label_col}")
        print(f"  - ID列：{id_col if id_col else '自动生成'}")

        return result

    def prepare_datasets_from_sav(self, sav_file_path: str):
        """
        从单个SAV文件准备所有数据集

        参数：
        - sav_file_path: SAV文件路径（包含text和label两列）

        工作流程：
        1. 读取SAV文件
        2. 自动检测text和label列
        3. 分离已标注（label有值）和未标注（label无值）
        4. 对已标注数据进行分层抽样 → dev/test/holdout
        5. 保存未标注数据 → production/input
        """
        if self.config["split_done"]:
            print(
                "⚠️  数据集已分割，如需重新分割请删除 dataset_config.json"
            )
            return

        if not os.path.exists(sav_file_path):
            raise FileNotFoundError(
                f"SAV文件不存在：{sav_file_path}"
            )

        print("=" * 60)
        print("✓ 从SAV文件准备数据集")
        print("=" * 60)

        # 1. 读取SAV文件
        df = self.read_sav_file(sav_file_path)

        # 2. 检测列名
        col_info = self.detect_columns(df)

        text_col = col_info["text_col"]
        label_col = col_info["label_col"]
        id_col = col_info["id_col"]

        # 3. 标准化列名
        print(f"\n✓ 标准化数据格式...")

        standardized = pd.DataFrame()

        if id_col:
            standardized["编号"] = df[id_col].astype(str)
        else:
            standardized["编号"] = [str(i + 1) for i in range(len(df))]

        standardized["回答内容"] = (
            df[text_col].fillna("").astype(str).str.strip()
        )

        label_series = df[label_col].copy()
        label_series = label_series.fillna(999)
        standardized["人工评分"] = label_series.astype(float).astype(int)

        print(f"✓ 标准化完成")

        # 4. 数据质量检查
        print(f"\n✓ 数据质量检查...")

        labeled_mask = standardized["人工评分"].isin([0, 1, 2])
        unlabeled_mask = ~labeled_mask

        n_labeled = int(labeled_mask.sum())
        n_unlabeled = int(unlabeled_mask.sum())

        print(f"  - 总数据：{len(standardized)}条")
        print(f"  - 已标注：{n_labeled}条（0/1/2分）")
        print(f"  - 未标注：{n_unlabeled}条（999或空值）")

        if n_labeled > 0:
            print(f"\n  已标注数据分布：")
            score_dist = (
                standardized[labeled_mask]["人工评分"]
                .value_counts()
                .sort_index()
            )
            for score, count in score_dist.items():
                pct = count / n_labeled * 100
                print(f"    {score}分：{count}条 ({pct:.1f}%)")

        empty_text = (standardized["回答内容"] == "").sum()
        if empty_text > 0:
            print(f"\n  ⚠️  发现{empty_text}条空文本")

        # 5. 分离已标注和未标注数据
        print(f"\n✓ 分离数据...")

        labeled_df = standardized[labeled_mask].copy()
        unlabeled_df = standardized[unlabeled_mask].copy()

        labeled_path = os.path.join(self.labeled_dir, "ground_truth.csv")
        labeled_df.to_csv(labeled_path, index=False, encoding="utf-8-sig")
        print(f"  ✓ 已标注数据保存：{labeled_path}")

        production_dir = "data/production/input"
        os.makedirs(production_dir, exist_ok=True)

        unlabeled_production = unlabeled_df[["编号", "回答内容"]].copy()
        unlabeled_path = os.path.join(
            production_dir, "questionnaire.csv"
        )
        unlabeled_production.to_csv(
            unlabeled_path, index=False, encoding="utf-8-sig"
        )
        print(f"  ✓ 未标注数据保存：{unlabeled_path}")

        # 6. 分层抽样
        min_required = 50  # dev+test 最小需求
        if n_labeled < min_required:
            raise ValueError(
                f"已标注数据不足{min_required}条（当前{n_labeled}条），无法分割数据集"
            )

        print(f"\n✓ 分层抽样（已标注数据）...")

        np.random.seed(self.config["split_seed"])

        dev_df, test_df, holdout_df = self._stratified_split(
            labeled_df,
            self.config["splits"],
        )

        dev_df.to_csv(
            os.path.join(self.splits_dir, "dev_set.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        test_df.to_csv(
            os.path.join(self.splits_dir, "test_set.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        holdout_df.to_csv(
            os.path.join(self.splits_dir, "holdout_set.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        print(f"  ✓ 开发集：{len(dev_df)}条")
        print(f"  ✓ 测试集：{len(test_df)}条")
        print(f"  ✓ 保留集：{len(holdout_df)}条")

        # 7. 生成报告
        report = self._generate_split_report(
            dev_df,
            test_df,
            holdout_df,
            unlabeled_production,
            n_labeled,
        )

        report_path = os.path.join(self.data_dir, "split_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # 8. 更新配置
        self.config["split_done"] = True
        self.config["split_date"] = datetime.now().isoformat()
        self.config["source_file"] = sav_file_path
        self.config["total_labeled"] = n_labeled
        self.config["total_unlabeled"] = n_unlabeled
        self.config["column_mapping"] = {
            "original_text_col": text_col,
            "original_label_col": label_col,
            "original_id_col": id_col,
        }

        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

        # 9. 显示总结
        print("\n" + "=" * 60)
        print("✅ 数据集准备完成！")
        print("=" * 60)
        print(report)

        print("\n✓ 生成的文件：")
        print(f"  评估数据：")
        print(f"    - {self.splits_dir}/dev_set.csv ({len(dev_df)}条)")
        print(f"    - {self.splits_dir}/test_set.csv ({len(test_df)}条)")
        print(f"    - {self.splits_dir}/holdout_set.csv ({len(holdout_df)}条)")
        print(f"  生产数据：")
        print(f"    - {unlabeled_path} ({len(unlabeled_production)}条)")
        print(f"  报告：")
        print(f"    - {report_path}")
        print()

    def _stratified_split(
        self, df: pd.DataFrame, splits: dict
    ) -> tuple:
        """
        分层抽样（保持各分数比例）

        参数：
        - df: 已标注数据
        - splits: {"dev": 100, "test": 200, "holdout": 500}

        返回：(dev_df, test_df, holdout_df)
        """
        dev_list = []
        test_list = []
        holdout_list = []

        total = len(df)
        if total == 0:
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
            )

        for score in df["人工评分"].unique():
            if score == 999:
                continue

            score_df = df[df["人工评分"] == score].copy()
            score_df = score_df.sample(
                frac=1, random_state=42
            ).reset_index(drop=True)

            n_total = len(score_df)

            n_dev = max(1, int(n_total * (splits["dev"] / total)))
            n_test = max(1, int(n_total * (splits["test"] / total)))

            n_dev = min(n_dev, n_total)
            n_test = min(n_test, n_total - n_dev)

            dev_list.append(score_df.iloc[:n_dev])
            test_list.append(score_df.iloc[n_dev : n_dev + n_test])
            holdout_list.append(score_df.iloc[n_dev + n_test :])

        return (
            pd.concat(dev_list).reset_index(drop=True)
            if dev_list
            else pd.DataFrame(),
            pd.concat(test_list).reset_index(drop=True)
            if test_list
            else pd.DataFrame(),
            pd.concat(holdout_list).reset_index(drop=True)
            if holdout_list
            else pd.DataFrame(),
        )

    def _generate_split_report(
        self, dev_df, test_df, holdout_df, unlabeled_df, n_labeled
    ) -> str:
        """生成分割报告"""

        def score_dist(df):
            if len(df) == 0:
                return "无数据"
            dist = df["人工评分"].value_counts().sort_index()
            return ", ".join(
                [f"{score}分:{count}个" for score, count in dist.items()]
            )

        report = f"""
# 数据集分割报告

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
随机种子：{self.config['split_seed']}
来源文件：{self.config.get('source_file', 'unknown')}

## 数据统计

- 总数据：{self.config.get('total_labeled', 0) + self.config.get('total_unlabeled', 0)}条
- 已标注：{n_labeled}条
- 未标注：{self.config.get('total_unlabeled', 0)}条

## 分割策略

采用**分层抽样**确保各数据集的分数分布一致，避免分布偏差。

## 评估数据集

### 1. 开发集 (Dev Set) - {len(dev_df)}条
**用途**：日常开发、快速迭代
**分数分布**：{score_dist(dev_df)}
**文件**：`data/evaluation/splits/dev_set.csv`

### 2. 测试集 (Test Set) - {len(test_df)}条
**用途**：最终评估（禁止调参！）
**分数分布**：{score_dist(test_df)}
**文件**：`data/evaluation/splits/test_set.csv`
⚠️  **重要**：测试集仅用于最终报告

### 3. 保留集 (Holdout Set) - {len(holdout_df)}条
**用途**：规则学习、标准优化
**分数分布**：{score_dist(holdout_df)}
**文件**：`data/evaluation/splits/holdout_set.csv`

## 生产数据

### 未标注池 - {len(unlabeled_df)}条
**用途**：生产环境处理
**文件**：`data/production/input/questionnaire.csv`

## 使用建议

1. **日常开发**：
```bash
   python scripts/evaluate_agent.py --version v1.0-baseline --dataset dev
```

2. **生产处理**：
```bash
   python main.py
```

3. **最终评估**（仅一次）：
```bash
   python scripts/evaluate_agent.py --version v2.0-final --dataset test
```
"""
        return report

    def get_dev_set(self) -> pd.DataFrame:
        """获取开发集（日常使用）"""
        path = os.path.join(self.splits_dir, "dev_set.csv")
        if not os.path.exists(path):
            path_legacy = os.path.join(self.data_dir, "dev_set.csv")
            if os.path.exists(path_legacy):
                return pd.read_csv(path_legacy, encoding="utf-8-sig")
            raise FileNotFoundError(
                "开发集不存在，请先运行数据集准备：\n"
                "python scripts/setup_evaluation.py"
            )
        return pd.read_csv(path, encoding="utf-8-sig")

    def get_test_set(self) -> pd.DataFrame:
        """获取测试集（最终评估）"""
        print("⚠️  警告：正在访问测试集，请确保这是最终评估！")
        path = os.path.join(self.splits_dir, "test_set.csv")
        if not os.path.exists(path):
            path_legacy = os.path.join(self.data_dir, "test_set.csv")
            if os.path.exists(path_legacy):
                return pd.read_csv(path_legacy, encoding="utf-8-sig")
            raise FileNotFoundError(
                "测试集不存在，请先运行数据集准备：\n"
                "python scripts/setup_evaluation.py"
            )
        return pd.read_csv(path, encoding="utf-8-sig")

    def get_holdout_set(self) -> pd.DataFrame:
        """获取保留集（规则学习）"""
        path = os.path.join(self.splits_dir, "holdout_set.csv")
        if not os.path.exists(path):
            path_legacy = os.path.join(self.data_dir, "holdout_set.csv")
            if os.path.exists(path_legacy):
                return pd.read_csv(path_legacy, encoding="utf-8-sig")
            raise FileNotFoundError(
                "保留集不存在，请先运行数据集准备：\n"
                "python scripts/setup_evaluation.py"
            )
        return pd.read_csv(path, encoding="utf-8-sig")
