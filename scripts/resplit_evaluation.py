"""
评估数据重新划分 - 使用推荐比例（15% : 30% : 55%）重新划分 dev/test/holdout

基于 data/evaluation/labeled/ground_truth.csv，无需重新读取 SAV 文件。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.evaluation.data_manager import EvaluationDataManager


def main():
    print("""
╔════════════════════════════════════════════════════════╗
║     评估数据重新划分（推荐比例 15% : 30% : 55%）        ║
╚════════════════════════════════════════════════════════╝
    """)

    try:
        data_mgr = EvaluationDataManager()
        data_mgr.resplit_from_labeled()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 处理失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
