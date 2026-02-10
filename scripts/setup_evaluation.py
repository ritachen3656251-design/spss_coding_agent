"""
评估系统初始化 - 从SAV文件一键初始化
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.evaluation.data_manager import EvaluationDataManager


def main():
    print("""
╔════════════════════════════════════════════════════════╗
║     评估系统初始化（从SAV文件）                         ║
╚════════════════════════════════════════════════════════╝
    """)

    if len(sys.argv) > 1:
        sav_file = sys.argv[1]
    else:
        sav_file = input("请输入SAV文件路径（支持拖拽）：\n> ").strip()

    sav_file = sav_file.strip('"').strip("'")

    if not os.path.exists(sav_file):
        print(f"❌ 文件不存在：{sav_file}")
        return

    if not sav_file.lower().endswith(".sav"):
        print("⚠️  警告：文件不是.sav格式，但仍尝试处理")

    file_size = os.path.getsize(sav_file) / 1024
    print(f"\n文件信息：")
    print(f"  - 路径：{sav_file}")
    print(f"  - 大小：{file_size:.1f} KB")

    confirm = input(f"\n确认处理此文件？(y/n): ").strip().lower()
    if confirm != "y":
        print("取消操作")
        return

    print(f"\n开始处理...\n")

    try:
        data_mgr = EvaluationDataManager()
        data_mgr.prepare_datasets_from_sav(sav_file)

        print("""
╔════════════════════════════════════════════════════════╗
║  ✅ 初始化完成！                                        ║
║                                                        ║
║  下一步：                                               ║
║  1. 处理生产数据：python main.py                        ║
║  2. 评估Agent：python evaluate.py --version v1.0       ║
╚════════════════════════════════════════════════════════╝
        """)

    except ImportError as e:
        print(f"\n❌ 缺少依赖库：{e}")
        print("\n请安装：pip install pyreadstat")

    except Exception as e:
        print(f"\n❌ 处理失败：{e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
