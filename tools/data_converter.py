"""
SPSS .sav 文件与 CSV 转换模块
"""
import glob
import os

import pyreadstat


def find_and_convert_sav_to_questionnaire(
    input_dir: str = "data/input",
    output_csv_path: str = "data/input/questionnaire.csv",
) -> str | None:
    """
    在 input_dir 中查找 .sav 文件，将其转换为 questionnaire.csv

    参数：
    - input_dir: 输入目录，用于搜索 .sav 文件
    - output_csv_path: 输出的 CSV 文件路径（默认 questionnaire.csv）

    返回：
    - 转换成功：返回输出 CSV 的路径
    - 未找到 SAV 文件：返回 None
    - 转换失败：返回 None
    """
    sav_files = glob.glob(os.path.join(input_dir, "*.sav"))
    if not sav_files:
        return None
    # 按文件名排序，保证行为一致；优先 questionnaire.sav
    sav_files.sort(key=lambda p: (os.path.basename(p) != "questionnaire.sav", p))
    sav_path = sav_files[0]
    print(f"   发现 SAV 文件：{os.path.basename(sav_path)}")
    success = sav_to_csv(sav_path, output_csv_path)
    if success:
        print(f"   ✓ 已转换并保存为：{output_csv_path}")
        return output_csv_path
    return None


def sav_to_csv(sav_file_path: str, output_csv_path: str) -> bool:
    """
    将SPSS的.sav文件转换为CSV

    参数：
    - sav_file_path: 输入的.sav文件路径
    - output_csv_path: 输出的CSV文件路径

    返回：
    - True: 转换成功
    - False: 转换失败

    要求：
    1. 使用pyreadstat库读取.sav文件
    2. 保留所有列的原始编号和数据
    3. 将数据保存为CSV格式（utf-8-sig编码，避免中文乱码）
    4. 添加错误处理，打印失败原因
    """
    try:
        df, _ = pyreadstat.read_sav(sav_file_path)
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        return True
    except FileNotFoundError as e:
        print(f"转换失败：找不到文件 {sav_file_path}。{e}")
        return False
    except OSError as e:
        print(f"转换失败：读写文件时出错。{e}")
        return False
    except Exception as e:
        print(f"转换失败：{type(e).__name__} - {e}")
        return False
