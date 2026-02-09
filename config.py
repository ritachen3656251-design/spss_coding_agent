"""
项目配置：主模型 Qwen，辅助验证模型 DeepSeek
"""
import os
from dotenv import load_dotenv

load_dotenv()  # 默认加载 .env
load_dotenv("env")  # 若存在 env 文件则也加载

# API配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # Qwen 主模型
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # 可选，用于双模型验证

# 模型配置
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "qwen-max")  # 主模型
SECONDARY_MODEL = os.getenv("SECONDARY_MODEL", "deepseek-chat")  # 用于边界 case 的第二意见

# DeepSeek API 端点（OpenAI 兼容）
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

# Agent行为配置
CONFIDENCE_THRESHOLD = 0.7  # 低于此置信度触发二次验证
ENABLE_DUAL_MODEL = True  # 是否启用双模型验证
ENABLE_KEYWORD_FALLBACK = True  # 是否启用关键词回退策略
ENABLE_REACT = True  # 是否启用 ReAct 循环（思考→行动→观察→检查）
REACT_CONFIDENCE_TARGET = 0.8  # ReAct 循环满意阈值
REACT_MAX_ITERATIONS = 3  # ReAct 最大迭代次数

# 评分配置
SCORE_OPTIONS = [0, 1, 2, 999]
AUTO_999_KEYWORDS = ["无法播放", "看不了", "加载失败", "视频错误", "显示不出", "跳过", "视频播放不了"]

# 文件路径
RUBRIC_PATH = "prompts/scoring_rubric.txt"
CACHE_DIR = "data/cache"
INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"
QUESTIONNAIRE_CSV = "data/input/questionnaire.csv"

# CSV 列名（SAV 转换后可能是 VAR00001 等，可通过环境变量覆盖）
ID_COL = os.getenv("ID_COL", "编号")  # 若不存在则自动用行号
ANSWER_COL = os.getenv("ANSWER_COL", "回答内容")  # 若不存在则用第一列
