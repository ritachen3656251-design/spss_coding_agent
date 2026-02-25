# SPSS Agent System - 智能问卷编码系统

基于多 Agent 的问卷文本智能编码系统，完整流程：**SAV → CSV → 预处理 → 智能打分 → 质量检查 → 分析报告**。

支持 0/1/2/999 四级心理化编码（无心理化、心理状态、错误信念、无效值），适用于心理量表、开放题编码等场景。

---

## 功能特性

- **多 Agent 流水线**：预处理 → 打分 → 质量检查 → 分析，全流程自动化
- **双模型验证**：主模型 Qwen + 辅助模型 DeepSeek，低置信度时自动二次验证
- **关键词回退**：对「无法播放」「看不了」等表述自动判为 999
- **ReAct 循环**：打分 Agent 支持思考→行动→观察→检查，提升推理质量
- **断点续传**：支持中断后从断点继续处理
- **评估体系**：dev/test/holdout 划分，Kappa、准确率、混淆矩阵等指标

---

## 环境与安装

**Python 3.10+**

```bash
pip install -r requirements.txt
```

**依赖**：dashscope、openai、pandas、pyreadstat、python-dotenv、scikit-learn、openpyxl、aiofiles、tqdm

---

## 配置

复制 `.env.example` 为 `.env`，并填入 API Key：

```bash
cp .env.example .env
```

```env
DASHSCOPE_API_KEY=your-dashscope-key      # Qwen 主模型（必填）
DEEPSEEK_API_KEY=your-deepseek-key        # 双模型验证（可选）
# 可选覆盖
# PRIMARY_MODEL=qwen-max
# SECONDARY_MODEL=deepseek-chat
```

---

## 使用方式

### 生产环境：`python main.py`

批量处理问卷，生成打分结果、分析报告、质量报告。

```bash
python main.py
```

**输入**：
- 优先使用 `data/input/questionnaire.csv`
- 若无 CSV，则自动在 `data/input/` 下查找 `.sav` 并转换为 questionnaire.csv

**参数**：
- `--async`：异步处理模式
- `--no-resume`：忽略断点，从头开始

**输出**：
- `data/output/scored_results.csv`：打分结果
- `data/output/scored_results_report.md`：分析报告
- `data/output/scored_results_quality.json`：质量报告

---

### 评估环境：`python evaluate.py`

对 Agent 版本进行评测，计算 Kappa、准确率等指标。

**1. 首次使用（初始化评估数据集）**

```bash
python evaluate.py --setup --sav-file path/to/your_data.sav
```

会生成 `data/evaluation/splits/` 下的 dev/test/holdout，以及 `data/production/input/questionnaire.csv`。

**2. 评估指定版本**

```bash
python evaluate.py --version v1.0
python evaluate.py --version v1.0 --dataset test --description "baseline"
```

**3. 对比所有版本**

```bash
python evaluate.py --compare
```

**输出**：
- `data/evaluation/results/{version}_detailed.csv`：详细结果
- `data/evaluation/results/{version}_errors.csv`：错误案例
- `data/evaluation/results/evaluation_history.json`：评估历史

---

## 目录结构

```
├── main.py              # 生产入口
├── evaluate.py          # 评估入口
├── config.py            # 项目配置
├── agents/              # Agent 模块
│   ├── preprocessing_agent.py
│   ├── scoring_agent.py
│   ├── quality_agent.py
│   ├── analysis_agent.py
│   └── ...
├── tools/               # 工具
│   ├── batch_processor.py
│   ├── data_converter.py
│   └── evaluation/      # 评估相关
├── prompts/             # 提示词与标准
│   ├── scoring_rubric.txt
│   └── scenario.txt
├── data/
│   ├── input/           # 生产输入（questionnaire.csv 或 .sav）
│   ├── output/          # 生产输出
│   ├── evaluation/      # 评估数据与结果
│   │   ├── splits/      # dev/test/holdout
│   │   └── results/
│   └── cache/           # 记忆与缓存
└── .env                 # API Key（需自行创建）
```

---

## 评分标准

在 `prompts/scoring_rubric.txt` 中配置：

- **0 分**：只有行为描述，无心理状态
- **1 分**：描述心理状态，但无错误识别
- **2 分**：明确人物对物体产生错误识别/信念
- **999 分**：无效值（如视频无法播放、跳过等）

---

## 生产流程概览

1. 将 SAV 放入 `data/input/`，或直接提供 `questionnaire.csv`
2. 运行 `python main.py`
3. 查看 `data/output/` 下的 CSV、报告和质量 JSON
4. 含 `NEEDS_REVIEW` 的条目需人工复核

---

## License

MIT
