"""
SPSS Agent 编码系统 - Streamlit 前端
"""
import json
import sys
from pathlib import Path

# 确保项目根目录在 path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from agents.scoring_agent import load_rubric
from tools.batch_processor import AgentPipeline
from tools.evaluation.data_manager import EvaluationDataManager
from tools.evaluation.evaluator import AgentEvaluator

# ═══════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════

st.set_page_config(
    page_title="SPSS Agent 编码系统",
    page_icon="📋",
    layout="wide",
)

st.title("🧪 心理学问卷自动编码系统")
st.caption("基于大语言模型的开放题自动打分")

# ═══════════════════════════════════════
# 侧边栏：模式选择
# ═══════════════════════════════════════

with st.sidebar:
    st.header("⚙️ 设置")

    mode = st.radio(
        "选择模式",
        ["🧪 单条测试", "📤 批量打分", "📊 评估模式"],
    )

    st.divider()
    st.caption("当前模型：通义千问")
    st.caption("评分维度：0/1/2分")

# ═══════════════════════════════════════
# 模式一：单条测试
# ═══════════════════════════════════════

if mode == "🧪 单条测试":
    st.subheader("单条文本测试")

    text_input = st.text_area(
        "输入回答内容",
        placeholder="例如：他以为是女人的围巾...",
        height=100,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("开始评分", type="primary")

    if run_btn and text_input:
        with st.spinner("Agent评分中..."):
            rubric = load_rubric(config.RUBRIC_PATH)
            pipeline = AgentPipeline(rubric)
            result = pipeline.score_single(text_input)

        # 显示结果
        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            score = result.get("score", 0)
            score_label = {
                0: "无心理化",
                1: "心理状态",
                2: "错误信念",
                999: "缺失值",
            }.get(score, "未知")
            st.metric("评分结果", f"{score}分", score_label)

        with col2:
            conf = result.get("confidence", 0)
            st.metric(
                "置信度",
                f"{conf:.0%}",
                "高" if conf >= 0.8 else "低",
            )

        with col3:
            st.metric("处理时间", f"{result.get('time', 0):.1f}s")

        st.info(f"**判分理由**：{result.get('reason', '无')}")

# ═══════════════════════════════════════
# 模式二：批量打分
# ═══════════════════════════════════════

elif mode == "📤 批量打分":
    st.subheader("批量文件打分")

    uploaded_file = st.file_uploader(
        "上传文件",
        type=["csv", "xlsx"],
        help="文件需包含：编号、回答内容两列",
    )

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"✅ 读取成功：{len(df)}条数据")

        with st.expander("预览数据（前5行）"):
            st.dataframe(df.head())

        if st.button("开始批量打分", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("批量处理中..."):
                rubric = load_rubric(config.RUBRIC_PATH)
                pipeline = AgentPipeline(rubric)

                id_col = "编号" if "编号" in df.columns else df.columns[0]
                answer_col = "回答内容" if "回答内容" in df.columns else df.columns[0]

                results = []
                for i, row in df.iterrows():
                    result = pipeline.score_single(
                        str(row.get(answer_col, row.get("回答内容", "")))
                    )
                    results.append(result)
                    progress_bar.progress((i + 1) / len(df))
                    status_text.text(f"处理中：{i+1}/{len(df)}")

                result_df = df.copy()
                result_df["AI评分"] = [r["score"] for r in results]
                result_df["置信度"] = [r["confidence"] for r in results]
                result_df["判分理由"] = [r["reason"] for r in results]

            status_text.text("✅ 处理完成！")

            st.subheader("📊 结果统计")

            col1, col2 = st.columns(2)

            with col1:
                score_dist = result_df["AI评分"].value_counts()
                fig = px.pie(
                    values=score_dist.values,
                    names=[f"{k}分" for k in score_dist.index],
                    title="评分分布",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig2 = px.histogram(
                    result_df, x="置信度", nbins=10, title="置信度分布"
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("📋 详细结果")
            st.dataframe(result_df, use_container_width=True)

            csv = result_df.to_csv(
                index=False, encoding="utf-8-sig"
            ).encode("utf-8-sig")
            st.download_button(
                "⬇️ 下载结果CSV", csv, "scored_results.csv", "text/csv"
            )

# ═══════════════════════════════════════
# 模式三：评估模式
# ═══════════════════════════════════════

elif mode == "📊 评估模式":
    st.subheader("Agent性能评估")

    tab1, tab2 = st.tabs(["📈 运行评估", "📋 历史记录"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            version_name = st.text_input(
                "版本名称", placeholder="v1.2-new-rules"
            )

        with col2:
            dataset = st.selectbox(
                "数据集",
                ["dev（开发集）", "holdout（保留集）", "test（测试集）"],
            )

        if st.button("运行评估", type="primary"):
            if not version_name:
                st.error("请输入版本名称")
            else:
                dataset_key = dataset.split("（")[0]

                with st.spinner("评估中..."):
                    data_mgr = EvaluationDataManager()

                    if dataset_key == "dev":
                        gt_df = data_mgr.get_dev_set()
                    elif dataset_key == "holdout":
                        gt_df = data_mgr.get_holdout_set()
                    else:
                        gt_df = data_mgr.get_test_set()

                    rubric = load_rubric(config.RUBRIC_PATH)
                    scenario = None
                    scenario_path = "prompts/scenario.txt"
                    if Path(scenario_path).exists():
                        with open(scenario_path, "r", encoding="utf-8") as f:
                            scenario = f.read()

                    pipeline = AgentPipeline(rubric, scenario)
                    input_df = gt_df[["编号", "回答内容"]].copy()
                    result_df = pipeline.process_dataframe(input_df)

                    evaluator = AgentEvaluator()
                    metrics = evaluator.evaluate(
                        result_df, gt_df, version_name, ""
                    )

                st.subheader("核心指标")

                col1, col2, col3, col4 = st.columns(4)

                kappa = metrics["kappa"]
                accuracy = metrics["accuracy"]["overall"]
                mae = metrics.get("error_analysis", {}).get(
                    "mean_absolute_error", 0
                )
                n_valid = metrics.get("meta", {}).get("n_samples", 0)

                with col1:
                    st.metric(
                        "Kappa系数",
                        f"{kappa:.3f}",
                        "优秀"
                        if kappa >= 0.8
                        else "良好"
                        if kappa >= 0.7
                        else "需改进",
                    )
                with col2:
                    st.metric("准确率", f"{accuracy:.1%}")
                with col3:
                    st.metric("平均误差", f"{mae:.2f}分")
                with col4:
                    st.metric("有效样本", f"{n_valid}条")

                st.subheader("混淆矩阵")

                cm = metrics.get("confusion_matrix", [])
                if cm:
                    fig = px.imshow(
                        cm,
                        labels=dict(x="AI评分", y="人工评分", color="数量"),
                        x=["0分", "1分", "2分"],
                        y=["0分", "1分", "2分"],
                        color_continuous_scale="Blues",
                        title="混淆矩阵",
                    )
                    st.plotly_chart(fig)

    with tab2:
        st.subheader("历史版本对比")

        try:
            history_path = (
                Path(_PROJECT_ROOT)
                / "data"
                / "evaluation"
                / "results"
                / "evaluation_history.json"
            )
            if not history_path.exists():
                st.info("暂无历史记录，请先运行评估")
            else:
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)

                evals = history.get("evaluations", [])
                if not evals:
                    st.info("暂无历史记录，请先运行评估")
                else:
                    hist_df = pd.DataFrame(
                        [
                            {
                                "version": e["meta"]["version_name"],
                                "kappa": e["kappa"],
                                "accuracy": e["accuracy"]["overall"],
                            }
                            for e in evals
                        ]
                    )

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=hist_df["version"],
                            y=hist_df["kappa"],
                            mode="lines+markers",
                            name="Kappa",
                            line=dict(color="blue"),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=hist_df["version"],
                            y=hist_df["accuracy"],
                            mode="lines+markers",
                            name="准确率",
                            line=dict(color="green"),
                        )
                    )
                    fig.add_hline(
                        y=0.7,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="论文最低线(0.70)",
                    )
                    fig.update_layout(
                        title="版本迭代进度",
                        xaxis_title="版本",
                        yaxis_title="指标值",
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(hist_df)

        except Exception as e:
            st.warning(f"加载历史失败：{e}")
