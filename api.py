"""
SPSS Agent 编码系统 - FastAPI 后端
"""
import io
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from agents.scoring_agent import load_rubric
from tools.batch_processor import AgentPipeline
from tools.evaluation.data_manager import EvaluationDataManager
from tools.evaluation.evaluator import AgentEvaluator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        rubric = load_rubric(config.RUBRIC_PATH)
        scenario = None
        scenario_path = "prompts/scenario.txt"
        if os.path.exists(scenario_path):
            with open(scenario_path, "r", encoding="utf-8") as f:
                scenario = f.read()
        _pipeline = AgentPipeline(rubric, scenario)
    return _pipeline


batch_jobs = {}

# ══════════════════════════════════════════
# 首页 — 返回前端 HTML
# ══════════════════════════════════════════


@app.get("/")
def root():
    return FileResponse("frontend/codemind_v2.html")


# ══════════════════════════════════════════
# 接口1：上传文件，返回列名
# ══════════════════════════════════════════


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传SAV/CSV/Excel文件
    返回：行数、列名列表
    """
    content = await file.read()
    ext = file.filename.split(".")[-1].lower()

    try:
        if ext == "csv":
            df_full = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
            df = df_full.head(5)
            total_rows = len(df_full)
        elif ext in ["xlsx", "xls"]:
            df_full = pd.read_excel(io.BytesIO(content))
            df = df_full.head(5)
            total_rows = len(df_full)
        elif ext == "sav":
            import pyreadstat

            tmp_path = f"/tmp/{uuid.uuid4()}.sav"
            with open(tmp_path, "wb") as f:
                f.write(content)
            df, meta = pyreadstat.read_sav(tmp_path)
            os.remove(tmp_path)
            total_rows = len(df)
        else:
            return JSONResponse(
                {"error": "不支持的文件格式"}, status_code=400
            )

        text_col_guess = None
        max_len = 0
        for col in df.columns:
            if df[col].dtype == object:
                avg_len = df[col].dropna().astype(str).str.len().mean()
                if avg_len and avg_len > max_len:
                    max_len = avg_len
                    text_col_guess = col

        label_col_guess = None
        for col in df.columns:
            if str(df[col].dtype) in ["int64", "float64"]:
                vals = df[col].dropna()
                if len(vals) > 0 and vals.max() <= 999 and vals.min() >= 0:
                    label_col_guess = col
                    break

        save_path = f"data/uploads/{uuid.uuid4()}_{file.filename}"
        Path("data/uploads").mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(content)

        return {
            "filename": file.filename,
            "total_rows": total_rows,
            "columns": list(df.columns),
            "text_col_guess": text_col_guess,
            "label_col_guess": label_col_guess,
            "file_path": save_path,
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ══════════════════════════════════════════
# 接口2：单条编码
# ══════════════════════════════════════════


class SingleRequest(BaseModel):
    text: str
    rubric: str = ""


@app.post("/api/score/single")
def score_single(req: SingleRequest):
    """单条文本编码"""
    if req.rubric:
        tmp_pipeline = AgentPipeline(req.rubric)
        result = tmp_pipeline.score_single(req.text)
    else:
        pipeline = get_pipeline()
        result = pipeline.score_single(req.text)

    score = result.get("score", 0)
    labels = {0: "无心理化", 1: "心理状态", 2: "错误信念", 999: "无效值"}

    return {
        "score": score,
        "label": labels.get(score, "未知"),
        "confidence": result.get("confidence", 0),
        "reason": result.get("reason", ""),
    }


# ══════════════════════════════════════════
# 接口3：批量编码（异步）
# ══════════════════════════════════════════


class BatchRequest(BaseModel):
    file_path: str
    text_col: str
    id_col: str = ""
    version: str = "v1.0"


@app.post("/api/score/batch/start")
def batch_start(req: BatchRequest, background_tasks: BackgroundTasks):
    """启动批量任务"""
    job_id = str(uuid.uuid4())
    batch_jobs[job_id] = {
        "status": "running",
        "done": 0,
        "total": 0,
        "results": [],
    }
    background_tasks.add_task(run_batch_job, job_id, req)
    return {"job_id": job_id}


def run_batch_job(job_id: str, req: BatchRequest):
    """后台执行批量编码"""
    try:
        ext = req.file_path.split(".")[-1].lower()
        if ext == "csv":
            df = pd.read_csv(req.file_path, encoding="utf-8-sig")
        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(req.file_path)
        elif ext == "sav":
            import pyreadstat

            df, _ = pyreadstat.read_sav(req.file_path)
        else:
            raise ValueError("不支持的文件格式")

        batch_jobs[job_id]["total"] = len(df)

        pipeline = get_pipeline()
        results = []

        for i, row in df.iterrows():
            text = str(row.get(req.text_col, ""))
            result = pipeline.score_single(text)

            item_id = (
                str(row.get(req.id_col, i + 1))
                if req.id_col
                else str(i + 1)
            )
            results.append(
                {
                    "id": item_id,
                    "text": text[:60],
                    "score": result.get("score", 0),
                    "confidence": round(result.get("confidence", 0) * 100),
                    "reason": result.get("reason", ""),
                }
            )

            batch_jobs[job_id]["done"] = i + 1
            batch_jobs[job_id]["results"] = results

        result_df = df.copy()
        result_df["AI评分"] = [r["score"] for r in results]
        result_df["置信度"] = [r["confidence"] for r in results]
        result_df["判分理由"] = [r["reason"] for r in results]

        out_path = f"data/output/{req.version}_results.csv"
        Path("data/output").mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        batch_jobs[job_id]["status"] = "done"
        batch_jobs[job_id]["out_path"] = out_path

    except Exception as e:
        batch_jobs[job_id]["status"] = "error"
        batch_jobs[job_id]["error"] = str(e)


@app.get("/api/score/batch/progress/{job_id}")
def batch_progress(job_id: str):
    """轮询批量任务进度"""
    job = batch_jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    return job


@app.get("/api/download/{version}")
def download_result(version: str):
    path = f"data/output/{version}_results.csv"
    if not Path(path).exists():
        return JSONResponse({"error": "文件不存在"}, status_code=404)
    return FileResponse(path, filename=f"{version}_results.csv")


# ══════════════════════════════════════════
# 接口4：评估
# ══════════════════════════════════════════


class EvalRequest(BaseModel):
    version: str
    dataset: str = "dev"


@app.post("/api/evaluate")
def evaluate(req: EvalRequest):
    """运行评估，返回 Kappa 等指标"""
    try:
        data_mgr = EvaluationDataManager()

        if req.dataset == "dev":
            gt_df = data_mgr.get_dev_set()
        elif req.dataset == "holdout":
            gt_df = data_mgr.get_holdout_set()
        else:
            gt_df = data_mgr.get_test_set()

        input_df = gt_df[["编号", "回答内容"]].copy()
        pipeline = get_pipeline()
        result_df = pipeline.process_dataframe(input_df)

        evaluator = AgentEvaluator()
        metrics = evaluator.evaluate(result_df, gt_df, req.version, "")
    except FileNotFoundError:
        return JSONResponse(
            {"error": "评估数据集未初始化，请先运行 python evaluate.py --setup --sav-file 您的数据.sav"},
            status_code=404,
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    err = metrics.get("error_analysis", {})
    meta = metrics.get("meta", {})
    conf = metrics.get("confidence_analysis", {})

    return {
        "version": req.version,
        "dataset": req.dataset,
        "kappa": round(metrics["kappa"], 3),
        "accuracy": round(metrics["accuracy"]["overall"], 3),
        "mae": round(err.get("mean_absolute_error", 0), 3),
        "n_valid": meta.get("n_samples", 0),
        "mean_confidence": round(conf.get("mean", 0), 3),
        "by_score": metrics.get("accuracy", {}).get("by_score", {}),
        "confusion_matrix": metrics.get("confusion_matrix", []),
    }


@app.get("/api/history")
def get_history():
    """历史版本记录"""
    try:
        path = "data/evaluation/results/evaluation_history.json"
        if Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"evaluations": []}
    except Exception:
        return {"evaluations": []}
