import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd

try:
    from tqdm.asyncio import tqdm as atqdm
    HAS_TQDM_ASYNC = hasattr(atqdm, "gather")
except (ImportError, AttributeError):
    HAS_TQDM_ASYNC = False

try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False


class AsyncBatchProcessor:
    """
    异步批处理器：
    1. 并发调用 API（提升速度 3–5 倍）
    2. 断点续传（崩溃后从断点继续）
    3. 实时进度保存
    """

    def __init__(self, pipeline, checkpoint_dir="data/cache/checkpoints"):
        self.pipeline = pipeline
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.semaphore = asyncio.Semaphore(5)
        self.rate_limit_delay = 0.1
        self._executor = ThreadPoolExecutor(max_workers=8)

    async def process_csv_async(
        self,
        input_csv: str,
        output_csv: str,
        id_col: str = "编号",
        answer_col: str = "回答内容",
        score_col: str = "编码分数",
        resume: bool = True,
    ) -> pd.DataFrame:
        """异步批量处理（支持断点续传）"""
        try:
            df = pd.read_csv(input_csv, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(input_csv, encoding="gbk")

        if answer_col not in df.columns:
            answer_col = df.columns[0]
            print(f"   使用列「{answer_col}」作为回答内容")
        if id_col not in df.columns:
            df["_row_id"] = range(1, len(df) + 1)
            id_col = "_row_id"
            print(f"   未找到编号列，使用行号作为编号")

        df[score_col] = None
        df["置信度"] = None
        df["判分理由"] = None
        df["使用策略"] = None
        df["质量标记"] = None

        total = len(df)

        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{os.path.basename(input_csv)}.json",
        )

        processed_ids, results_cache = self._load_checkpoint(checkpoint_file)

        if resume and processed_ids:
            remaining_indices = [
                i for i in df.index if str(df.at[i, id_col]) not in processed_ids
            ]
            print(f"📂 从检查点恢复：已完成 {len(processed_ids)}/{total}，剩余 {len(remaining_indices)} 条")
            for idx in df.index:
                if str(df.at[idx, id_col]) in processed_ids:
                    rid = str(df.at[idx, id_col])
                    if rid in results_cache:
                        r = results_cache[rid]
                        df.at[idx, score_col] = r.get("score")
                        df.at[idx, "置信度"] = r.get("confidence")
                        df.at[idx, "判分理由"] = r.get("reasoning", "")
                        df.at[idx, "使用策略"] = r.get("strategy", "")
                        df.at[idx, "质量标记"] = ", ".join(
                            r.get("quality_flags", [])
                        )
        else:
            remaining_indices = list(df.index)
            print(f"📂 开始新任务：共 {total} 条数据")

        if not remaining_indices:
            print("✓ 所有数据已处理完成")
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            return df

        tasks = []
        for idx in remaining_indices:
            row_id = str(df.at[idx, id_col])
            text = (
                str(df.at[idx, answer_col])
                if pd.notna(df.at[idx, answer_col])
                else ""
            )
            task = self._process_single_async(
                idx, row_id, text, checkpoint_file, id_col, answer_col
            )
            tasks.append(task)

        print(f"\n🚀 开始异步处理（并发数：5）...")
        if HAS_TQDM_ASYNC:
            gather_results = await atqdm.gather(*tasks, desc="处理进度")
        else:
            gather_results = await asyncio.gather(*tasks)

        for idx, result in gather_results:
            if result:
                df.at[idx, score_col] = result.get("score")
                df.at[idx, "置信度"] = result.get("confidence")
                df.at[idx, "判分理由"] = result.get("reasoning", "")
                df.at[idx, "使用策略"] = result.get("strategy", "")
                df.at[idx, "质量标记"] = ", ".join(
                    result.get("quality_flags", [])
                )

        df.to_csv(output_csv, index=False, encoding="utf-8-sig")

        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("✓ 检查点已清理")

        return df

    async def _process_single_async(
        self, idx, row_id, text, checkpoint_file, id_col, answer_col
    ):
        async with self.semaphore:
            try:
                await asyncio.sleep(self.rate_limit_delay)

                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    self._process_sync,
                    text,
                    row_id,
                )

                await self._save_checkpoint_async(
                    checkpoint_file, row_id, result
                )

                return (idx, result)

            except Exception as e:
                print(f"❌ 处理失败 [{row_id}]: {e}")
                return (idx, None)

    def _process_sync(self, text, row_id):
        preprocess_result = self.pipeline.preprocessing_agent.process(text)

        if preprocess_result["is_999"]:
            return {
                "score": 999,
                "reasoning": preprocess_result["reason"],
                "strategy": "auto_999",
                "confidence": 1.0,
                "quality_flags": [],
            }

        scoring_result = self.pipeline.scoring_agent.process(
            preprocess_result["text"], respondent_id=row_id
        )
        quality_result = self.pipeline.quality_agent.process(
            scoring_result, text, row_id
        )

        return quality_result

    def _load_checkpoint(self, checkpoint_file):
        if not os.path.exists(checkpoint_file):
            return set(), {}

        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)

            processed_ids = set(checkpoint.get("processed_ids", []))
            results_cache = checkpoint.get("results", {})

            return processed_ids, results_cache
        except Exception:
            return set(), {}

    async def _save_checkpoint_async(self, checkpoint_file, row_id, result):
        if HAS_AIOFILES:
            if os.path.exists(checkpoint_file):
                async with aiofiles.open(
                    checkpoint_file, "r", encoding="utf-8"
                ) as f:
                    content = await f.read()
                    checkpoint = json.loads(content)
            else:
                checkpoint = {
                    "processed_ids": [],
                    "results": {},
                    "last_updated": None,
                }

            checkpoint["processed_ids"].append(row_id)
            checkpoint["results"][row_id] = result
            checkpoint["last_updated"] = datetime.now().isoformat()

            async with aiofiles.open(
                checkpoint_file, "w", encoding="utf-8"
            ) as f:
                await f.write(
                    json.dumps(checkpoint, ensure_ascii=False, indent=2)
                )
        else:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
            else:
                checkpoint = {
                    "processed_ids": [],
                    "results": {},
                    "last_updated": None,
                }

            checkpoint["processed_ids"].append(row_id)
            checkpoint["results"][row_id] = result
            checkpoint["last_updated"] = datetime.now().isoformat()

            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def process_csv_with_async(pipeline, input_csv, output_csv, **kwargs):
    """同步接口包装（方便在非 async 环境调用）"""
    processor = AsyncBatchProcessor(pipeline)
    return asyncio.run(
        processor.process_csv_async(input_csv, output_csv, **kwargs)
    )
