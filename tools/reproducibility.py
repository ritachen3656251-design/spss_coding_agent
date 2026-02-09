import hashlib
import json
from datetime import datetime

import config


class ReproducibilityTracker:
    """
    可重复性追踪器：
    记录所有影响结果的因素，确保研究可重复
    """

    def __init__(self):
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            "system_version": "1.0.0",
            "models": {
                "primary": config.PRIMARY_MODEL,
                "secondary": (
                    config.SECONDARY_MODEL
                    if config.ENABLE_DUAL_MODEL
                    else None
                ),
            },
            "config": {
                "confidence_threshold": config.CONFIDENCE_THRESHOLD,
                "dual_model_enabled": config.ENABLE_DUAL_MODEL,
                "keyword_fallback_enabled": config.ENABLE_KEYWORD_FALLBACK,
            },
            "data_hash": None,
            "rubric_hash": None,
        }

    def set_data_hash(self, input_file: str):
        """计算输入数据的哈希值"""
        with open(input_file, "rb") as f:
            self.metadata["data_hash"] = hashlib.md5(f.read()).hexdigest()

    def set_rubric_hash(self, rubric: str):
        """计算评分标准的哈希值"""
        self.metadata["rubric_hash"] = hashlib.md5(
            rubric.encode()
        ).hexdigest()

    def save_metadata(self, output_path: str):
        """保存元数据"""
        import os

        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"✓ 可重复性元数据已保存：{output_path}")
