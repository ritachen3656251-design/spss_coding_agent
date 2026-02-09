"""
Agent 基类：提供 LLM 调用与日志接口
"""
import json
import re
from tools.llm_client import LLMClient


class BaseAgent:
    def __init__(self, name: str = "BaseAgent", llm_client=None):
        self.name = name
        self.llm_client = llm_client if llm_client is not None else LLMClient(model_type="qwen")

    def log(self, msg: str) -> None:
        print(f"[{self.name}] {msg}")

    def call_llm_with_json(self, system_prompt: str, user_message: str) -> dict:
        """
        调用 LLM 并解析 JSON 返回。
        返回：{"success": bool, "data": dict | None}
        """
        out = self.llm_client.call(system_prompt, user_message)
        if not out["success"] or not out.get("response"):
            return {"success": False, "data": None}
        raw = out["response"].strip()
        # 尝试从回复中提取 JSON 块
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return {"success": True, "data": data}
            except json.JSONDecodeError:
                pass
        return {"success": False, "data": None}
