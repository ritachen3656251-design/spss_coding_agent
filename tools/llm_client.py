"""
支持多模型的 LLM 客户端：Qwen（主）、DeepSeek（辅助验证）
"""
import os

import dashscope
from dashscope import Generation
from openai import OpenAI
import config


class LLMClient:
    """
    支持多模型的 LLM 客户端
    可以调用 Qwen 或 DeepSeek
    """

    def __init__(self, model_type: str = "qwen"):
        """
        参数:
        - model_type: "qwen" 或 "deepseek"
        """
        self.model_type = model_type

        if model_type == "qwen":
            dashscope.api_key = config.DASHSCOPE_API_KEY
            self.model = config.PRIMARY_MODEL
        elif model_type == "deepseek":
            api_key = config.DEEPSEEK_API_KEY or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "DeepSeek 需要 API Key，请在 .env 中设置 DEEPSEEK_API_KEY，"
                    "或将 OPENAI_API_KEY 设置为 DeepSeek 的 API Key"
                )
            self._client = OpenAI(
                api_key=api_key,
                base_url=config.DEEPSEEK_API_BASE,
            )
            self.model = config.SECONDARY_MODEL
        else:
            raise ValueError('model_type 须为 "qwen" 或 "deepseek"')

    def call(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0,
        max_tokens: int = 500,
    ) -> dict:
        """
        调用 LLM 并返回结构化结果

        返回格式：
        {
            "response": "原始响应文本",
            "success": True/False,
            "error": "错误信息（如果有）"
        }
        """
        try:
            if self.model_type == "qwen":
                response = Generation.call(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    result_format="message",
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if response.status_code == 200:
                    text = response.output.choices[0].message.content
                    return {"response": text, "success": True, "error": None}
                return {
                    "response": None,
                    "success": False,
                    "error": getattr(response, "message", str(response)),
                }

            elif self.model_type == "deepseek":
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                )
                text = response.choices[0].message.content
                return {"response": text, "success": True, "error": None}

        except Exception as e:
            return {
                "response": None,
                "success": False,
                "error": str(e),
            }

        return {"response": None, "success": False, "error": "未知 model_type"}
