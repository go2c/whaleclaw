"""Alibaba Qwen (通义千问) provider adapter (OpenAI-compatible DashScope)."""

from __future__ import annotations

from whaleclaw.providers.openai_compat import OpenAICompatProvider


class QwenProvider(OpenAICompatProvider):
    """DashScope API (qwen3.5-plus, qwen3-max, qwq-plus, qwen-max, qwen-plus)."""

    provider_name = "qwen"
    default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    env_key = "DASHSCOPE_API_KEY"
