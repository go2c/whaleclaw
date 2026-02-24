"""MiniMax (海螺 AI) provider adapter (OpenAI-compatible).

MiniMax OpenAI-compatible API quirks:
- Base URL is ``https://api.minimax.io/v1`` (NOT api.minimax.chat)
- ``temperature`` must be in (0.0, 1.0] — 0 is rejected
- ``assistant`` message ``content`` must be a string, not null
- Image/audio inputs are NOT supported
- Streaming requires ``stream_options.include_usage=true`` to get token counts
- Stream usage may only contain ``total_tokens`` without the prompt/completion split
"""

from __future__ import annotations

from typing import Any

from whaleclaw.providers.base import Message, ToolSchema
from whaleclaw.providers.openai_compat import OpenAICompatProvider


class MiniMaxProvider(OpenAICompatProvider):
    """MiniMax API (MiniMax-M2.5, MiniMax-M2.1, etc.)."""

    provider_name = "minimax"
    default_base_url = "https://api.minimax.io/v1"
    env_key = "MINIMAX_API_KEY"

    def _build_body(
        self,
        messages: list[Message],
        model: str,
        tools: list[ToolSchema] | None,
    ) -> dict[str, Any]:
        body = super()._build_body(messages, model, tools)

        for msg in body["messages"]:
            if msg.get("role") == "assistant" and msg.get("content") is None:
                msg["content"] = ""

            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                text_parts = [
                    p["text"] for p in msg["content"]
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                msg["content"] = "\n".join(text_parts) if text_parts else ""

        if body.get("stream"):
            body["stream_options"] = {"include_usage": True}

        return body
