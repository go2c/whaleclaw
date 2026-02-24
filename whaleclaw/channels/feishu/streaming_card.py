"""Streaming card updater — simulates typing in Feishu cards."""

from __future__ import annotations

import json
import time
from typing import Any

from whaleclaw.channels.feishu.client import FeishuClient
from whaleclaw.utils.log import get_logger

log = get_logger(__name__)

_MIN_INTERVAL = 3.0


class StreamingCard:
    """Incrementally update a Feishu card message.

    Throttles updates to at most one every 3s to avoid API rate limits.
    """

    def __init__(self, client: FeishuClient, message_id: str) -> None:
        self._client = client
        self._message_id = message_id
        self._content = ""
        self._tool_sections: list[dict[str, Any]] = []
        self._last_update: float = 0

    def _build_card(self) -> str:
        elements: list[dict[str, Any]] = [
            {"tag": "div", "text": {"tag": "lark_md", "content": self._content or "思考中..."}},
        ]
        elements.extend(self._tool_sections)
        card: dict[str, Any] = {
            "config": {"wide_screen_mode": True},
            "elements": elements,
        }
        return json.dumps(card, ensure_ascii=False)

    async def update(self, content: str) -> None:
        """Update the card content (throttled)."""
        self._content = content
        now = time.monotonic()
        if now - self._last_update < _MIN_INTERVAL:
            return
        self._last_update = now
        await self._client.update_message(self._message_id, self._build_card())

    async def append_tool_call(
        self, tool_name: str, result: str
    ) -> None:
        """Append a tool-call section to the card."""
        self._tool_sections.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"**🔧 {tool_name}**\n```\n{result}\n```",
            },
        })
        await self._client.update_message(self._message_id, self._build_card())
        self._last_update = time.monotonic()

    async def finalize(self, content: str) -> None:
        """Final update — always sends regardless of throttle."""
        self._content = content
        await self._client.update_message(self._message_id, self._build_card())
