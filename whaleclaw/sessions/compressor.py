"""L0/L1 hierarchical context compression.

Generates two layers of summaries for older conversation segments:

* **L0** (~100 tokens) — one-sentence abstract for quick context
* **L1** (~800 tokens) — structured overview with key facts, decisions,
  and tool usage

Summaries are persisted in SQLite (``context_summaries`` table) and
loaded on demand into the prompt by ``ContextWindow.trim_with_summaries``.

L2 is simply the raw message history already stored in the ``messages``
table — no extra processing needed.

If the LLM call fails, the compression attempt is silently abandoned;
the next turn will retry.  No rule-based fallback is generated.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from whaleclaw.providers.base import Message
from whaleclaw.sessions.context_window import _estimate_tokens
from whaleclaw.types import ProviderRateLimitError
from whaleclaw.utils.log import get_logger

if TYPE_CHECKING:
    from whaleclaw.providers.router import ModelRouter
    from whaleclaw.sessions.store import SessionStore

log = get_logger(__name__)

_L0_PROMPT = """\
用一句话概括以下对话的核心内容（不超过 50 字）。
直接输出摘要，不要加前缀。"""

_L1_PROMPT = """\
将以下对话压缩为结构化概要（约 300-500 字）。

要求：
1. 保留所有关键事实（用户身份、偏好、已完成的任务、重要决定）
2. 保留使用过的工具及其关键结果
3. 省略寒暄、重复内容和冗长的工具输出细节
4. 使用与原对话相同的语言
5. 直接输出概要文本，不要加标题或前缀"""

_L0_TARGET_TOKENS = 100
_L1_TARGET_TOKENS = 800
_MIN_MESSAGES_FOR_COMPRESSION = 8
_RATE_LIMIT_COOLDOWN_SECONDS = 90


def _format_transcript(messages: list[Message]) -> str:
    """Format messages into a readable transcript for the summarizer LLM."""
    lines: list[str] = []
    for m in messages:
        label = {"user": "用户", "assistant": "助手", "tool": "工具", "system": "系统"}.get(
            m.role, m.role
        )
        content = m.content.strip()
        if len(content) > 500:
            content = content[:500] + "…"
        lines.append(f"[{label}] {content}")
    return "\n".join(lines)


class ContextCompressor:
    """Generate and persist L0/L1 summaries for conversation segments."""

    def __init__(self, *, rate_limit_cooldown_seconds: int = _RATE_LIMIT_COOLDOWN_SECONDS) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._cooldown_until: dict[str, float] = {}
        self._rate_limit_cooldown_seconds = rate_limit_cooldown_seconds

    async def generate_l0(
        self,
        messages: list[Message],
        *,
        router: ModelRouter,
        model: str,
    ) -> str:
        """Generate L0 one-sentence abstract (~100 tokens)."""
        transcript = _format_transcript(messages)
        llm_msgs = [
            Message(role="system", content=_L0_PROMPT),
            Message(role="user", content=transcript),
        ]
        response = await router.chat(model, llm_msgs)
        return response.content.strip()

    async def generate_l1(
        self,
        messages: list[Message],
        *,
        router: ModelRouter,
        model: str,
    ) -> str:
        """Generate L1 structured overview (~800 tokens)."""
        transcript = _format_transcript(messages)
        llm_msgs = [
            Message(role="system", content=_L1_PROMPT),
            Message(role="user", content=transcript),
        ]
        response = await router.chat(model, llm_msgs)
        return response.content.strip()

    async def compress_segment(
        self,
        *,
        session_id: str,
        messages: list[Message],
        msg_id_start: int,
        msg_id_end: int,
        store: SessionStore,
        router: ModelRouter,
        model: str,
    ) -> bool:
        """Generate L0 + L1 summaries for a segment and persist to DB.

        Returns True on success, False if LLM call failed.
        """
        if len(messages) < _MIN_MESSAGES_FOR_COMPRESSION:
            return False
        now = time.monotonic()
        cooldown_until = self._cooldown_until.get(session_id, 0.0)
        if now < cooldown_until:
            log.info(
                "compressor.cooldown_skip",
                session_id=session_id,
                retry_after_s=int(cooldown_until - now),
            )
            return False

        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        if lock.locked():
            log.info("compressor.inflight_skip", session_id=session_id)
            return False

        async with lock:
            log.info(
                "compressor.generating",
                session_id=session_id,
                msg_range=f"{msg_id_start}-{msg_id_end}",
                message_count=len(messages),
            )

            try:
                l0 = await self.generate_l0(messages, router=router, model=model)
                l1 = await self.generate_l1(messages, router=router, model=model)
            except ProviderRateLimitError as exc:
                self._cooldown_until[session_id] = (
                    time.monotonic() + self._rate_limit_cooldown_seconds
                )
                log.warning(
                    "compressor.rate_limited",
                    session_id=session_id,
                    cooldown_s=self._rate_limit_cooldown_seconds,
                    error=str(exc),
                )
                return False
            except Exception as exc:
                log.warning("compressor.llm_failed", error=str(exc))
                return False

            l0_tokens = _estimate_tokens(l0)
            l1_tokens = _estimate_tokens(l1)

            await store.save_summary(
                session_id=session_id,
                level="L0",
                content=l0,
                source_msg_start=msg_id_start,
                source_msg_end=msg_id_end,
                token_count=l0_tokens,
            )
            await store.save_summary(
                session_id=session_id,
                level="L1",
                content=l1,
                source_msg_start=msg_id_start,
                source_msg_end=msg_id_end,
                token_count=l1_tokens,
            )

            log.info(
                "compressor.done",
                session_id=session_id,
                l0_tokens=l0_tokens,
                l1_tokens=l1_tokens,
            )
            return True

    @staticmethod
    def should_compress(message_count: int) -> bool:
        """Whether the conversation is long enough to warrant compression."""
        return message_count >= _MIN_MESSAGES_FOR_COMPRESSION * 2
