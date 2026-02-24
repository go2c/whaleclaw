"""Context window management — lazy, on-demand compression.

Compression is applied in stages, stopping as soon as we're within budget:
  Stage 0: No compression — if everything fits, ship it as-is
  Stage 1: Compress tool outputs only (biggest per-message savings)
  Stage 2: Compress tool-call annotations
  Stage 3: Compress old user/assistant messages (light — keep first 2 lines)
  Stage 4: Hard truncate — drop oldest messages (no summary injected)

When L0/L1 summaries exist in the DB, ``trim_with_summaries`` injects them
into the prompt before Stage 1-4, giving the model awareness of prior
context even after truncation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from whaleclaw.providers.base import Message
from whaleclaw.utils.log import get_logger

if TYPE_CHECKING:
    from whaleclaw.sessions.store import SummaryRow

log = get_logger(__name__)

MODEL_MAX_CONTEXT: dict[str, int] = {
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    "gpt-5.2": 1_000_000,
    "gpt-5.2-codex": 1_000_000,
    "gpt-5.3-codex": 1_000_000,
    "deepseek-chat": 64_000,
    "deepseek-reasoner": 64_000,
    "qwen-max": 32_000,
    "qwen-plus": 131_072,
    "qwen-turbo": 131_072,
    "glm-5": 200_000,
    "glm-4.7": 200_000,
    "glm-4.7-flash": 200_000,
    "MiniMax-M2.5": 1_000_000,
    "MiniMax-M2.1": 1_000_000,
    "kimi-k2.5": 256_000,
    "kimi-k2-thinking": 128_000,
    "gemini-3.1-pro-preview": 1_000_000,
    "gemini-3-pro-preview": 1_000_000,
    "gemini-3-flash-preview": 1_000_000,
    "meta/llama-3.1-8b-instruct": 128_000,
}

_DEFAULT_CONTEXT = 128_000
_REPLY_RESERVE = 8_000
RECENT_PROTECTED = 6


def _estimate_tokens(text: str) -> int:
    """Quick token estimate: ~1.5 chars/token CJK, ~4 chars/token Latin."""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = len(text) - cjk
    return max(1, int(cjk / 1.5 + latin / 4))


def _total_tokens(msgs: list[Message]) -> int:
    return sum(_estimate_tokens(m.content) for m in msgs)


def _already_compressed(msg: Message) -> bool:
    return msg.content.endswith(" ...") or msg.content.endswith("已压缩)")


def _is_tool_output(msg: Message) -> bool:
    if _already_compressed(msg):
        return False
    return (
        msg.content.startswith("[")
        and "] " in msg.content[:60]
        and msg.role == "assistant"
    )


def _is_tool_call_annotation(msg: Message) -> bool:
    if _already_compressed(msg):
        return False
    return msg.role == "assistant" and msg.content.startswith("(调用工具:")


def _compress_tool_output(msg: Message) -> Message:
    """Compress tool output: keep first line + paths + status lines."""
    text = msg.content
    if _estimate_tokens(text) <= 300:
        return msg

    lines = text.split("\n")
    kept = [lines[0][:200]]
    for line in lines[1:]:
        s = line.strip()
        if not s:
            continue
        if (s.startswith("/") or "文件:" in s or "路径:" in s
                or "成功" in s or "失败" in s or "ERROR" in s):
            kept.append(s[:200])
    compressed = "\n".join(kept[:10])
    if len(compressed) >= len(text):
        return msg
    return Message(role=msg.role, content=compressed)


def _compress_tool_annotation(msg: Message) -> Message:
    if _estimate_tokens(msg.content) <= 80:
        return msg
    end = msg.content.find(")")
    if end > 0:
        return Message(role=msg.role, content=msg.content[:end + 1])
    return msg


def _compress_old_user_assistant(msg: Message) -> Message:
    """Light compression: keep first 2 meaningful lines."""
    if _already_compressed(msg):
        return msg
    text = msg.content.strip()
    if _estimate_tokens(text) <= 300:
        return msg
    lines = [l for l in text.split("\n") if l.strip()][:2]
    preview = "\n".join(lines)
    if len(preview) > 300:
        preview = preview[:300]
    return Message(role=msg.role, content=preview + " ...")


def _build_summary_message(summaries: list[SummaryRow], budget: int) -> Message | None:
    """Build a single system message from DB summaries, preferring L1 over L0.

    If L1 fits within *budget*, use it.  Otherwise fall back to L0.
    Returns None if no summaries available or none fit.
    """
    l0_parts: list[str] = []
    l1_parts: list[str] = []

    for s in summaries:
        if s.level == "L0":
            l0_parts.append(s.content)
        elif s.level == "L1":
            l1_parts.append(s.content)

    if l1_parts:
        l1_text = "\n\n".join(l1_parts)
        if _estimate_tokens(l1_text) <= budget:
            return Message(
                role="system",
                content=f"(早期对话概要：\n{l1_text}\n--- 以下为完整对话 ---)",
            )

    if l0_parts:
        l0_text = " ".join(l0_parts)
        if _estimate_tokens(l0_text) <= budget:
            return Message(
                role="system",
                content=f"(早期对话摘要：{l0_text}\n--- 以下为完整对话 ---)",
            )

    return None


class ContextWindow:
    """On-demand, lazy context window compression."""

    @staticmethod
    def get_max_context(model: str) -> int:
        return MODEL_MAX_CONTEXT.get(model, _DEFAULT_CONTEXT)

    def trim(
        self, messages: list[Message], model: str,
    ) -> list[Message]:
        """Fit messages into budget. Stage 4 = hard truncate (no summary)."""
        max_ctx = self.get_max_context(model)

        system: list[Message] = []
        non_system: list[Message] = []
        for m in messages:
            (system if m.role == "system" else non_system).append(m)

        budget = max_ctx - _total_tokens(system) - _REPLY_RESERVE
        if budget < 2000:
            budget = 2000

        if _total_tokens(non_system) <= budget:
            return [*system, *non_system]

        protected = min(RECENT_PROTECTED, len(non_system))
        split = len(non_system) - protected

        old = list(non_system[:split])
        recent = list(non_system[split:])

        old = self._stage_compress(old, recent, budget, _is_tool_output, _compress_tool_output)
        if _total_tokens(old) + _total_tokens(recent) <= budget:
            return [*system, *old, *recent]

        old = self._stage_compress(old, recent, budget, _is_tool_call_annotation, _compress_tool_annotation)
        if _total_tokens(old) + _total_tokens(recent) <= budget:
            return [*system, *old, *recent]

        old = self._stage_compress(
            old, recent, budget,
            lambda m: m.role in ("user", "assistant"),
            _compress_old_user_assistant,
        )
        if _total_tokens(old) + _total_tokens(recent) <= budget:
            return [*system, *old, *recent]

        # Stage 4: hard truncate — keep as many recent messages as fit
        all_msgs = [*old, *recent]
        kept: list[Message] = []
        used = 0
        for msg in reversed(all_msgs):
            cost = _estimate_tokens(msg.content)
            if used + cost > budget:
                break
            kept.append(msg)
            used += cost
        kept.reverse()

        return [*system, *kept]

    def trim_with_summaries(
        self,
        messages: list[Message],
        model: str,
        summaries: list[SummaryRow],
    ) -> list[Message]:
        """Fit messages into budget, injecting DB summaries when truncation occurs.

        Summaries are loaded hierarchically: try L1 first (richer context),
        fall back to L0 (cheaper) if L1 doesn't fit the remaining budget.
        """
        max_ctx = self.get_max_context(model)

        system: list[Message] = []
        non_system: list[Message] = []
        for m in messages:
            (system if m.role == "system" else non_system).append(m)

        budget = max_ctx - _total_tokens(system) - _REPLY_RESERVE
        if budget < 2000:
            budget = 2000

        if _total_tokens(non_system) <= budget:
            return [*system, *non_system]

        protected = min(RECENT_PROTECTED, len(non_system))
        split = len(non_system) - protected
        old = list(non_system[:split])
        recent = list(non_system[split:])

        old = self._stage_compress(old, recent, budget, _is_tool_output, _compress_tool_output)
        if _total_tokens(old) + _total_tokens(recent) <= budget:
            return [*system, *old, *recent]

        old = self._stage_compress(old, recent, budget, _is_tool_call_annotation, _compress_tool_annotation)
        if _total_tokens(old) + _total_tokens(recent) <= budget:
            return [*system, *old, *recent]

        old = self._stage_compress(
            old, recent, budget,
            lambda m: m.role in ("user", "assistant"),
            _compress_old_user_assistant,
        )
        if _total_tokens(old) + _total_tokens(recent) <= budget:
            return [*system, *old, *recent]

        # Stage 4: hard truncate + inject summaries
        all_msgs = [*old, *recent]
        kept: list[Message] = []
        used = 0
        for msg in reversed(all_msgs):
            cost = _estimate_tokens(msg.content)
            if used + cost > budget:
                break
            kept.append(msg)
            used += cost
        kept.reverse()

        result = list(system)

        if summaries:
            remaining_budget = budget - used
            summary_budget = min(remaining_budget // 3, 2000)
            if summary_budget > 50:
                summary_msg = _build_summary_message(summaries, summary_budget)
                if summary_msg:
                    result.append(summary_msg)

        result.extend(kept)
        return result

    @staticmethod
    def _stage_compress(
        old: list[Message],
        recent: list[Message],
        budget: int,
        match_fn: object,
        compress_fn: object,
    ) -> list[Message]:
        """Compress matching messages in old zone one by one, stopping when under budget."""
        from typing import Callable
        _match = match_fn  # type: Callable[[Message], bool]
        _compress = compress_fn  # type: Callable[[Message], Message]

        recent_cost = _total_tokens(recent)
        result = list(old)
        for i, msg in enumerate(result):
            if _total_tokens(result) + recent_cost <= budget:
                break
            if _match(msg):
                result[i] = _compress(msg)
        return result
