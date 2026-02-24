"""Tests for context window management."""

from __future__ import annotations

from whaleclaw.providers.base import Message
from whaleclaw.sessions.context_window import ContextWindow


class TestContextWindow:
    def test_get_max_context_known(self) -> None:
        cw = ContextWindow()
        assert cw.get_max_context("claude-sonnet-4-20250514") == 200_000

    def test_get_max_context_unknown(self) -> None:
        cw = ContextWindow()
        assert cw.get_max_context("unknown-model") == 128_000

    def test_no_compression_when_within_budget(self) -> None:
        """All messages fit -> nothing should be changed."""
        cw = ContextWindow()
        msgs = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there"),
            Message(role="user", content="做个PPT"),
            Message(role="assistant", content="好的，PPT做好了"),
        ]
        trimmed = cw.trim(msgs, "glm-4.7")
        assert len(trimmed) == 5
        assert trimmed[1].content == "hello"
        assert trimmed[4].content == "好的，PPT做好了"

    def test_tool_output_compressed_when_needed(self) -> None:
        """Tool outputs in old zone get compressed when budget is tight."""
        cw = ContextWindow()
        long_tool = "[bash] " + "成功执行\n路径: /tmp/test\nlog output\n" * 200
        msgs = [
            Message(role="system", content="s" * 20000),
            Message(role="user", content="运行命令"),
            Message(role="assistant", content=long_tool),
            *[Message(role="user", content="对话内容 " * 200) for _ in range(40)],
            Message(role="user", content="最新消息"),
            Message(role="assistant", content="好的"),
        ]
        trimmed = cw.trim(msgs, "qwen-max")
        tool_msgs = [m for m in trimmed if "[bash]" in m.content]
        if tool_msgs:
            assert len(tool_msgs[0].content) < len(long_tool)

    def test_no_compression_when_fits(self) -> None:
        """Nothing compressed when everything fits within budget."""
        cw = ContextWindow()
        long_tool = "[bash] " + "log output line\n" * 500
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="运行命令"),
            Message(role="assistant", content=long_tool),
            Message(role="user", content="好"),
        ]
        trimmed = cw.trim(msgs, "glm-4.7")
        tool_msg = [m for m in trimmed if m.content.startswith("[bash]")][0]
        assert tool_msg.content == long_tool

    def test_recent_messages_protected(self) -> None:
        """Recent messages should not be compressed."""
        cw = ContextWindow()
        recent_content = "这是最近的重要消息，包含具体指令和要求" * 10
        msgs = [
            Message(role="system", content="s" * 20000),
            *[Message(role="user", content="old " * 200) for _ in range(50)],
            Message(role="user", content=recent_content),
            Message(role="assistant", content="收到"),
        ]
        trimmed = cw.trim(msgs, "qwen-max")
        assert trimmed[-1].content == "收到"
        assert trimmed[-2].content == recent_content

    def test_hard_truncate_no_summary(self) -> None:
        """Stage 4 should hard truncate without injecting any summary."""
        cw = ContextWindow()
        msgs = [
            Message(role="system", content="s" * 25000),
            *[Message(role="user", content="重要消息 " * 300) for _ in range(100)],
            Message(role="user", content="最新消息"),
        ]
        trimmed = cw.trim(msgs, "qwen-max")
        assert trimmed[-1].content == "最新消息"
        assert len(trimmed) < len(msgs)
        # No summary message should be injected
        has_summary = any("摘要" in m.content for m in trimmed if m.role == "system")
        system_msgs = [m for m in trimmed if m.role == "system"]
        # Only the original system message should remain
        assert all("s" * 100 in m.content or m.content == trimmed[0].content for m in system_msgs)


class TestTrimWithSummaries:
    """Test trim_with_summaries loads DB summaries on truncation."""

    def _make_summary_row(
        self, *, level: str, content: str, token_count: int,
    ) -> object:
        """Create a mock SummaryRow."""
        from unittest.mock import MagicMock
        row = MagicMock()
        row.level = level
        row.content = content
        row.source_msg_start = 1
        row.source_msg_end = 10
        row.token_count = token_count
        return row

    def test_no_truncation_returns_all(self) -> None:
        cw = ContextWindow()
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="hello"),
        ]
        summaries = [self._make_summary_row(level="L0", content="摘要", token_count=10)]
        trimmed = cw.trim_with_summaries(msgs, "glm-4.7", summaries)
        assert len(trimmed) == 2

    def test_injects_l1_summary_on_truncation(self) -> None:
        """When truncation occurs and L1 fits, inject L1."""
        cw = ContextWindow()
        msgs = [
            Message(role="system", content="s" * 25000),
            *[Message(role="user", content="重要消息 " * 300) for _ in range(100)],
            Message(role="user", content="最新消息"),
        ]
        summaries = [
            self._make_summary_row(level="L0", content="一句话摘要", token_count=10),
            self._make_summary_row(level="L1", content="详细概要内容", token_count=100),
        ]
        trimmed = cw.trim_with_summaries(msgs, "qwen-max", summaries)
        assert trimmed[-1].content == "最新消息"
        has_l1 = any("概要" in m.content for m in trimmed)
        assert has_l1

    def test_falls_back_to_l0_when_l1_too_large(self) -> None:
        """When L1 exceeds budget, fall back to L0."""
        cw = ContextWindow()
        msgs = [
            Message(role="system", content="s" * 25000),
            *[Message(role="user", content="重要消息 " * 300) for _ in range(100)],
            Message(role="user", content="最新消息"),
        ]
        summaries = [
            self._make_summary_row(level="L0", content="一句话", token_count=5),
            self._make_summary_row(level="L1", content="x" * 50000, token_count=15000),
        ]
        trimmed = cw.trim_with_summaries(msgs, "qwen-max", summaries)
        assert trimmed[-1].content == "最新消息"
        has_l0 = any("一句话" in m.content for m in trimmed)
        assert has_l0

    def test_empty_summaries_no_injection(self) -> None:
        """With empty summaries list, behave like plain hard truncate."""
        cw = ContextWindow()
        msgs = [
            Message(role="system", content="s" * 25000),
            *[Message(role="user", content="重要消息 " * 300) for _ in range(100)],
            Message(role="user", content="最新消息"),
        ]
        trimmed = cw.trim_with_summaries(msgs, "qwen-max", [])
        assert trimmed[-1].content == "最新消息"
        assert len(trimmed) < len(msgs)
