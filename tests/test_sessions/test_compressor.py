"""Tests for L0/L1 hierarchical context compression."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from whaleclaw.providers.base import AgentResponse, Message
from whaleclaw.sessions.compressor import (
    ContextCompressor,
    _format_transcript,
)
from whaleclaw.types import ProviderRateLimitError


@pytest.fixture
def mock_router() -> MagicMock:
    router = MagicMock()
    call_count = 0

    async def _chat_side_effect(*args: object, **kwargs: object) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 1:
            return AgentResponse(
                content="用户讨论了项目架构，决定使用微服务部署。",
                model="glm-4.7-flash",
                input_tokens=200,
                output_tokens=20,
            )
        return AgentResponse(
            content=(
                "关键事实：用户需要微服务架构，支持高并发。\n"
                "决定：使用 Kubernetes 部署，PostgreSQL 主库，Redis 缓存。\n"
                "工具使用：bash 执行了部署脚本。"
            ),
            model="glm-4.7-flash",
            input_tokens=500,
            output_tokens=60,
        )

    router.chat = AsyncMock(side_effect=_chat_side_effect)
    return router


@pytest.fixture
def sample_messages() -> list[Message]:
    return [
        Message(role="user", content="你好，我想讨论项目架构"),
        Message(role="assistant", content="好的，请告诉我你的项目需求。"),
        Message(role="user", content="我们需要一个微服务架构，支持高并发"),
        Message(role="assistant", content="建议使用 Kubernetes 部署，配合消息队列实现异步处理。"),
        Message(role="user", content="那数据库怎么选？"),
        Message(role="assistant", content="PostgreSQL 作为主库，Redis 作为缓存层。"),
        Message(role="user", content="部署环境呢？"),
        Message(role="assistant", content="建议使用 Docker + K8s，CI/CD 用 GitHub Actions。"),
    ]


class TestFormatTranscript:
    def test_formats_roles(self) -> None:
        msgs = [
            Message(role="user", content="你好"),
            Message(role="assistant", content="你好！"),
            Message(role="tool", content="[bash] ok"),
        ]
        text = _format_transcript(msgs)
        assert "[用户]" in text
        assert "[助手]" in text
        assert "[工具]" in text

    def test_truncates_long_content(self) -> None:
        msgs = [Message(role="user", content="x" * 1000)]
        text = _format_transcript(msgs)
        assert len(text) < 600
        assert "…" in text


class TestContextCompressor:
    @pytest.mark.asyncio
    async def test_generate_l0(
        self,
        sample_messages: list[Message],
        mock_router: MagicMock,
    ) -> None:
        compressor = ContextCompressor()
        l0 = await compressor.generate_l0(
            sample_messages, router=mock_router, model="zhipu/glm-4.7-flash",
        )
        assert l0
        assert len(l0) < 500

    @pytest.mark.asyncio
    async def test_generate_l1(
        self,
        sample_messages: list[Message],
        mock_router: MagicMock,
    ) -> None:
        compressor = ContextCompressor()
        # First call returns L0-style, second returns L1-style
        await compressor.generate_l0(
            sample_messages, router=mock_router, model="zhipu/glm-4.7-flash",
        )
        l1 = await compressor.generate_l1(
            sample_messages, router=mock_router, model="zhipu/glm-4.7-flash",
        )
        assert l1
        assert "Kubernetes" in l1 or "关键" in l1

    @pytest.mark.asyncio
    async def test_compress_segment_persists_to_db(
        self,
        sample_messages: list[Message],
        mock_router: MagicMock,
    ) -> None:
        from pathlib import Path
        import tempfile

        from whaleclaw.sessions.store import SessionStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(db_path=Path(tmpdir) / "test.db")
            await store.open()
            try:
                await store.save_session(
                    session_id="s1",
                    channel="test",
                    peer_id="p1",
                    model="test",
                    created_at="2026-01-01",
                    updated_at="2026-01-01",
                )

                compressor = ContextCompressor()
                ok = await compressor.compress_segment(
                    session_id="s1",
                    messages=sample_messages,
                    msg_id_start=1,
                    msg_id_end=8,
                    store=store,
                    router=mock_router,
                    model="zhipu/glm-4.7-flash",
                )
                assert ok

                summaries = await store.get_summaries("s1")
                assert len(summaries) == 2

                l0_rows = [s for s in summaries if s.level == "L0"]
                l1_rows = [s for s in summaries if s.level == "L1"]
                assert len(l0_rows) == 1
                assert len(l1_rows) == 1
                assert l0_rows[0].source_msg_start == 1
                assert l0_rows[0].source_msg_end == 8
                assert l0_rows[0].token_count > 0
            finally:
                await store.close()

    @pytest.mark.asyncio
    async def test_compress_segment_fails_on_llm_error(
        self,
        sample_messages: list[Message],
    ) -> None:
        from pathlib import Path
        import tempfile

        from whaleclaw.sessions.store import SessionStore

        router = MagicMock()
        router.chat = AsyncMock(side_effect=RuntimeError("API error"))

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(db_path=Path(tmpdir) / "test.db")
            await store.open()
            try:
                await store.save_session(
                    session_id="s1",
                    channel="test",
                    peer_id="p1",
                    model="test",
                    created_at="2026-01-01",
                    updated_at="2026-01-01",
                )

                compressor = ContextCompressor()
                ok = await compressor.compress_segment(
                    session_id="s1",
                    messages=sample_messages,
                    msg_id_start=1,
                    msg_id_end=8,
                    store=store,
                    router=router,
                    model="zhipu/glm-4.7-flash",
                )
                assert not ok

                summaries = await store.get_summaries("s1")
                assert len(summaries) == 0
            finally:
                await store.close()

    @pytest.mark.asyncio
    async def test_too_few_messages_skips_compression(
        self,
        mock_router: MagicMock,
    ) -> None:
        from pathlib import Path
        import tempfile

        from whaleclaw.sessions.store import SessionStore

        short = [Message(role="user", content="hi")]

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(db_path=Path(tmpdir) / "test.db")
            await store.open()
            try:
                compressor = ContextCompressor()
                ok = await compressor.compress_segment(
                    session_id="s1",
                    messages=short,
                    msg_id_start=1,
                    msg_id_end=1,
                    store=store,
                    router=mock_router,
                    model="test",
                )
                assert not ok
                mock_router.chat.assert_not_called()
            finally:
                await store.close()

    @pytest.mark.asyncio
    async def test_rate_limit_enters_cooldown_and_skips_retry(
        self,
        sample_messages: list[Message],
    ) -> None:
        from pathlib import Path
        import tempfile

        from whaleclaw.sessions.store import SessionStore

        router = MagicMock()
        router.chat = AsyncMock(side_effect=ProviderRateLimitError("zhipu API 速率限制"))

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(db_path=Path(tmpdir) / "test.db")
            await store.open()
            try:
                await store.save_session(
                    session_id="s1",
                    channel="test",
                    peer_id="p1",
                    model="test",
                    created_at="2026-01-01",
                    updated_at="2026-01-01",
                )

                compressor = ContextCompressor(rate_limit_cooldown_seconds=60)
                ok1 = await compressor.compress_segment(
                    session_id="s1",
                    messages=sample_messages,
                    msg_id_start=1,
                    msg_id_end=8,
                    store=store,
                    router=router,
                    model="zhipu/glm-4.7-flash",
                )
                ok2 = await compressor.compress_segment(
                    session_id="s1",
                    messages=sample_messages,
                    msg_id_start=1,
                    msg_id_end=8,
                    store=store,
                    router=router,
                    model="zhipu/glm-4.7-flash",
                )

                assert not ok1
                assert not ok2
                # second call should be skipped by cooldown, not calling LLM again
                assert router.chat.await_count == 1
            finally:
                await store.close()

    def test_should_compress(self) -> None:
        compressor = ContextCompressor()
        assert not compressor.should_compress(10)
        assert compressor.should_compress(20)


class TestSessionStoreSummaries:
    """Test the summary CRUD methods on SessionStore."""

    @pytest.mark.asyncio
    async def test_save_and_get_summaries(self) -> None:
        from pathlib import Path
        import tempfile

        from whaleclaw.sessions.store import SessionStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(db_path=Path(tmpdir) / "test.db")
            await store.open()
            try:
                await store.save_session(
                    session_id="s1",
                    channel="test",
                    peer_id="p1",
                    model="test",
                    created_at="2026-01-01",
                    updated_at="2026-01-01",
                )

                await store.save_summary(
                    session_id="s1", level="L0", content="摘要",
                    source_msg_start=1, source_msg_end=10, token_count=50,
                )
                await store.save_summary(
                    session_id="s1", level="L1", content="详细概要",
                    source_msg_start=1, source_msg_end=10, token_count=200,
                )

                all_s = await store.get_summaries("s1")
                assert len(all_s) == 2

                l0_only = await store.get_summaries("s1", level="L0")
                assert len(l0_only) == 1
                assert l0_only[0].level == "L0"
                assert l0_only[0].content == "摘要"

                latest = await store.get_latest_summary("s1", "L0")
                assert latest is not None
                assert latest.source_msg_end == 10
            finally:
                await store.close()

    @pytest.mark.asyncio
    async def test_delete_summaries(self) -> None:
        from pathlib import Path
        import tempfile

        from whaleclaw.sessions.store import SessionStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(db_path=Path(tmpdir) / "test.db")
            await store.open()
            try:
                await store.save_session(
                    session_id="s1",
                    channel="test",
                    peer_id="p1",
                    model="test",
                    created_at="2026-01-01",
                    updated_at="2026-01-01",
                )
                await store.save_summary(
                    session_id="s1", level="L0", content="摘要",
                    source_msg_start=1, source_msg_end=5, token_count=50,
                )
                await store.delete_summaries("s1")
                assert await store.get_summaries("s1") == []
            finally:
                await store.close()
