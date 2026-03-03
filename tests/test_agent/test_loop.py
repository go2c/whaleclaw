"""Tests for the Agent main loop (mocked provider)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import whaleclaw.agent.loop as loop_mod
from whaleclaw.agent.loop import _is_image_generation_request, _parse_fallback_tool_calls, run_agent
from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.providers.base import AgentResponse, ToolCall
from whaleclaw.sessions.manager import Session
from whaleclaw.skills.parser import Skill, SkillParamGuard, SkillParamItem
from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult
from whaleclaw.tools.registry import ToolRegistry


def _make_router(
    chat_fn: Any = None,
    response: AgentResponse | None = None,
    native_tools: bool = True,
) -> MagicMock:
    """Build a mock ModelRouter with proper sync/async methods."""
    router = MagicMock()
    router.supports_native_tools = MagicMock(return_value=native_tools)
    if chat_fn is not None:
        router.chat = chat_fn
    elif response is not None:
        router.chat = AsyncMock(return_value=response)
    return router


@pytest.mark.asyncio
async def test_run_agent_returns_reply() -> None:
    mock_response = AgentResponse(
        content="你好！我是 WhaleClaw。",
        model="claude-sonnet-4-20250514",
        input_tokens=50,
        output_tokens=20,
    )

    router = _make_router(response=mock_response)

    result = await run_agent(
        message="你好",
        session_id="test-session",
        config=WhaleclawConfig(),
        router=router,
    )

    assert result == "你好！我是 WhaleClaw。"
    router.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_agent_retries_once_on_empty_reply_then_recovers() -> None:
    call_count = 0

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],  # noqa: ARG001
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AgentResponse(content="", model="test-model", input_tokens=0, output_tokens=0)
        return AgentResponse(content="请告诉我你要我做什么。", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    result = await run_agent(
        message="？？？",
        session_id="test-empty-retry",
        config=WhaleclawConfig(),
        router=router,
    )
    assert result == "请告诉我你要我做什么。"
    assert call_count == 2


@pytest.mark.asyncio
async def test_run_agent_returns_fallback_after_two_empty_replies() -> None:
    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],  # noqa: ARG001
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        return AgentResponse(content="", model="test-model", input_tokens=0, output_tokens=0)

    router = _make_router(chat_fn=fake_chat)
    result = await run_agent(
        message="？？？",
        session_id="test-empty-fallback",
        config=WhaleclawConfig(),
        router=router,
    )
    assert result == "我这边没收到模型有效回复。请再发一次需求，我会继续处理。"


@pytest.mark.asyncio
async def test_run_agent_streams() -> None:
    mock_response = AgentResponse(
        content="Hello world",
        model="claude-sonnet-4-20250514",
        input_tokens=10,
        output_tokens=5,
    )

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,
    ) -> AgentResponse:
        if on_stream:
            await on_stream("Hello ")
            await on_stream("world")
        return mock_response

    router = _make_router(chat_fn=fake_chat)

    chunks: list[str] = []

    async def collect(chunk: str) -> None:
        chunks.append(chunk)

    result = await run_agent(
        message="hi",
        session_id="test-session",
        config=WhaleclawConfig(),
        on_stream=collect,
        router=router,
    )

    assert result == "Hello world"
    assert chunks == ["Hello ", "world"]


class _EchoTool(Tool):
    """Dummy tool that echoes its input."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo",
            description="Echo text back.",
            parameters=[
                ToolParameter(
                    name="text", type="string", description="Text to echo."
                )
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output=kwargs.get("text", ""))


class _BrowserProbeTool(Tool):
    """Dummy browser tool to assert required browser arguments."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="browser",
            description="Probe browser arguments.",
            parameters=[
                ToolParameter(name="action", type="string", description="action"),
                ToolParameter(name="text", type="string", description="text"),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "")
        text = kwargs.get("text", "")
        if action == "search_images" and bool(text):
            return ToolResult(success=True, output=f"ok:{text}")
        return ToolResult(success=False, output="", error="bad args")


class _BrowserAlwaysFailTool(Tool):
    """Dummy browser tool that always fails."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="browser",
            description="Always fails.",
            parameters=[
                ToolParameter(name="action", type="string", description="action"),
                ToolParameter(name="text", type="string", description="text"),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:  # noqa: ARG002
        return ToolResult(success=False, output="", error="browser failed")


class _BashProbeTool(Tool):
    """Dummy bash tool to assert command arg exists."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bash",
            description="Probe bash arguments.",
            parameters=[ToolParameter(name="command", type="string", description="command")],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        command = str(kwargs.get("command", "")).strip()
        if command:
            return ToolResult(success=True, output=f"ok:{command}")
        return ToolResult(success=False, output="", error="bad command")


class _BashAlwaysFailTool(Tool):
    """Dummy bash tool that always fails."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bash",
            description="Always fails.",
            parameters=[ToolParameter(name="command", type="string", description="command")],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:  # noqa: ARG002
        return ToolResult(success=False, output="", error="bash failed")


class _PptEditNoopTool(Tool):
    """Dummy ppt_edit tool used for tool-selection assertions."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="ppt_edit",
            description="noop ppt edit.",
            parameters=[
                ToolParameter(name="path", type="string", description="path"),
                ToolParameter(name="slide_index", type="integer", description="index"),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:  # noqa: ARG002
        return ToolResult(success=True, output="ok")


class _PptEditBusinessNoHitTool(Tool):
    """Dummy ppt_edit business style tool that reports zero hit."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="ppt_edit",
            description="business no hit",
            parameters=[
                ToolParameter(name="path", type="string", description="path"),
                ToolParameter(name="slide_index", type="integer", description="index"),
                ToolParameter(name="action", type="string", description="action"),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:  # noqa: ARG002
        return ToolResult(
            success=True,
            output="已应用 /tmp/a.pptx 第 1 页商务风格，重设深色条 0 处",
        )


class _NameMemoryManager:
    def __init__(self, name: str = "") -> None:
        self.name = name
        self.set_calls = 0
        self.clear_calls = 0

    async def get_assistant_name(self) -> str:
        return self.name

    async def set_assistant_name(self, name: str, *, source: str = "manual") -> bool:  # noqa: ARG002
        self.name = name
        self.set_calls += 1
        return True

    async def clear_assistant_name(self) -> int:
        old = 1 if self.name else 0
        self.name = ""
        self.clear_calls += 1
        return old


@pytest.mark.asyncio
async def test_run_agent_tool_call_loop() -> None:
    """Agent should execute tools and loop back to LLM."""
    tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(id="tc_1", name="echo", arguments={"text": "hello"})
        ],
    )
    final_response = AgentResponse(
        content="Echo result: hello",
        model="test-model",
    )

    call_count = 0

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return final_response

    router = _make_router(chat_fn=fake_chat)

    registry = ToolRegistry()
    registry.register(_EchoTool())

    tool_calls_seen: list[str] = []
    tool_results_seen: list[bool] = []

    async def on_tc(name: str, _args: dict[str, Any]) -> None:
        tool_calls_seen.append(name)

    async def on_tr(name: str, result: ToolResult) -> None:
        tool_results_seen.append(result.success)

    result = await run_agent(
        message="echo hello",
        session_id="test-tool",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        on_tool_call=on_tc,
        on_tool_result=on_tr,
    )

    assert result == "Echo result: hello"
    assert call_count == 2
    assert tool_calls_seen == ["echo"]
    assert tool_results_seen == [True]


@pytest.mark.asyncio
async def test_run_agent_updates_assistant_name_from_user_message() -> None:
    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        system_text = messages[0].content if messages else ""
        return AgentResponse(content=system_text, model="test-model")

    cfg = WhaleclawConfig()
    cfg.agent.memory.enabled = False
    mm = _NameMemoryManager()
    router = _make_router(chat_fn=fake_chat)

    result = await run_agent(
        message="以后你叫旺财",
        session_id="test-rename",
        config=cfg,
        router=router,
        memory_manager=mm,  # type: ignore[arg-type]
    )

    assert "你是 旺财" in result
    assert mm.name == "旺财"
    assert mm.set_calls == 1


@pytest.mark.asyncio
async def test_run_agent_does_not_rename_on_plain_name_question() -> None:
    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        system_text = messages[0].content if messages else ""
        return AgentResponse(content=system_text, model="test-model")

    cfg = WhaleclawConfig()
    cfg.agent.memory.enabled = False
    mm = _NameMemoryManager("WhaleClaw")
    router = _make_router(chat_fn=fake_chat)

    result = await run_agent(
        message="你叫什么名字？",
        session_id="test-no-rename",
        config=cfg,
        router=router,
        memory_manager=mm,  # type: ignore[arg-type]
    )

    assert "你是 WhaleClaw" in result
    assert mm.set_calls == 0


@pytest.mark.asyncio
async def test_run_agent_unknown_tool() -> None:
    """Unknown tool should not crash, returns error to LLM."""
    tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(id="tc_bad", name="nonexistent", arguments={})
        ],
    )
    final_response = AgentResponse(
        content="I could not find that tool.",
        model="test-model",
    )

    call_count = 0

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return final_response

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()

    result = await run_agent(
        message="do something",
        session_id="test-unknown",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "I could not find that tool."
    assert call_count == 2


@pytest.mark.asyncio
async def test_run_agent_fallback_mode() -> None:
    """Provider without native tools: parse JSON from text output."""
    json_text = (
        '我来查一下。\n'
        '```json\n'
        '{"tool": "echo", "arguments": {"text": "hello"}}\n'
        '```'
    )
    tool_response = AgentResponse(
        content=json_text,
        model="test-model",
    )
    final_response = AgentResponse(
        content="查到了: hello",
        model="test-model",
    )

    call_count = 0

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return final_response

    router = _make_router(chat_fn=fake_chat, native_tools=False)

    registry = ToolRegistry()
    registry.register(_EchoTool())

    result = await run_agent(
        message="echo hello",
        session_id="test-fallback",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "查到了: hello"
    assert call_count == 2


@pytest.mark.asyncio
async def test_run_agent_retries_when_tool_args_invalid_then_succeeds() -> None:
    invalid_tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[ToolCall(id="tc_browser", name="browser", arguments={})],
    )
    valid_tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(
                id="tc_browser_2",
                name="browser",
                arguments={"action": "search_images", "text": "杨幂近照"},
            )
        ],
    )
    final_response = AgentResponse(
        content="已完成",
        model="test-model",
    )

    call_count = 0

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return invalid_tool_response
        if call_count == 2:
            return valid_tool_response
        return final_response

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_BrowserProbeTool())

    result = await run_agent(
        message="给我张杨幂近照",
        session_id="test-browser-repair",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "已完成"
    assert call_count == 3


def test_is_image_generation_request_matches_expected_queries() -> None:
    assert _is_image_generation_request("请帮我文生图，主题是赛博朋克街景") is True
    assert _is_image_generation_request("这张图做图生图，风格改成宫崎骏") is True
    assert _is_image_generation_request("帮我改这个 ppt 第三页文案") is False
    assert _is_image_generation_request("帮我测试一下 API key 是否可用") is False


@pytest.mark.asyncio
async def test_run_agent_circuit_breaker_blocks_repeated_browser_failures() -> None:
    browser_tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(
                id="tc_browser",
                name="browser",
                arguments={"action": "search_images", "text": "杨幂近照"},
            )
        ],
    )
    final_response = AgentResponse(
        content="改用 bash 处理",
        model="test-model",
    )

    call_count = 0
    prompts_seen: list[str] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        prompts_seen.append("\n".join(m.content for m in messages if hasattr(m, "content")))
        if call_count <= 2:
            return browser_tool_response
        return final_response

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_BrowserAlwaysFailTool())

    result = await run_agent(
        message="给我张杨幂近照",
        session_id="test-browser-circuit",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "改用 bash 处理"
    assert call_count == 3
    assert any("browser 工具连续失败，已自动熔断" in p for p in prompts_seen)


@pytest.mark.asyncio
async def test_run_agent_circuit_breaker_blocks_repeated_bash_failures() -> None:
    bash_tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(
                id="tc_bash",
                name="bash",
                arguments={"command": "python3 /tmp/a.py"},
            )
        ],
    )
    final_response = AgentResponse(
        content="改用 ppt_edit 处理",
        model="test-model",
    )

    call_count = 0
    prompts_seen: list[str] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        prompts_seen.append("\n".join(m.content for m in messages if hasattr(m, "content")))
        if call_count <= 3:
            return bash_tool_response
        return final_response

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_BashAlwaysFailTool())

    result = await run_agent(
        message="给第二页配图",
        session_id="test-bash-circuit",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "改用 ppt_edit 处理"
    assert call_count == 4
    assert any("同一 bash 命令模板已连续失败 3 次" in p for p in prompts_seen)


@pytest.mark.asyncio
async def test_run_agent_includes_ppt_edit_for_followup_office_message() -> None:
    captured_tool_names: list[str] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],  # noqa: ARG001
        *,
        tools: Any = None,
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        if isinstance(tools, list):
            for t in tools:
                if hasattr(t, "name"):
                    name = str(getattr(t, "name", "")).strip()
                    if name:
                        captured_tool_names.append(name)
                    continue
                if isinstance(t, dict):
                    name = str(t.get("name", "")).strip()
                    if not name and isinstance(t.get("function"), dict):
                        name = str(t["function"].get("name", "")).strip()
                    if name:
                        captured_tool_names.append(name)
        return AgentResponse(content="收到", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_BashProbeTool())
    registry.register(_PptEditNoopTool())

    now = datetime.now(UTC)
    session = Session(
        id="s-followup-office",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="anthropic/claude-sonnet-4-20250514",
        created_at=now,
        updated_at=now,
        metadata={"last_pptx_path": "/tmp/贵州2日游.pptx"},
    )

    result = await run_agent(
        message="第一页的黑色条不好看，换种格式",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    assert result == "收到"
    assert "ppt_edit" in captured_tool_names


@pytest.mark.asyncio
async def test_run_agent_requires_dark_bar_target_hit_for_ppt_edit() -> None:
    first = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(
                id="tc_ppt",
                name="ppt_edit",
                arguments={
                    "path": "/tmp/a.pptx",
                    "slide_index": 1,
                    "action": "apply_business_style",
                },
            )
        ],
    )
    second = AgentResponse(content="继续处理", model="test-model")
    call_count = 0
    prompts_seen: list[str] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        prompts_seen.append("\n".join(m.content for m in messages if hasattr(m, "content")))
        if call_count == 1:
            return first
        return second

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_PptEditBusinessNoHitTool())

    now = datetime.now(UTC)
    session = Session(
        id="s-dark-bar",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="anthropic/claude-sonnet-4-20250514",
        created_at=now,
        updated_at=now,
        metadata={"last_pptx_path": "/tmp/a.pptx"},
    )

    result = await run_agent(
        message="第一页封面的黑色横条不好看，换一种方式",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    assert result == "继续处理"
    assert any("未命中用户指定对象：黑色横条仍未被替换" in p for p in prompts_seen)


@pytest.mark.asyncio
async def test_run_agent_repairs_browser_query_without_action() -> None:
    tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[ToolCall(id="tc_browser", name="browser", arguments={"query": "杨幂近照"})],
    )
    final_response = AgentResponse(content="已完成", model="test-model")

    call_count = 0

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return final_response

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_BrowserProbeTool())

    result = await run_agent(
        message="给我张杨幂近照",
        session_id="test-browser-repair-query",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "已完成"
    assert call_count == 2


@pytest.mark.asyncio
async def test_run_agent_repairs_bash_cmd_alias() -> None:
    tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[ToolCall(id="tc_bash", name="bash", arguments={"cmd": "echo hi"})],
    )
    final_response = AgentResponse(content="done", model="test-model")

    call_count = 0

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return final_response

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_BashProbeTool())

    result = await run_agent(
        message="执行命令",
        session_id="test-bash-repair-cmd",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "done"
    assert call_count == 2


@pytest.mark.asyncio
async def test_run_agent_repairs_garbled_browser_query_to_user_message() -> None:
    tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(
                id="tc_browser",
                name="browser",
                arguments={"action": "search_images", "text": "2026 \\n0\\n0\\n0\\n0"},
            )
        ],
    )
    final_response = AgentResponse(content="ok", model="test-model")

    call_count = 0
    captured: list[str] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return final_response

    class _BrowserCaptureTool(Tool):
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                name="browser",
                description="capture",
                parameters=[
                    ToolParameter(name="action", type="string", description="action"),
                    ToolParameter(name="text", type="string", description="text"),
                ],
            )

        async def execute(self, **kwargs: Any) -> ToolResult:
            captured.append(str(kwargs.get("text", "")))
            return ToolResult(success=True, output="ok")

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_BrowserCaptureTool())

    result = await run_agent(
        message="给我杨幂新年写真高清图",
        session_id="test-browser-repair-garbled",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "ok"
    assert call_count == 2
    assert captured and captured[0] == "给我杨幂新年写真高清图"


@pytest.mark.asyncio
async def test_run_agent_rejects_escaped_block_file_edit_args() -> None:
    bad_file_edit = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(
                id="tc_edit",
                name="file_edit",
                arguments={
                    "path": "/tmp/a.py",
                    "old_string": "line1\\nline2\\nline3\\nline4",
                    "new_string": "x\\ny\\nz\\nw",
                },
            )
        ],
    )
    final_response = AgentResponse(content="我改用 file_write 重写脚本", model="test-model")

    call_count = 0
    tool_called = False

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return bad_file_edit
        return final_response

    class _FileEditProbeTool(Tool):
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                name="file_edit",
                description="probe file_edit",
                parameters=[
                    ToolParameter(name="path", type="string", description="path"),
                    ToolParameter(name="old_string", type="string", description="old"),
                    ToolParameter(name="new_string", type="string", description="new"),
                ],
            )

        async def execute(self, **kwargs: Any) -> ToolResult:  # noqa: ARG002
            nonlocal tool_called
            tool_called = True
            return ToolResult(success=True, output="edited")

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_FileEditProbeTool())

    result = await run_agent(
        message="重做这个 python 脚本",
        session_id="test-file-edit-escaped-block",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "我改用 file_write 重写脚本"
    assert call_count == 2
    assert not tool_called


class TestParseFallbackToolCalls:
    def test_fenced_json(self) -> None:
        text = '```json\n{"tool": "bash", "arguments": {"command": "ls"}}\n```'
        calls = _parse_fallback_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "bash"
        assert calls[0].arguments == {"command": "ls"}

    def test_bare_json(self) -> None:
        text = '好的，我来执行 {"tool": "bash", "arguments": {"command": "pwd"}}'
        calls = _parse_fallback_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "bash"

    def test_no_tool(self) -> None:
        text = "这是普通文本，没有工具调用。"
        calls = _parse_fallback_tool_calls(text)
        assert calls == []


class _DummyMemoryManager:
    def __init__(self, recalled: str = "") -> None:
        self._recalled = recalled
        self.recall_calls = 0
        self.capture_calls = 0
        self.capture_payloads: list[str] = []
        self.policy_calls = 0
        self.style_calls = 0

    def recall_policy(self, query: str) -> tuple[bool, bool]:  # noqa: ARG002
        self.policy_calls += 1
        return (True, True)

    async def get_global_style_directive(self) -> str:
        self.style_calls += 1
        return ""

    async def recall(  # noqa: PLR0913
        self,
        query: str,  # noqa: ARG002
        max_tokens: int = 500,  # noqa: ARG002
        limit: int = 10,  # noqa: ARG002
        *,
        include_profile: bool = True,
        include_raw: bool = True,
    ) -> str:
        self.recall_calls += 1
        if include_profile and not include_raw:
            return "【长期记忆画像】\n用户偏好简洁。"
        if include_raw and not include_profile:
            return self._recalled
        return self._recalled

    async def build_profile_for_injection(  # noqa: PLR0913
        self,
        *,
        max_tokens: int,  # noqa: ARG002
        router: Any = None,  # noqa: ARG002
        model_id: str = "",  # noqa: ARG002
    ) -> str:
        self.recall_calls += 1
        return "【长期记忆画像】\n用户偏好简洁。"

    async def auto_capture_user_message(  # noqa: PLR0913
        self,
        content: str,
        *,
        source: str,  # noqa: ARG002
        mode: str = "balanced",  # noqa: ARG002
        cooldown_seconds: int = 180,  # noqa: ARG002
        max_per_hour: int = 12,  # noqa: ARG002
        batch_size: int = 3,  # noqa: ARG002
        merge_window_seconds: int = 120,  # noqa: ARG002
    ) -> bool:
        self.capture_calls += 1
        self.capture_payloads.append(content)
        return True

    async def organize_if_needed(self, **kwargs: Any) -> bool:  # noqa: ARG002
        return False


@pytest.mark.asyncio
async def test_run_agent_injects_recalled_memory_into_system_prompt() -> None:
    captured_messages: list[Any] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        captured_messages[:] = messages
        return AgentResponse(content="收到", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    memory: Any = _DummyMemoryManager(recalled="- 用户喜欢简洁回答")
    cfg = WhaleclawConfig()
    cfg.agent.memory.organizer_background = False

    result = await run_agent(
        message="继续上次的话题",
        session_id="test-memory-recall",
        config=cfg,
        router=router,
        memory_manager=memory,
    )

    assert result == "收到"
    assert memory.policy_calls == 1
    assert memory.recall_calls == 2
    assert any(
        m.role == "system" and "长期记忆召回" in m.content
        for m in captured_messages
    )


@pytest.mark.asyncio
async def test_run_agent_auto_captures_user_fact_into_memory() -> None:
    router = _make_router(response=AgentResponse(content="记住了", model="test-model"))
    memory: Any = _DummyMemoryManager()
    cfg = WhaleclawConfig()
    cfg.agent.memory.organizer_background = False

    _ = await run_agent(
        message="我喜欢 Rust，请记住",
        session_id="test-memory-compact",
        config=cfg,
        router=router,
        memory_manager=memory,
    )

    assert memory.capture_calls == 1
    assert "我喜欢 Rust" in memory.capture_payloads[0]


@pytest.mark.asyncio
async def test_run_agent_skips_recall_when_policy_not_triggered() -> None:
    class _NoRecallMemory(_DummyMemoryManager):
        def recall_policy(self, query: str) -> tuple[bool, bool]:  # noqa: ARG002
            self.policy_calls += 1
            return (False, False)

    router = _make_router(response=AgentResponse(content="ok", model="test-model"))
    memory: Any = _NoRecallMemory(recalled="- should_not_be_used")
    cfg = WhaleclawConfig()
    cfg.agent.memory.organizer_background = False

    result = await run_agent(
        message="你好",
        session_id="test-memory-no-recall",
        config=cfg,
        router=router,
        memory_manager=memory,
    )

    assert result == "ok"
    assert memory.policy_calls == 1
    assert memory.recall_calls == 0


@pytest.mark.asyncio
async def test_run_agent_creation_task_auto_injects_profile_memory() -> None:
    class _NoRecallMemory(_DummyMemoryManager):
        def recall_policy(self, query: str) -> tuple[bool, bool]:  # noqa: ARG002
            self.policy_calls += 1
            return (False, False)

    router = _make_router(response=AgentResponse(content="已开始制作", model="test-model"))
    memory: Any = _NoRecallMemory(recalled="- raw_should_not_be_used")
    cfg = WhaleclawConfig()
    cfg.agent.memory.organizer_background = False

    result = await run_agent(
        message="帮我做一份香港两日游PPT",
        session_id="test-memory-creation-auto-l0",
        config=cfg,
        router=router,
        memory_manager=memory,
    )

    assert result == "已开始制作"
    assert memory.policy_calls == 1
    assert memory.recall_calls == 1


@pytest.mark.asyncio
async def test_run_agent_injects_global_style_directive() -> None:
    captured_messages: list[Any] = []

    class _StyleMemory(_DummyMemoryManager):
        async def get_global_style_directive(self) -> str:
            self.style_calls += 1
            return "回答风格：简洁明了，先结论后细节。"

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        captured_messages[:] = messages
        return AgentResponse(content="ok", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    memory: Any = _StyleMemory()
    cfg = WhaleclawConfig()
    cfg.agent.memory.organizer_background = False

    _ = await run_agent(
        message="你好",
        session_id="test-memory-style-inject",
        config=cfg,
        router=router,
        memory_manager=memory,
    )
    assert memory.style_calls == 1
    assert any(
        m.role == "system" and "全局回复风格偏好" in m.content
        for m in captured_messages
    )


@pytest.mark.asyncio
async def test_run_agent_injects_external_memory_hint() -> None:
    captured_messages: list[Any] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        captured_messages[:] = messages
        return AgentResponse(content="ok", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    memory: Any = _DummyMemoryManager()
    cfg = WhaleclawConfig()
    cfg.agent.memory.organizer_background = False

    _ = await run_agent(
        message="帮我优化这个脚本",
        session_id="test-external-memory",
        config=cfg,
        router=router,
        memory_manager=memory,
        extra_memory="【EvoMap 协作经验候选】\n- 遇到超时优先增加重试和退避",
    )

    assert any(
        m.role == "system" and "协作网络的外部经验候选" in m.content
        for m in captured_messages
    )


@pytest.mark.asyncio
async def test_run_agent_truncates_external_memory_when_compressor_unavailable() -> None:
    captured_messages: list[Any] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        captured_messages[:] = messages
        return AgentResponse(content="ok", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    router.resolve = MagicMock(side_effect=RuntimeError("compress model missing"))
    cfg = WhaleclawConfig()
    cfg.agent.memory.organizer_background = False

    huge = "X" * 12000
    _ = await run_agent(
        message="测试外部经验注入",
        session_id="test-external-memory-truncate",
        config=cfg,
        router=router,
        extra_memory=huge,
    )

    ext_msg = next(
        m for m in captured_messages
        if m.role == "system" and "协作网络的外部经验候选" in m.content
    )
    assert ext_msg.content.count("X") <= 3000


@pytest.mark.asyncio
async def test_run_agent_keeps_short_external_memory_without_compress() -> None:
    captured_messages: list[Any] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        if messages and messages[0].role == "system" and "外部经验压缩器" in messages[0].content:
            return AgentResponse(content="压缩后经验", model="compress-model")
        captured_messages[:] = messages
        return AgentResponse(content="ok", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    cfg = WhaleclawConfig()
    cfg.agent.summarizer.enabled = False

    _ = await run_agent(
        message="测试短经验压缩",
        session_id="test-external-memory-short-compress",
        config=cfg,
        router=router,
        extra_memory="【EvoMap 协作经验候选】\n- 原始经验文本",
    )

    ext_msg = next(
        m for m in captured_messages
        if m.role == "system" and "协作网络的外部经验候选" in m.content
    )
    assert "压缩后经验" not in ext_msg.content
    assert "原始经验文本" in ext_msg.content


@pytest.mark.asyncio
async def test_run_agent_skill_lock_requires_explicit_done_confirmation() -> None:
    call_count = 0

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],  # noqa: ARG001
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AgentResponse(
                content="",
                model="test-model",
                tool_calls=[ToolCall(id="tc_bash", name="bash", arguments={"command": "echo ok"})],
            )
        return AgentResponse(content="已出图", model="test-model")

    registry = ToolRegistry()
    registry.register(_BashProbeTool())
    router = _make_router(chat_fn=fake_chat)
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-1",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={},
    )

    first = await run_agent(
        message="/use nano-banana-image-t8 一只熊猫在上海街头跳舞",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )
    assert "已出图" in first
    assert "任务完成" in first
    assert session.metadata.get("locked_skill_ids") == ["nano-banana-image-t8"]
    assert session.metadata.get("skill_lock_waiting_done") is True
    assert call_count == 2

    router2 = _make_router(response=AgentResponse(content="不应调用", model="test-model"))
    second = await run_agent(
        message="任务完成",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router2,
        registry=registry,
        session=session,
    )
    assert second == "已确认任务完成，已解除本轮技能锁定。"
    assert "locked_skill_ids" not in session.metadata
    assert "skill_lock_waiting_done" not in session.metadata
    router2.chat.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_applies_locked_skill_set_to_system_prompt() -> None:
    seen_messages: list[Any] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        seen_messages.extend(messages)
        return AgentResponse(content="继续处理", model="test-model")

    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-2",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={"locked_skill_ids": ["skill-a", "skill-b"]},
    )
    router = _make_router(chat_fn=fake_chat)
    await run_agent(
        message="继续改一下",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    joined = "\n".join(
        str(m.content) for m in seen_messages if getattr(m, "role", "") == "system"
    )
    assert "当前会话已锁定技能：skill-a, skill-b" in joined


@pytest.mark.asyncio
async def test_run_agent_auto_locks_when_user_explicitly_mentions_skill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana", "文生图"],
        instructions="x",
        lock_session=True,
        source_path=Path("/tmp/SKILL.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    router = _make_router(response=AgentResponse(content="收到", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-3",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={},
    )
    result = await run_agent(
        message="使用nanobanana的技能，文生图",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "nano-banana-image-t8" in result
    assert "技能" in result
    assert "收到" in result
    assert session.metadata.get("locked_skill_ids") == ["nano-banana-image-t8"]
    assert session.metadata.get("skill_lock_waiting_done") is False


@pytest.mark.asyncio
async def test_run_agent_auto_locks_even_for_one_shot_skill_in_task_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="ppt-generator",
        name="PPT Generator",
        triggers=["ppt"],
        instructions="x",
        lock_session=False,
        source_path=Path("/tmp/SKILL2.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    router = _make_router(response=AgentResponse(content="收到", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-4",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={},
    )

    result = await run_agent(
        message="使用ppt-generator技能，帮我制作个PPT",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "ppt-generator" in result
    assert "技能" in result
    assert "收到" in result
    assert session.metadata.get("locked_skill_ids") == ["ppt-generator"]


@pytest.mark.asyncio
async def test_run_agent_rejects_skill_switch_without_user_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_a = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        source_path=Path("/tmp/a.md"),
    )
    skill_b = Skill(
        id="ppt-generator",
        name="PPT Generator",
        triggers=["ppt"],
        instructions="x",
        lock_session=False,
        source_path=Path("/tmp/b.md"),
    )

    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill_b] if "ppt" in user_message.lower() else [skill_a],  # noqa: ARG005,E501
    )

    router = _make_router(response=AgentResponse(content="收到", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-5",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={"locked_skill_ids": ["nano-banana-image-t8"]},
    )

    result = await run_agent(
        message="我在想是不是该用ppt-generator技能做个PPT",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "同意切换技能" in result
    assert session.metadata.get("locked_skill_ids") == ["nano-banana-image-t8"]


@pytest.mark.asyncio
async def test_run_agent_allows_skill_switch_with_user_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_a = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        source_path=Path("/tmp/a2.md"),
    )
    skill_b = Skill(
        id="ppt-generator",
        name="PPT Generator",
        triggers=["ppt"],
        instructions="x",
        lock_session=False,
        source_path=Path("/tmp/b2.md"),
    )

    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill_b] if "ppt" in user_message.lower() else [skill_a],  # noqa: ARG005,E501
    )

    router = _make_router(response=AgentResponse(content="收到", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-6",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={"locked_skill_ids": ["nano-banana-image-t8"]},
    )

    result = await run_agent(
        message="同意切换技能，改用ppt-generator技能做个PPT",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "切换" in result
    assert "ppt-generator" in result
    assert "收到" in result
    assert session.metadata.get("locked_skill_ids") == ["ppt-generator"]


@pytest.mark.asyncio
async def test_run_agent_allows_skill_switch_by_direct_switch_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_a = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        source_path=Path("/tmp/a3.md"),
    )
    skill_b = Skill(
        id="ppt-generator",
        name="PPT Generator",
        triggers=["ppt"],
        instructions="x",
        lock_session=False,
        source_path=Path("/tmp/b3.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill_b] if "ppt" in user_message.lower() else [skill_a],  # noqa: ARG005,E501
    )

    router = _make_router(response=AgentResponse(content="收到", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-7",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={"locked_skill_ids": ["nano-banana-image-t8"]},
    )

    result = await run_agent(
        message="换成ppt-generator技能做个PPT",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "切换" in result
    assert "ppt-generator" in result
    assert "收到" in result
    assert session.metadata.get("locked_skill_ids") == ["ppt-generator"]


@pytest.mark.asyncio
async def test_nano_banana_guard_lists_missing_params_before_execution() -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(
                    key="api_key",
                    label="API Key",
                    type="api_key",
                    required=True,
                    prompt="请提供 API Key",
                ),
                SkillParamItem(
                    key="prompt",
                    label="提示词",
                    type="text",
                    required=True,
                    aliases=["提示词", "prompt"],
                    prompt="请提供提示词",
                ),
                SkillParamItem(
                    key="ratio",
                    label="尺寸/比例",
                    type="ratio",
                    required=False,
                    aliases=["比例", "尺寸", "size"],
                    prompt="可选填写比例或尺寸",
                ),
            ],
        ),
        source_path=Path("/tmp/nano_guard.md"),
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )
    router = _make_router(response=AgentResponse(content="不应调用", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-nano-guard-1",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={"locked_skill_ids": ["nano-banana-image-t8"]},
    )

    result = await run_agent(
        message="使用nano banana制作文生图",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "API Key" in result
    assert "提示词" in result
    assert "尺寸/比例" in result
    router.chat.assert_not_called()
    monkeypatch.undo()
