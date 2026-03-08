"""Tests for process tool with background bash sessions."""

from __future__ import annotations

import json

import pytest

from whaleclaw.tools.bash import BashTool
from whaleclaw.tools.process import ProcessTool
from whaleclaw.tools.process_registry import list_sessions


@pytest.fixture()
def process_tool() -> ProcessTool:
    return ProcessTool()


@pytest.fixture()
def bash_tool() -> BashTool:
    return BashTool()


def _extract_session_id(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("session_id: "):
            return line.split(": ", 1)[1].strip()
    raise AssertionError(f"session_id missing in output: {output}")


@pytest.mark.asyncio
async def test_process_list_and_poll_background_session(
    bash_tool: BashTool,
    process_tool: ProcessTool,
) -> None:
    started = await bash_tool.execute(
        command="printf 'hello'; sleep 0.2; printf ' world'",
        background=True,
    )
    assert started.success
    session_id = _extract_session_id(started.output)

    listed = await process_tool.execute(action="list")
    payload = json.loads(listed.output)
    assert any(item["session_id"] == session_id for item in payload)

    polled = await process_tool.execute(action="poll", session_id=session_id, timeout_ms=1000)
    assert polled.success
    assert "Process" in polled.output


@pytest.mark.asyncio
async def test_process_write_and_kill_session(
    bash_tool: BashTool,
    process_tool: ProcessTool,
) -> None:
    started = await bash_tool.execute(
        command="python -c \"import sys,time; data=sys.stdin.read(); print(data); time.sleep(30)\"",
        background=True,
    )
    session_id = _extract_session_id(started.output)

    wrote = await process_tool.execute(action="write", session_id=session_id, data="abc\n")
    assert wrote.success

    killed = await process_tool.execute(action="kill", session_id=session_id)
    assert killed.success


@pytest.mark.asyncio
async def test_process_log_and_clear_finished_session(
    bash_tool: BashTool,
    process_tool: ProcessTool,
) -> None:
    started = await bash_tool.execute(command="printf 'line1\\nline2\\n'", background=True)
    session_id = _extract_session_id(started.output)

    await process_tool.execute(action="poll", session_id=session_id, timeout_ms=500)
    logged = await process_tool.execute(action="log", session_id=session_id, lines=2)
    assert logged.success
    assert "line1" in logged.output

    cleared = await process_tool.execute(action="clear", session_id=session_id)
    assert cleared.success
    assert all(session.id != session_id for session in list_sessions())
