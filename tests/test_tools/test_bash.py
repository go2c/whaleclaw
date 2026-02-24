"""Tests for the bash tool."""

from __future__ import annotations

import pytest

from whaleclaw.tools.bash import BashTool


@pytest.fixture()
def tool() -> BashTool:
    return BashTool()


@pytest.mark.asyncio
async def test_echo(tool: BashTool) -> None:
    result = await tool.execute(command="echo hello")
    assert result.success
    assert "hello" in result.output


@pytest.mark.asyncio
async def test_exit_code(tool: BashTool) -> None:
    result = await tool.execute(command="exit 1")
    assert not result.success
    assert "exit_code: 1" in result.output


@pytest.mark.asyncio
async def test_empty_command(tool: BashTool) -> None:
    result = await tool.execute(command="")
    assert not result.success
    assert result.error == "命令为空"


@pytest.mark.asyncio
async def test_dangerous_command(tool: BashTool) -> None:
    result = await tool.execute(command="rm -rf /")
    assert not result.success
    assert "危险命令" in (result.error or "")


@pytest.mark.asyncio
async def test_timeout(tool: BashTool) -> None:
    result = await tool.execute(command="sleep 10", timeout=1)
    assert not result.success
    assert "超时" in (result.error or "")


@pytest.mark.asyncio
async def test_control_chars_are_stripped(tool: BashTool) -> None:
    result = await tool.execute(command="\x18echo hello\x18")
    assert result.success
    assert "hello" in result.output


@pytest.mark.asyncio
async def test_only_control_chars_becomes_empty(tool: BashTool) -> None:
    result = await tool.execute(command="\x10\x18\x00")
    assert not result.success
    assert result.error == "命令为空"
