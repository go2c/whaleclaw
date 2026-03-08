"""Background process session tool aligned with bash background mode."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult
from whaleclaw.tools.process_registry import delete_session, get_session, list_sessions

_ACTIONS = ["list", "poll", "log", "write", "kill", "clear", "remove"]
_DEFAULT_LOG_LINES = 200
_MAX_LOG_LINES = 1000


def _split_tail_lines(text: str, lines: int) -> str:
    return "\n".join(text.splitlines()[-lines:])


class ProcessTool(Tool):
    """Inspect and control background bash sessions."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="process",
            description=(
                "Manage background bash sessions. Actions: "
                "list — list sessions; "
                "poll(session_id, timeout_ms?) — get new output and status; "
                "log(session_id, lines?) — read recent full log; "
                "write(session_id, data) — write to stdin; "
                "kill(session_id) — terminate session; "
                "clear(session_id) — clear finished session; "
                "remove(session_id) — remove session record."
            ),
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="Process action to perform.",
                    enum=_ACTIONS,
                ),
                ToolParameter(
                    name="session_id",
                    type="string",
                    description="Background session id for non-list actions.",
                    required=False,
                ),
                ToolParameter(
                    name="lines",
                    type="integer",
                    description="Recent log lines for log action. Default 200, max 1000.",
                    required=False,
                ),
                ToolParameter(
                    name="timeout_ms",
                    type="integer",
                    description="Wait before returning from poll, up to 120000 ms.",
                    required=False,
                ),
                ToolParameter(
                    name="data",
                    type="string",
                    description="Data written to stdin for write action.",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "")).strip().lower()
        if action not in _ACTIONS:
            return ToolResult(success=False, output="", error="不支持的 process.action")
        if action == "list":
            return await self._list_sessions()
        session_id = str(kwargs.get("session_id", "")).strip()
        if not session_id:
            return ToolResult(success=False, output="", error="session_id 不能为空")
        if action == "poll":
            return await self._poll_session(
                session_id=session_id,
                timeout_ms=int(kwargs.get("timeout_ms", 0)),
            )
        if action == "log":
            return await self._log_session(
                session_id=session_id,
                lines=int(kwargs.get("lines", _DEFAULT_LOG_LINES)),
            )
        if action == "write":
            return await self._write_session(
                session_id=session_id,
                data=str(kwargs.get("data", "")),
            )
        if action == "kill":
            return await self._kill_session(session_id=session_id)
        if action == "clear":
            return await self._clear_session(session_id=session_id)
        return await self._remove_session(session_id=session_id)

    async def _list_sessions(self) -> ToolResult:
        rows: list[dict[str, object]] = []
        now = time.time()
        for session in list_sessions():
            rows.append(
                {
                    "session_id": session.id,
                    "status": "completed" if session.exited else "running",
                    "pid": session.process.pid or 0,
                    "runtime_ms": int((now - session.started_at) * 1000),
                    "cwd": session.cwd,
                    "command": session.command,
                    "exit_code": session.exit_code if session.exited else None,
                }
            )
        return ToolResult(success=True, output=json.dumps(rows, ensure_ascii=False, indent=2))

    async def _poll_session(self, *, session_id: str, timeout_ms: int) -> ToolResult:
        session = get_session(session_id)
        if session is None:
            return ToolResult(success=False, output="", error=f"会话不存在: {session_id}")
        wait_ms = max(0, min(timeout_ms, 120_000))
        if wait_ms > 0 and not session.exited:
            deadline = time.time() + (wait_ms / 1000)
            while not session.exited and time.time() < deadline:
                await asyncio.sleep(0.25)
        chunk = session.aggregated[session.last_poll_pos :]
        session.last_poll_pos = len(session.aggregated)
        suffix = (
            f"\n\nProcess exited with code {session.exit_code or 0}."
            if session.exited
            else "\n\nProcess still running."
        )
        text = (chunk.strip() or "(no new output)") + suffix
        return ToolResult(success=True, output=text)

    async def _log_session(self, *, session_id: str, lines: int) -> ToolResult:
        session = get_session(session_id)
        if session is None:
            return ToolResult(success=False, output="", error=f"会话不存在: {session_id}")
        window = max(1, min(lines, _MAX_LOG_LINES))
        return ToolResult(success=True, output=_split_tail_lines(session.aggregated, window))

    async def _write_session(self, *, session_id: str, data: str) -> ToolResult:
        session = get_session(session_id)
        if session is None:
            return ToolResult(success=False, output="", error=f"会话不存在: {session_id}")
        stdin = session.process.stdin
        if stdin is None or stdin.is_closing():
            return ToolResult(success=False, output="", error=f"会话 stdin 不可写: {session_id}")
        stdin.write(data.encode())
        await stdin.drain()
        return ToolResult(success=True, output=f"已写入 {len(data)} 字节到会话 {session_id}")

    async def _kill_session(self, *, session_id: str) -> ToolResult:
        session = get_session(session_id)
        if session is None:
            return ToolResult(success=False, output="", error=f"会话不存在: {session_id}")
        if session.exited:
            return ToolResult(success=True, output=f"会话已结束: {session_id}")
        session.process.terminate()
        try:
            await asyncio.wait_for(session.process.wait(), timeout=2)
        except TimeoutError:
            session.process.kill()
            await session.process.wait()
        session.exited = True
        session.exit_code = session.process.returncode
        return ToolResult(success=True, output=f"已终止会话 {session_id}")

    async def _clear_session(self, *, session_id: str) -> ToolResult:
        session = get_session(session_id)
        if session is None:
            return ToolResult(success=False, output="", error=f"会话不存在: {session_id}")
        if not session.exited:
            return ToolResult(success=False, output="", error=f"会话仍在运行: {session_id}")
        delete_session(session_id)
        return ToolResult(success=True, output=f"已清理会话 {session_id}")

    async def _remove_session(self, *, session_id: str) -> ToolResult:
        session = get_session(session_id)
        if session is None:
            return ToolResult(success=False, output="", error=f"会话不存在: {session_id}")
        if not session.exited:
            await self._kill_session(session_id=session_id)
        delete_session(session_id)
        return ToolResult(success=True, output=f"已移除会话 {session_id}")
