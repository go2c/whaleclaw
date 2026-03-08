"""Background process session registry for bash/process tools."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

_MAX_AGGREGATED_CHARS = 200_000
_MAX_TAIL_CHARS = 12_000


@dataclass
class ProcessSession:
    """A background bash process session."""

    id: str
    command: str
    cwd: str
    process: asyncio.subprocess.Process
    started_at: float
    aggregated: str = ""
    last_poll_pos: int = 0
    exited: bool = False
    exit_code: int | None = None
    tail: str = ""
    stdout_task: asyncio.Task[None] | None = None
    stderr_task: asyncio.Task[None] | None = None
    wait_task: asyncio.Task[None] | None = None


_SESSIONS: dict[str, ProcessSession] = {}


def _append_text(session: ProcessSession, text: str) -> None:
    if not text:
        return
    session.aggregated = (session.aggregated + text)[-_MAX_AGGREGATED_CHARS:]
    session.tail = session.aggregated[-_MAX_TAIL_CHARS:]
    session.last_poll_pos = min(session.last_poll_pos, len(session.aggregated))


async def _consume_stream(
    session: ProcessSession,
    stream: asyncio.StreamReader | None,
    *,
    label: str,
) -> None:
    if stream is None:
        return
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return
        text = chunk.decode(errors="replace")
        prefix = "" if label == "stdout" else "[stderr]\n"
        _append_text(session, f"{prefix}{text}")


async def _watch_exit(session: ProcessSession) -> None:
    rc = await session.process.wait()
    session.exited = True
    session.exit_code = rc


def register_background_process(
    *,
    command: str,
    cwd: str,
    process: asyncio.subprocess.Process,
) -> ProcessSession:
    session = ProcessSession(
        id=f"proc_{uuid.uuid4().hex[:10]}",
        command=command,
        cwd=cwd,
        process=process,
        started_at=time.time(),
    )
    session.stdout_task = asyncio.create_task(
        _consume_stream(session, process.stdout, label="stdout")
    )
    session.stderr_task = asyncio.create_task(
        _consume_stream(session, process.stderr, label="stderr")
    )
    session.wait_task = asyncio.create_task(_watch_exit(session))
    _SESSIONS[session.id] = session
    return session


def list_sessions() -> list[ProcessSession]:
    return sorted(_SESSIONS.values(), key=lambda item: item.started_at, reverse=True)


def get_session(session_id: str) -> ProcessSession | None:
    return _SESSIONS.get(session_id)


def delete_session(session_id: str) -> bool:
    session = _SESSIONS.pop(session_id, None)
    if session is None:
        return False
    for task in (session.stdout_task, session.stderr_task, session.wait_task):
        if task and not task.done():
            task.cancel()
    return True
