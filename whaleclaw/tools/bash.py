"""Bash command execution tool."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult

_DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+-rf\s+/\s*$"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+if=/dev/zero\b"),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;"),
]

_MAX_OUTPUT = 50_000

_PROJECT_PYTHON_BIN = Path(__file__).resolve().parents[2] / "python" / "bin"


def _strip_control_chars(text: str) -> str:
    """Remove ASCII control characters except LF/TAB/CR."""
    return "".join(
        ch for ch in text
        if ch in ("\n", "\t", "\r") or (ord(ch) >= 32 and ord(ch) != 127)
    )


class BashTool(Tool):
    """Execute a bash command and return stdout/stderr/exit_code."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bash",
            description="Execute a bash command. Returns stdout, stderr, and exit code.",
            parameters=[
                ToolParameter(
                    name="command", type="string", description="The bash command to execute."
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Timeout in seconds (default 30, max 300).",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        raw_command: str = kwargs.get("command", "")
        command = _strip_control_chars(raw_command)
        timeout: int = int(kwargs.get("timeout", 30))

        if not command.strip():
            return ToolResult(success=False, output="", error="命令为空")

        for pattern in _DANGEROUS_PATTERNS:
            if pattern.search(command):
                return ToolResult(success=False, output="", error=f"危险命令被拦截: {command}")

        env = os.environ.copy()
        if _PROJECT_PYTHON_BIN.is_dir():
            env["PATH"] = f"{_PROJECT_PYTHON_BIN}:{env.get('PATH', '')}"

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            return ToolResult(success=False, output="", error=f"命令超时 ({timeout}s)")
        except OSError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        out = stdout.decode(errors="replace")[:_MAX_OUTPUT]
        err = stderr.decode(errors="replace")[:_MAX_OUTPUT]
        exit_code = proc.returncode or 0

        output = out
        if err:
            output += f"\n[stderr]\n{err}"
        output += f"\n[exit_code: {exit_code}]"

        return ToolResult(
            success=exit_code == 0,
            output=output.strip(),
            error=err if exit_code != 0 else None,
        )
