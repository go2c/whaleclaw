"""Health check definitions for WhaleClaw doctor."""

from __future__ import annotations

import shutil
import socket
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from whaleclaw.config.paths import CONFIG_FILE, SESSIONS_DIR

STATUS_OK: Literal["ok"] = "ok"
STATUS_WARNING: Literal["warning"] = "warning"
STATUS_ERROR: Literal["error"] = "error"


class CheckResult(BaseModel):
    """Result of a single health check."""

    name: str
    status: Literal["ok", "warning", "error"]
    message: str
    details: str | None = None
    fix_hint: str | None = None


class HealthCheck(ABC):
    """Abstract base class for health checks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name of this check."""
        ...

    @abstractmethod
    async def check(self) -> CheckResult:
        """Run the check and return result."""
        ...


class PythonVersionCheck(HealthCheck):
    """Verify Python version >= 3.12."""

    @property
    def name(self) -> str:
        return "Python 版本"

    async def check(self) -> CheckResult:
        vi = sys.version_info
        if vi >= (3, 12):  # noqa: UP036
            ver = f"{vi.major}.{vi.minor}.{vi.micro}"
            return CheckResult(name=self.name, status="ok", message=f"Python {ver}")
        return CheckResult(
            name=self.name,
            status="error",
            message="Python 版本过低",
            details=f"当前: {sys.version_info.major}.{sys.version_info.minor}，需要 3.12+",
            fix_hint="请安装 Python 3.12 或更高版本",
        )


class ConfigFileCheck(HealthCheck):
    """Verify config file exists."""

    @property
    def name(self) -> str:
        return "配置文件"

    async def check(self) -> CheckResult:
        if CONFIG_FILE.exists():
            return CheckResult(
                name=self.name,
                status="ok",
                message=str(CONFIG_FILE),
            )
        return CheckResult(
            name=self.name,
            status="warning",
            message="配置文件不存在",
            details=f"预期路径: {CONFIG_FILE}",
            fix_hint="运行 whaleclaw onboard 或手动创建配置文件",
        )


class PortCheck(HealthCheck):
    """Verify gateway port is available."""

    def __init__(self, port: int = 18666) -> None:
        self._port = port

    @property
    def name(self) -> str:
        return "Gateway 端口"

    async def check(self) -> CheckResult:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", self._port))
            return CheckResult(
                name=self.name,
                status="ok",
                message=f"端口 {self._port} 可用",
            )
        except OSError as e:
            return CheckResult(
                name=self.name,
                status="error",
                message=f"端口 {self._port} 不可用",
                details=str(e),
                fix_hint="更换端口或关闭占用该端口的进程",
            )


class DatabaseCheck(HealthCheck):
    """Verify sessions directory exists."""

    @property
    def name(self) -> str:
        return "会话数据库"

    async def check(self) -> CheckResult:
        if SESSIONS_DIR.exists() and SESSIONS_DIR.is_dir():
            return CheckResult(
                name=self.name,
                status="ok",
                message="会话目录正常",
            )
        return CheckResult(
            name=self.name,
            status="warning",
            message="会话目录不存在",
            details=str(SESSIONS_DIR),
            fix_hint="运行 whaleclaw gateway run 将自动创建",
        )


class DiskSpaceCheck(HealthCheck):
    """Verify disk free space > 100MB."""

    MIN_FREE_MB = 100

    @property
    def name(self) -> str:
        return "磁盘空间"

    async def check(self) -> CheckResult:
        home = Path.home()
        usage = shutil.disk_usage(home)
        free_mb = usage.free // (1024 * 1024)
        if free_mb >= self.MIN_FREE_MB:
            return CheckResult(
                name=self.name,
                status="ok",
                message=f"充足 ({free_mb} MB 可用)",
            )
        return CheckResult(
            name=self.name,
            status="error",
            message="磁盘空间不足",
            details=f"仅剩 {free_mb} MB，需要至少 {self.MIN_FREE_MB} MB",
            fix_hint="请清理磁盘空间",
        )
