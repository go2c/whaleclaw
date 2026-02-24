"""Daemon manager — platform detection and service delegation."""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from whaleclaw.daemon.launchd import LaunchdService
    from whaleclaw.daemon.systemd import SystemdService


class ServiceStatus(BaseModel):
    """Daemon service status model."""

    installed: bool
    running: bool = False
    platform: str


class DaemonManager:
    """Daemon manager — delegates to launchd (macOS) or systemd (Linux)."""

    def __init__(self) -> None:
        self._platform = platform.system().lower()
        if self._platform == "darwin":
            from whaleclaw.daemon.launchd import LaunchdService

            self._service: LaunchdService | SystemdService = LaunchdService()
        elif self._platform == "linux":
            from whaleclaw.daemon.systemd import SystemdService

            self._service = SystemdService()
        else:
            raise RuntimeError(f"不支持的平台: {self._platform}")

    def install(
        self,
        python_path: str,
        port: int = 18666,
        bind: str = "127.0.0.1",
    ) -> None:
        self._service.install(python_path=python_path, port=port, bind=bind)

    def uninstall(self) -> None:
        self._service.uninstall()

    def start(self) -> None:
        self._service.start()

    def stop(self) -> None:
        self._service.stop()

    def status(self) -> ServiceStatus:
        installed = self._service.is_installed()
        running = False
        return ServiceStatus(
            installed=installed,
            running=running,
            platform=self._platform,
        )
