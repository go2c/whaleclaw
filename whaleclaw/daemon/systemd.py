"""Linux systemd user service management for WhaleClaw Gateway."""

from __future__ import annotations

import subprocess
from pathlib import Path

UNIT_NAME = "whaleclaw-gateway"
UNIT_PATH = Path.home() / ".config/systemd/user" / f"{UNIT_NAME}.service"


class SystemdService:
    """Linux systemd user service for WhaleClaw Gateway."""

    def __init__(self, unit_path: Path | None = None) -> None:
        self._unit_path = unit_path or UNIT_PATH

    def _gen_unit(
        self,
        python_path: str,
        port: int = 18666,
        bind: str = "127.0.0.1",
    ) -> str:
        return f"""[Unit]
Description=WhaleClaw Gateway
After=network.target

[Service]
Type=simple
ExecStart={python_path} -m whaleclaw gateway run --port {port} --bind {bind}
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""

    def install(
        self,
        python_path: str,
        port: int = 18666,
        bind: str = "127.0.0.1",
    ) -> None:
        self._unit_path.parent.mkdir(parents=True, exist_ok=True)
        content = self._gen_unit(python_path, port, bind)
        self._unit_path.write_text(content, encoding="utf-8")
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True,
            check=True,
        )

    def uninstall(self) -> None:
        subprocess.run(
            ["systemctl", "--user", "stop", UNIT_NAME],
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ["systemctl", "--user", "disable", UNIT_NAME],
            capture_output=True,
            check=False,
        )
        if self._unit_path.exists():
            self._unit_path.unlink()
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True,
            check=True,
        )

    def start(self) -> None:
        subprocess.run(
            ["systemctl", "--user", "start", UNIT_NAME],
            capture_output=True,
            check=True,
        )

    def stop(self) -> None:
        subprocess.run(
            ["systemctl", "--user", "stop", UNIT_NAME],
            capture_output=True,
            check=True,
        )

    def is_installed(self) -> bool:
        return self._unit_path.exists()

    def status(self) -> str:
        return "installed" if self.is_installed() else "not_installed"
