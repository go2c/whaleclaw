"""macOS launchd service management for WhaleClaw Gateway."""

from __future__ import annotations

import subprocess
from pathlib import Path

from whaleclaw.config.paths import LOGS_DIR

PLIST_LABEL = "ai.whaleclaw.gateway"
PLIST_PATH = Path.home() / "Library/LaunchAgents" / f"{PLIST_LABEL}.plist"


class LaunchdService:
    """macOS launchd service for WhaleClaw Gateway."""

    def __init__(self, plist_path: Path | None = None) -> None:
        self._plist_path = plist_path or PLIST_PATH

    def _gen_plist(
        self,
        python_path: str,
        port: int = 18666,
        bind: str = "127.0.0.1",
    ) -> str:
        log_dir = Path.home() / ".whaleclaw/logs"
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>whaleclaw</string>
        <string>gateway</string>
        <string>run</string>
        <string>--port</string>
        <string>{port}</string>
        <string>--bind</string>
        <string>{bind}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir}/gateway.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/gateway.err</string>
</dict>
</plist>
"""

    def install(
        self,
        python_path: str,
        port: int = 18666,
        bind: str = "127.0.0.1",
    ) -> None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        content = self._gen_plist(python_path, port, bind)
        self._plist_path.write_text(content, encoding="utf-8")

    def uninstall(self) -> None:
        subprocess.run(
            ["launchctl", "unload", str(self._plist_path)],
            capture_output=True,
            check=False,
        )
        if self._plist_path.exists():
            self._plist_path.unlink()

    def start(self) -> None:
        subprocess.run(
            ["launchctl", "load", str(self._plist_path)],
            capture_output=True,
            check=True,
        )

    def stop(self) -> None:
        subprocess.run(
            ["launchctl", "unload", str(self._plist_path)],
            capture_output=True,
            check=True,
        )

    def is_installed(self) -> bool:
        return self._plist_path.exists()

    def status(self) -> str:
        return "installed" if self.is_installed() else "not_installed"
