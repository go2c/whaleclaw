"""Path constants for WhaleClaw data directories."""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_home() -> Path:
    raw = os.environ.get("WHALECLAW_HOME", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.home() / ".whaleclaw"


WHALECLAW_HOME = _resolve_home()
CONFIG_FILE = WHALECLAW_HOME / "whaleclaw.json"
CREDENTIALS_DIR = WHALECLAW_HOME / "credentials"
SESSIONS_DIR = WHALECLAW_HOME / "sessions"
MEMORY_DIR = WHALECLAW_HOME / "memory"
WORKSPACE_DIR = WHALECLAW_HOME / "workspace"
PLUGINS_DIR = WHALECLAW_HOME / "plugins"
LOGS_DIR = WHALECLAW_HOME / "logs"
EVOMAP_DIR = WHALECLAW_HOME / "evomap"

_ALL_DIRS = [
    WHALECLAW_HOME,
    CREDENTIALS_DIR,
    SESSIONS_DIR,
    MEMORY_DIR,
    WORKSPACE_DIR,
    PLUGINS_DIR,
    LOGS_DIR,
    EVOMAP_DIR,
]


def ensure_dirs() -> None:
    """Create all required directories if they don't exist."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
