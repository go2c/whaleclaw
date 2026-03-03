"""EvoMap node identity management."""

from __future__ import annotations

import json
import secrets
from pathlib import Path

from whaleclaw.config.paths import EVOMAP_DIR


class EvoMapIdentity:
    """Manage WhaleClaw node identity in EvoMap network."""

    IDENTITY_PATH = EVOMAP_DIR / "identity.json"

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or self.IDENTITY_PATH
        self._data: dict[str, str | int | None] | None = None

    def _load(self) -> dict[str, str | int | None]:
        if self._data is not None:
            return self._data
        path = self._path
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self._data = {}
        return self._data

    def _save(self, data: dict[str, str | int | None]) -> None:
        path = self._path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self._data = data

    def get_or_create_sender_id(self) -> str:
        """Generate or load sender_id. Returns 'node_' + 16 hex chars."""
        data = self._load()
        sid = data.get("sender_id")
        if sid and isinstance(sid, str):
            return sid
        sid = "node_" + secrets.token_hex(8)
        data["sender_id"] = sid
        self._save(data)
        return sid

    def get_claim_code(self) -> str | None:
        """Get current claim code for binding user account."""
        data = self._load()
        code = data.get("claim_code")
        return str(code) if code is not None else None

    def save_claim_code(self, code: str, url: str) -> None:
        """Save claim code and URL from hello response."""
        data = self._load()
        data["claim_code"] = code
        data["claim_url"] = url
        self._save(data)
