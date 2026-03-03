"""Asset fetching and local cache from EvoMap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from whaleclaw.config.paths import EVOMAP_DIR
from whaleclaw.plugins.evomap.client import A2AClient


class AssetFetcher:
    """Fetch promoted assets from EvoMap and cache locally."""

    CACHE_DIR = EVOMAP_DIR / "assets"

    def __init__(self, client: A2AClient, cache_dir: Path | None = None) -> None:
        self._client = client
        self.CACHE_DIR = cache_dir or self.CACHE_DIR
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, asset_id: str) -> Path:
        safe_id = asset_id.replace(":", "_").replace("/", "_")
        return self.CACHE_DIR / f"{safe_id}.json"

    async def fetch_promoted(self, asset_type: str = "Capsule") -> list[dict[str, Any]]:
        """Fetch latest promoted assets."""
        resp = await self._client.fetch(asset_type=asset_type, include_tasks=False)
        assets = resp.get("payload", {}).get("assets", resp.get("assets", []))
        for asset in assets:
            aid = asset.get("asset_id") or asset.get("assetId")
            if aid:
                self._cache_path(aid).write_text(
                    json.dumps(asset, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        return assets

    async def search_by_signals(self, signals: list[str]) -> list[dict[str, Any]]:
        """Search assets by signals (delegates to fetch with signal filter)."""
        resp = await self._client.fetch(
            asset_type="Capsule",
            include_tasks=False,
        )
        assets = resp.get("payload", {}).get("assets", resp.get("assets", []))
        for asset in assets:
            aid = asset.get("asset_id") or asset.get("assetId")
            if aid:
                self._cache_path(str(aid)).write_text(
                    json.dumps(asset, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        clean = [s.strip().lower() for s in signals if s.strip()]
        result: list[tuple[int, dict[str, Any]]] = []
        for asset in assets:
            score = self._score_asset_match(asset, clean)
            if score > 0:
                result.append((score, asset))
        result.sort(key=lambda x: x[0], reverse=True)
        return [asset for _score, asset in result]

    def search_cached_by_signals(
        self,
        signals: list[str],
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Search local cached assets by signal/summary text without network I/O."""
        clean = [s.strip().lower() for s in signals if s.strip()]
        if not clean:
            return []

        scored: list[tuple[int, dict[str, Any]]] = []
        for path in sorted(self.CACHE_DIR.glob("*.json"), reverse=True):
            try:
                asset = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(asset, dict):
                continue

            score = self._score_asset_match(asset, clean)
            if score > 0:
                scored.append((score, asset))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [asset for _score, asset in scored[: max(1, limit)]]

    def list_cached_recent(self, *, limit: int = 5) -> list[dict[str, Any]]:
        """Return recent cached assets without signal filtering."""
        results: list[dict[str, Any]] = []
        for path in sorted(self.CACHE_DIR.glob("*.json"), reverse=True):
            try:
                asset = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(asset, dict):
                results.append(asset)
            if len(results) >= max(1, limit):
                break
        return results

    def get_cached(self, asset_id: str) -> dict[str, Any] | None:
        """Load asset from local cache."""
        path = self._cache_path(asset_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _score_asset_match(asset: dict[str, Any], signals: list[str]) -> int:
        if not signals:
            return 0
        trigger = [str(t).strip().lower() for t in asset.get("trigger", []) if str(t).strip()]
        signals_match = [
            str(t).strip().lower() for t in asset.get("signals_match", []) if str(t).strip()
        ]
        hay = " ".join(str(asset.get(k, "")) for k in ("title", "summary", "description")).lower()
        score = 0
        for sig in signals:
            if sig in trigger or sig in signals_match:
                score += 3
                continue
            if any(sig in t or t in sig for t in trigger):
                score += 2
                continue
            if any(sig in t or t in sig for t in signals_match):
                score += 2
                continue
            if sig in hay:
                score += 1
        return score
