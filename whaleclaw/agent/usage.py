"""Usage tracking — token counts, cost estimation, and persistence."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import aiosqlite
from pydantic import BaseModel

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    "gpt-5.2": (2.0, 8.0),
    "gpt-5.2-codex": (3.0, 15.0),
    "gpt-5.3-codex": (3.0, 15.0),
    "deepseek-chat": (0.14, 0.28),
    "qwen3.5-plus": (0.11, 0.28),
    "qwen3-max": (0.35, 1.39),
    "qwen-max": (2.4, 9.6),
    "glm-5": (2.0, 8.0),
    "glm-4.7": (1.0, 4.0),
    "glm-4.7-flash": (0.0, 0.0),
    "MiniMax-M2.5": (1.0, 5.0),
    "MiniMax-M2.1": (1.0, 5.0),
    "kimi-k2.5": (0.6, 3.0),
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-3-flash-preview": (0.1, 0.4),
    "meta/llama-3.1-8b-instruct": (0.0, 0.0),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    session_id TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    thinking_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_usage_session ON usage_records(session_id);
CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_records(timestamp);
"""


class TokenUsage(BaseModel):
    """Single request token usage."""

    model: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int = 0
    cost_usd: float = 0.0


class SessionUsage(BaseModel):
    """Aggregated usage for a session."""

    session_id: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_thinking_tokens: int = 0
    total_cost_usd: float = 0.0
    request_count: int = 0


class DailyUsage(BaseModel):
    """Aggregated usage for a date."""

    date: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    request_count: int = 0


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD from token counts. Returns 0.0 for unknown models."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        for key in MODEL_PRICING:
            if key in model or model in key:
                pricing = MODEL_PRICING[key]
                break
    if pricing is None:
        return 0.0
    input_price, output_price = pricing
    return (input_tokens / 1_000_000 * input_price) + (output_tokens / 1_000_000 * output_price)


class UsageTracker:
    """Async SQLite-backed usage tracker."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def open(self) -> None:
        """Open database and ensure schema."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._path))
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("UsageTracker not opened")
        return self._conn

    async def record(self, session_id: str, usage: TokenUsage) -> None:
        """Insert a usage record."""
        from datetime import datetime

        ts = datetime.now().isoformat()
        cost = (
            usage.cost_usd
            if usage.cost_usd > 0
            else estimate_cost(usage.model, usage.input_tokens, usage.output_tokens)
        )
        await self._db().execute(
            """INSERT INTO usage_records
            (timestamp, session_id, model, input_tokens, output_tokens,
             thinking_tokens, cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                ts,
                session_id,
                usage.model,
                usage.input_tokens,
                usage.output_tokens,
                usage.thinking_tokens,
                cost,
            ),
        )
        await self._db().commit()

    async def get_session_usage(self, session_id: str) -> SessionUsage:
        """Aggregate usage for a session."""
        cursor = await self._db().execute(
            """SELECT
               COALESCE(SUM(input_tokens), 0),
               COALESCE(SUM(output_tokens), 0),
               COALESCE(SUM(thinking_tokens), 0),
               COALESCE(SUM(cost_usd), 0),
               COUNT(*) FROM usage_records WHERE session_id = ?""",
            (session_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return SessionUsage(session_id=session_id)
        return SessionUsage(
            session_id=session_id,
            total_input_tokens=row[0] or 0,
            total_output_tokens=row[1] or 0,
            total_thinking_tokens=row[2] or 0,
            total_cost_usd=row[3] or 0.0,
            request_count=row[4] or 0,
        )

    async def get_daily_usage(self, date_str: str | None = None) -> DailyUsage:
        """Aggregate usage by date. Defaults to today."""
        d = date_str or date.today().isoformat()
        cursor = await self._db().execute(
            """SELECT
               COALESCE(SUM(input_tokens), 0),
               COALESCE(SUM(output_tokens), 0),
               COALESCE(SUM(cost_usd), 0),
               COUNT(*) FROM usage_records
               WHERE date(timestamp) = date(?)""",
            (d,),
        )
        row = await cursor.fetchone()
        if row is None:
            return DailyUsage(date=d)
        return DailyUsage(
            date=d,
            total_input_tokens=row[0] or 0,
            total_output_tokens=row[1] or 0,
            total_cost_usd=row[2] or 0.0,
            request_count=row[3] or 0,
        )

    async def get_total_usage(self) -> dict[str, int | float]:
        """Return aggregate totals across all records."""
        cursor = await self._db().execute(
            """SELECT
               COALESCE(SUM(input_tokens), 0),
               COALESCE(SUM(output_tokens), 0),
               COALESCE(SUM(thinking_tokens), 0),
               COALESCE(SUM(cost_usd), 0),
               COUNT(*) FROM usage_records"""
        )
        row = await cursor.fetchone()
        if row is None:
            return {
                "total_input": 0,
                "total_output": 0,
                "total_thinking": 0,
                "total_cost": 0.0,
                "request_count": 0,
            }
        return {
            "total_input": row[0] or 0,
            "total_output": row[1] or 0,
            "total_thinking": row[2] or 0,
            "total_cost": row[3] or 0.0,
            "request_count": row[4] or 0,
        }
