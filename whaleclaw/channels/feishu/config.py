"""Feishu channel configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GroupConfig(BaseModel):
    """Per-group settings."""

    require_mention: bool = True
    activation: Literal["mention", "always"] = "mention"


class FeishuConfig(BaseModel):
    """Feishu bot configuration."""

    mode: Literal["ws", "webhook"] = "ws"
    app_id: str = ""
    app_secret: str = ""
    verification_token: str | None = None
    encrypt_key: str | None = None
    allow_from: list[str] = Field(default_factory=list)
    groups: dict[str, GroupConfig] = Field(default_factory=dict)
    dm_policy: Literal["pairing", "open", "closed"] = "pairing"
    webhook_path: str = "/webhook/feishu"
