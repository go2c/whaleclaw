"""Pydantic v2 configuration schema for WhaleClaw."""

from typing import Literal

from pydantic import BaseModel, Field

from whaleclaw.config.paths import WORKSPACE_DIR


class AuthConfig(BaseModel):
    """Authentication configuration for the Gateway."""

    mode: str = "none"
    password: str | None = None
    token: str | None = None
    jwt_secret: str = "whaleclaw-default-secret"
    jwt_expire_hours: int = 24


class GatewayConfig(BaseModel):
    """Gateway server configuration."""

    port: int = Field(default=18666, ge=1, le=65535)
    bind: str = "127.0.0.1"
    verbose: bool = False
    auth: AuthConfig = Field(default_factory=AuthConfig)


class ProviderModelEntry(BaseModel):
    """A single validated model under a provider (e.g. one NVIDIA NIM model)."""

    id: str
    name: str = ""
    base_url: str | None = None
    verified: bool = False
    thinking: str = "off"


class ProviderConfig(BaseModel):
    """Per-provider configuration."""

    api_key: str | None = None
    base_url: str | None = None
    timeout: int = 120
    configured_models: list[ProviderModelEntry] = Field(default_factory=list)
    auth_mode: Literal["api_key", "oauth"] = "api_key"
    oauth_access: str | None = None
    oauth_refresh: str | None = None
    oauth_expires: int = 0
    oauth_account_id: str | None = None


class ModelsConfig(BaseModel):
    """Configuration for all LLM providers."""

    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    qwen: ProviderConfig = Field(default_factory=ProviderConfig)
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig)
    minimax: ProviderConfig = Field(default_factory=ProviderConfig)
    moonshot: ProviderConfig = Field(default_factory=ProviderConfig)
    google: ProviderConfig = Field(default_factory=ProviderConfig)
    nvidia: ProviderConfig = Field(default_factory=ProviderConfig)


class SummarizerConfig(BaseModel):
    """Configuration for L0/L1 hierarchical context compression.

    Uses a cheap, fast model to generate layered summaries of older
    conversation history, persisted in SQLite for on-demand loading.
    Set ``enabled = false`` or leave the model provider unconfigured
    to disable; context will simply be truncated when it overflows.
    """

    model: str = "zhipu/glm-4.7-flash"
    enabled: bool = True


class AgentConfig(BaseModel):
    """Agent runtime configuration."""

    model: str = "anthropic/claude-sonnet-4-20250514"
    max_tool_rounds: int = 25
    workspace: str = str(WORKSPACE_DIR)
    thinking_level: str = "off"
    summarizer: SummarizerConfig = Field(default_factory=SummarizerConfig)


class FeishuGroupConfig(BaseModel):
    """Per-group settings for Feishu."""

    require_mention: bool = True
    activation: Literal["mention", "always"] = "mention"


class FeishuChannelConfig(BaseModel):
    """Feishu bot channel configuration."""

    mode: Literal["ws", "webhook"] = "ws"
    app_id: str = ""
    app_secret: str = ""
    verification_token: str | None = None
    encrypt_key: str | None = None
    allow_from: list[str] = Field(default_factory=list)
    groups: dict[str, FeishuGroupConfig] = Field(default_factory=dict)
    dm_policy: Literal["pairing", "open", "closed"] = "pairing"
    webhook_path: str = "/webhook/feishu"


class ChannelsConfig(BaseModel):
    """Configuration for all message channels."""

    feishu: FeishuChannelConfig = Field(default_factory=FeishuChannelConfig)


class SecurityConfig(BaseModel):
    """Security configuration."""

    sandbox_mode: str = "non-main"
    dm_policy: str = "pairing"
    audit: bool = True


class RoutingRuleConfig(BaseModel):
    """Single routing rule in config."""

    name: str
    priority: int = 0
    match: dict[str, object] = Field(default_factory=dict)
    target: dict[str, object] = Field(default_factory=dict)


class RoutingConfig(BaseModel):
    """Routing configuration."""

    rules: list[RoutingRuleConfig] = Field(default_factory=list)


class WhaleclawConfig(BaseModel):
    """Root configuration model."""

    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
