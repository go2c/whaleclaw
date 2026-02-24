"""Tests for configuration schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from whaleclaw.config.schema import AgentConfig, GatewayConfig, WhaleclawConfig


class TestGatewayConfig:
    def test_defaults(self) -> None:
        cfg = GatewayConfig()
        assert cfg.port == 18666
        assert cfg.bind == "127.0.0.1"
        assert cfg.verbose is False

    def test_custom_values(self) -> None:
        cfg = GatewayConfig(port=9000, bind="0.0.0.0", verbose=True)
        assert cfg.port == 9000
        assert cfg.bind == "0.0.0.0"
        assert cfg.verbose is True

    def test_invalid_port(self) -> None:
        with pytest.raises(ValidationError):
            GatewayConfig(port=0)
        with pytest.raises(ValidationError):
            GatewayConfig(port=70000)


class TestAgentConfig:
    def test_defaults(self) -> None:
        cfg = AgentConfig()
        assert "anthropic" in cfg.model
        assert cfg.workspace


class TestWhaleclawConfig:
    def test_defaults(self) -> None:
        cfg = WhaleclawConfig()
        assert cfg.gateway.port == 18666
        assert "anthropic" in cfg.agent.model

    def test_nested_override(self) -> None:
        cfg = WhaleclawConfig.model_validate({"gateway": {"port": 9000}})
        assert cfg.gateway.port == 9000
        assert cfg.gateway.bind == "127.0.0.1"
