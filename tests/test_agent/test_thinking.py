"""Tests for thinking mode and provider params."""

from __future__ import annotations

from whaleclaw.agent.thinking import (
    THINKING_BUDGET,
    ThinkingLevel,
    apply_thinking_params,
)


def test_thinking_level_values() -> None:
    """Verify enum values match expected strings."""
    assert ThinkingLevel.OFF.value == "off"
    assert ThinkingLevel.LOW.value == "low"
    assert ThinkingLevel.MEDIUM.value == "medium"
    assert ThinkingLevel.HIGH.value == "high"
    assert ThinkingLevel.XHIGH.value == "xhigh"
    assert THINKING_BUDGET[ThinkingLevel.OFF] == 0
    assert THINKING_BUDGET[ThinkingLevel.LOW] == 1024
    assert THINKING_BUDGET[ThinkingLevel.MEDIUM] == 4096
    assert THINKING_BUDGET[ThinkingLevel.HIGH] == 8192
    assert THINKING_BUDGET[ThinkingLevel.XHIGH] == 16384


def test_apply_anthropic() -> None:
    """Apply thinking params to anthropic provider sets thinking dict."""
    params: dict[str, object] = {"model": "claude-sonnet-4-20250514", "max_tokens": 8192}
    out = apply_thinking_params(ThinkingLevel.MEDIUM, "anthropic", params)
    assert out["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert out["model"] == "claude-sonnet-4-20250514"


def test_apply_openai() -> None:
    """Verify reasoning_effort set for OpenAI provider."""
    params: dict[str, object] = {"model": "gpt-5.2"}
    out = apply_thinking_params(ThinkingLevel.LOW, "openai", params)
    assert out["reasoning_effort"] == "low"
    out = apply_thinking_params(ThinkingLevel.MEDIUM, "openai", params)
    assert out["reasoning_effort"] == "medium"
    out = apply_thinking_params(ThinkingLevel.HIGH, "openai", params)
    assert out["reasoning_effort"] == "high"
    out = apply_thinking_params(ThinkingLevel.XHIGH, "openai", params)
    assert out["reasoning_effort"] == "high"


def test_apply_deepseek() -> None:
    """Deepseek switches to deepseek-reasoner when level >= HIGH."""
    params: dict[str, object] = {"model": "deepseek-chat"}
    out = apply_thinking_params(ThinkingLevel.LOW, "deepseek", params)
    assert out["model"] == "deepseek-chat"
    out = apply_thinking_params(ThinkingLevel.HIGH, "deepseek", params)
    assert out["model"] == "deepseek-reasoner"
    out = apply_thinking_params(ThinkingLevel.XHIGH, "deepseek", params)
    assert out["model"] == "deepseek-reasoner"


def test_apply_off() -> None:
    """Level OFF should not modify params."""
    params: dict[str, object] = {"model": "claude-sonnet", "foo": "bar"}
    out = apply_thinking_params(ThinkingLevel.OFF, "anthropic", params)
    assert out == params
    assert "thinking" not in out
    out = apply_thinking_params(ThinkingLevel.OFF, "openai", params)
    assert out == params
    assert "reasoning_effort" not in out
    out = apply_thinking_params(ThinkingLevel.OFF, "deepseek", params)
    assert out == params
    assert out["model"] == "claude-sonnet"
