"""Message routing engine."""

from __future__ import annotations

from pydantic import BaseModel

from whaleclaw.channels.base import ChannelMessage
from whaleclaw.config.paths import WORKSPACE_DIR
from whaleclaw.routing.rules import RoutingRule, RoutingTarget
from whaleclaw.security.permissions import SecurityPolicy, ToolPermission


class RoutingResult(BaseModel):
    """Result of routing a message."""

    agent_id: str
    session_id: str
    workspace: str
    security_policy: SecurityPolicy
    allowed: bool = True
    deny_reason: str | None = None


def _build_session_id(msg: ChannelMessage, agent_id: str) -> str:
    if msg.group_id:
        return f"{msg.channel}:{msg.group_id}:{msg.peer_id}"
    return f"{msg.channel}:{msg.peer_id}"


def _security_policy_from_target(target: RoutingTarget) -> SecurityPolicy:
    tools = ToolPermission()
    if target.tools is not None:
        tools = ToolPermission(allow=target.tools, deny=[])
    return SecurityPolicy(sandbox=target.sandbox, tools=tools)


class MessageRouter:
    """Routes messages to agents based on rules."""

    def __init__(
        self,
        rules: list[RoutingRule],
        default_target: RoutingTarget | None = None,
    ) -> None:
        self._rules = rules
        self._default = default_target or RoutingTarget()

    async def route(self, message: ChannelMessage) -> RoutingResult:
        sorted_rules = sorted(self._rules, key=lambda r: r.priority, reverse=True)
        target = self._default
        for rule in sorted_rules:
            if rule.match.matches(message):
                target = rule.target
                break
        session_id = _build_session_id(message, target.agent_id)
        workspace = target.workspace or str(WORKSPACE_DIR)
        security_policy = _security_policy_from_target(target)
        return RoutingResult(
            agent_id=target.agent_id,
            session_id=session_id,
            workspace=workspace,
            security_policy=security_policy,
        )
