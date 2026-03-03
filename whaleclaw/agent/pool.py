"""Agent instance pool."""

from __future__ import annotations

from pydantic import BaseModel

from whaleclaw.config.paths import WORKSPACE_DIR


class AgentInstance(BaseModel):
    """Single agent instance with workspace and config."""

    id: str
    workspace: str
    model: str | None = None
    tools: list[str] | None = None


class AgentPool:
    """Pool of agent instances, keyed by agent_id."""

    def __init__(self) -> None:
        self._instances: dict[str, AgentInstance] = {}

    async def get_or_create(
        self,
        agent_id: str,
        workspace: str | None = None,
        model: str | None = None,
        tools: list[str] | None = None,
    ) -> AgentInstance:
        if agent_id not in self._instances:
            ws = workspace or str(WORKSPACE_DIR)
            self._instances[agent_id] = AgentInstance(
                id=agent_id,
                workspace=ws,
                model=model,
                tools=tools,
            )
        return self._instances[agent_id]

    async def destroy(self, agent_id: str) -> None:
        self._instances.pop(agent_id, None)

    def list_agents(self) -> list[str]:
        return list(self._instances.keys())
