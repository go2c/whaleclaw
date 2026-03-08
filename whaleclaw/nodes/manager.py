"""Device node registration and invocation."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel


class DeviceNode(BaseModel):
    """Registered device node."""

    id: str
    name: str
    platform: str
    capabilities: list[str] = []
    connected_at: datetime
    last_heartbeat: datetime


class NodeManager:
    """Manages device node registration and invocation."""

    def __init__(self) -> None:
        self._nodes: dict[str, DeviceNode] = {}

    async def register(self, node: DeviceNode) -> None:
        """Register a device node."""
        self._nodes[node.id] = node

    async def unregister(self, node_id: str) -> None:
        """Unregister a device node."""
        self._nodes.pop(node_id, None)

    async def list_nodes(self) -> list[DeviceNode]:
        """List all registered nodes."""
        return list(self._nodes.values())

    async def invoke(
        self, node_id: str, action: str, params: dict[str, object]
    ) -> dict[str, object]:
        """Invoke action on node. Stub returns not_implemented."""
        if node_id not in self._nodes:
            return {"status": "error", "error": "节点不存在"}
        return {
            "status": "not_implemented",
            "error": "节点调用尚未实现",
        }

    async def heartbeat(self, node_id: str) -> bool:
        """Update last_heartbeat. Return True if node found."""
        node = self._nodes.get(node_id)
        if node is None:
            return False
        node.last_heartbeat = datetime.now(UTC)
        return True
