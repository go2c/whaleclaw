"""Memory tools for Agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult

if TYPE_CHECKING:
    from whaleclaw.memory.base import MemoryStore
    from whaleclaw.memory.manager import MemoryManager


class MemorySearchTool(Tool):
    """Search long-term knowledge memory by query."""

    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_search",
            description="按关键词搜索长期知识记忆",
            parameters=[
                ToolParameter(name="query", type="string", description="搜索关键词"),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="最多返回条数，默认 5",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        limit = int(kwargs.get("limit", 5)) if kwargs.get("limit") is not None else 5
        if not query.strip():
            return ToolResult(success=False, output="", error="搜索关键词为空")
        result = await self._manager.recall_knowledge(query, max_tokens=1000, limit=limit)
        return ToolResult(success=True, output=result or "未找到相关记忆")


class MemoryAddTool(Tool):
    """Add content to memory."""

    def __init__(self, manager: MemoryManager) -> None:
        self._manager = manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_add",
            description="将内容存入记忆",
            parameters=[
                ToolParameter(name="content", type="string", description="要存储的内容"),
                ToolParameter(
                    name="tags",
                    type="string",
                    description="标签，逗号分隔，可选",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        content = kwargs.get("content", "")
        tags_str = kwargs.get("tags") or ""
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        if not content.strip():
            return ToolResult(success=False, output="", error="内容为空")
        entry = await self._manager.memorize(content, source="manual", tags=tags)
        return ToolResult(success=True, output=f"已存入记忆，id={entry.id}")


class MemoryListTool(Tool):
    """List recent memories."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_list",
            description="列出最近的记忆",
            parameters=[
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="最多返回条数，默认 10",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        limit = int(kwargs.get("limit", 10)) if kwargs.get("limit") is not None else 10
        entries = await self._store.list_recent(limit=limit)
        lines = [
            f"- [{e.id}] {e.content[:80]}..." if len(e.content) > 80 else f"- [{e.id}] {e.content}"
            for e in entries
        ]
        return ToolResult(success=True, output="\n".join(lines) if lines else "暂无记忆")
