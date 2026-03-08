"""Node invoke tool — Call device node actions."""

from __future__ import annotations

import json
from typing import Any, cast

from whaleclaw.nodes.manager import NodeManager
from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult


class NodeInvokeTool(Tool):
    """Invoke actions on registered device nodes."""

    def __init__(self, node_manager: NodeManager) -> None:
        self._manager = node_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="node_invoke",
            description="调用已注册设备节点的能力 (如 camera.snap, notification 等)。",
            parameters=[
                ToolParameter(
                    name="node_id",
                    type="string",
                    description="节点 ID。",
                ),
                ToolParameter(
                    name="action",
                    type="string",
                    description="要执行的能力或操作，如 camera.snap, notification。",
                ),
                ToolParameter(
                    name="params",
                    type="string",
                    description="JSON 格式的参数字符串，可选。",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        node_id = str(kwargs.get("node_id", ""))
        action = str(kwargs.get("action", ""))
        params_raw = kwargs.get("params")

        if not node_id:
            return ToolResult(success=False, output="", error="node_id 不能为空")
        if not action:
            return ToolResult(success=False, output="", error="action 不能为空")

        params: dict[str, object] = {}
        if params_raw:
            try:
                parsed_params = json.loads(str(params_raw))
                if not isinstance(parsed_params, dict):
                    params = {}
                else:
                    params = {
                        str(key): value
                        for key, value in cast(dict[object, object], parsed_params).items()
                        if isinstance(key, str)
                    }
            except json.JSONDecodeError:
                return ToolResult(success=False, output="", error="params 不是有效 JSON")

        raw_result: dict[str, object] = await self._manager.invoke(node_id, action, params)
        result = {
            str(key): value
            for key, value in raw_result.items()
        }
        output = json.dumps(result)
        success = result.get("status") not in {"error", "not_implemented"}
        error_value = result.get("error")
        error = str(error_value) if not success and error_value is not None else None
        return ToolResult(
            success=success,
            output=output,
            error=error,
        )
