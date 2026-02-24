"""PromptAssembler — layered system prompt builder with token budgeting.

Architecture:
- Static layer (~150 tokens): core identity, always injected
- Dynamic layer (0~800 tokens): skills routed by user message keywords
- Fallback layer: tool descriptions for providers without native tools API

Tool JSON Schemas are passed via the LLM API ``tools`` parameter (not in
the system prompt), so the LLM discovers tool capabilities from the schema
itself — no need to describe tools in the prompt.
"""

from __future__ import annotations

from enum import StrEnum

from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.providers.base import CacheControl, Message
from whaleclaw.skills.manager import SkillManager


class PromptLayer(StrEnum):
    """Prompt layers by priority."""

    STATIC = "static"
    DYNAMIC = "dynamic"
    LAZY = "lazy"


_STATIC_PROMPT = """\
你是 WhaleClaw，一个运行在用户本地电脑上的 AI 助手。
- 使用用户的语言回复，简洁准确
- 你拥有多种工具（bash、文件读写、浏览器、定时任务、技能管理等），能做就做，不要说"我无法"
- 用 bash 执行网络请求时，curl 必须加 --connect-timeout 5 --max-time 15 防止卡住
- 用户要求做的事，先简要告诉用户你的计划（打算怎么做、大概多久），再调用工具执行
- 工具执行失败时尝试其他方案
- 下载或生成的图片用 markdown 显示：![描述](文件绝对路径)
- 生成的其他文件告诉用户绝对路径
- **严禁编造文件路径**：只使用工具返回的真实路径，绝不自己猜测或虚构文件名
- 运行 Python 脚本时使用 python3 命令（不要用 python3 -c，脚本长度会被截断）
- 需要生成文件（PPT/Excel/PDF等）时：先用 file_write 写 .py 脚本到 /tmp/，再用 bash 执行
- 如果你的技能不够好，可以用 skill(action="search", query="关键词") 搜索 GitHub 上更好的技能并安装
- **禁止自我否定已完成的工作**：如果你之前已经成功调用工具生成了文件，不要说"我之前没有真正执行"或"我一直在空谈"。工具调用成功就是成功，直接告诉用户文件路径即可
- 用户说"改一下"时，只修改用户指出的问题，不要推翻重做整个任务"""

_TOOL_FALLBACK_HEADER = """\
【工具调用规则 - 必须严格遵守】

你可以调用工具。调用方式：在回复中输出且**仅输出**以下格式的 JSON 代码块：

```json
{"tool": "工具名", "arguments": {"参数名": "参数值"}}
```

**严禁**：
- 不要用文字描述你打算使用什么工具，直接输出 JSON
- 不要假装工具已经执行，必须输出 JSON 等待真实结果
- 不要编造工具执行结果、文件路径或图片 URL，只使用工具返回的真实数据
- 不要在同一次回复中输出多个 JSON 块

**必须**：
- 需要执行操作时，立即输出一个工具调用 JSON 块，然后停止
- 等收到工具执行结果后，再继续回复
- 搜索/下载图片 → 用 browser 工具的 search_images
- 执行系统命令 → 用 bash
- 读写文件 → 用 file_read / file_write / file_edit

以下是全部可用工具：

"""

_DYNAMIC_BUDGET = 800


class PromptAssembler:
    """Build system prompts within a token budget.

    Static layer: core identity (~150 tokens), cached across turns.
    Dynamic layer: skills routed by SkillManager (0~800 tokens).
    """

    def __init__(self, skill_manager: SkillManager | None = None) -> None:
        self._skill_manager = skill_manager or SkillManager()

    def build(
        self,
        config: WhaleclawConfig,
        user_message: str,
        *,
        token_budget: int | None = None,
        tool_fallback_text: str = "",
    ) -> list[Message]:
        """Assemble system prompt messages.

        Args:
            config: Global configuration.
            user_message: Current user message (used for skill routing).
            token_budget: Max tokens for the system prompt.
            tool_fallback_text: Tool descriptions for providers without
                native tools support. Injected into prompt when non-empty.
        """
        parts: list[str] = [self._build_static(config)]

        dynamic = self._build_dynamic(user_message, _DYNAMIC_BUDGET)
        if dynamic:
            parts.append(dynamic)

        if tool_fallback_text:
            parts.append(_TOOL_FALLBACK_HEADER + "\n" + tool_fallback_text)

        messages: list[Message] = [
            Message(
                role="system",
                content=parts[0],
                cache_control=CacheControl(),
            ),
        ]
        if len(parts) > 1:
            messages.append(
                Message(role="system", content="\n\n".join(parts[1:]))
            )

        return messages

    def _build_static(self, config: WhaleclawConfig) -> str:
        """Static layer — core identity (~150 tokens), cacheable."""
        return _STATIC_PROMPT

    def _build_dynamic(self, user_message: str, budget: int) -> str:
        """Dynamic layer — skills routed by user message keywords."""
        skills = self._skill_manager.get_routed_skills(user_message)
        if not skills:
            return ""
        return self._skill_manager.format_for_prompt(skills, budget)

    def estimate_tokens(self, text: str) -> int:
        """Quick token estimate: ~1.5 chars/token for CJK, ~4 chars/token for Latin."""
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        latin = len(text) - cjk
        return int(cjk / 1.5 + latin / 4)
