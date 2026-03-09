"""Build and maintain AGENTS.md summary for prompt injection."""

from __future__ import annotations

import re
from pathlib import Path

from whaleclaw.config.paths import WORKSPACE_DIR

_TARGET_CHARS = 2000

_SECTION_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def _estimate_tokens(text: str) -> int:  # pyright: ignore[reportUnusedFunction]
    return max(1, len(text) // 3)


class AgentsSummaryBuilder:
    """Extract key sections from AGENTS.md for ~500 token summary."""

    def build(self, agents_md_path: Path) -> str:
        """Read AGENTS.md and extract headings + first paragraph per section."""
        if not agents_md_path.exists():
            return ""

        raw = agents_md_path.read_text(encoding="utf-8")
        parts: list[str] = []
        current_len = 0

        sections = list(_SECTION_RE.finditer(raw))
        for i, m in enumerate(sections):
            level, title = m.groups()
            start = m.end()
            end = sections[i + 1].start() if i + 1 < len(sections) else len(raw)
            body = raw[start:end].strip()
            para = body.split("\n\n")[0].strip() if body else ""
            block = f"{level} {title}\n\n{para}" if para else f"{level} {title}"
            block = block.strip()
            if block and current_len + len(block) <= _TARGET_CHARS:
                parts.append(block)
                current_len += len(block)
            elif block and not parts:
                parts.append(block[: _TARGET_CHARS - 10] + "...")
                break
            elif current_len >= _TARGET_CHARS:
                break

        return "\n\n".join(parts)

    def rebuild_if_stale(
        self,
        agents_md_path: Path,
        summary_path: Path | None = None,
    ) -> bool:
        """Rebuild summary if AGENTS.md is newer. Returns True if rebuilt."""
        if summary_path is None:
            summary_path = WORKSPACE_DIR / "AGENTS.summary.md"

        if not agents_md_path.exists():
            return False

        agents_mtime = agents_md_path.stat().st_mtime
        if summary_path.exists() and summary_path.stat().st_mtime >= agents_mtime:
            return False

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.build(agents_md_path)
        summary_path.write_text(content, encoding="utf-8")
        return True
