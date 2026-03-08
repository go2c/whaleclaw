"""Lightweight web fetch tool with text/markdown extraction modes."""

from __future__ import annotations

import re
from html import unescape
from typing import Any

import httpx

from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult

_MAX_CHARS = 50_000
_EXTRACT_MODES = ["markdown", "text"]


def _strip_tags(html: str) -> str:
    cleaned = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\\1>", " ", html)
    cleaned = re.sub(r"(?i)</(p|div|section|article|li|h[1-6]|br|tr)>", "\n", cleaned)
    cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"[ \t\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _html_to_markdown_like(html: str) -> str:
    text = html
    text = re.sub(r"(?is)<h1[^>]*>(.*?)</h1>", r"# \1\n\n", text)
    text = re.sub(r"(?is)<h2[^>]*>(.*?)</h2>", r"## \1\n\n", text)
    text = re.sub(r"(?is)<h3[^>]*>(.*?)</h3>", r"### \1\n\n", text)
    text = re.sub(r"(?is)<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", r"[\2](\1)", text)
    text = re.sub(r"(?is)<li[^>]*>(.*?)</li>", r"- \1\n", text)
    text = re.sub(r"(?is)<p[^>]*>(.*?)</p>", r"\1\n\n", text)
    return _strip_tags(text)


class WebFetchTool(Tool):
    """Fetch readable content from a single URL without browser automation."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_fetch",
            description=(
                "Fetch and extract readable content from a URL. "
                "Use for lightweight page access without browser automation. "
                "For multi-page flows or interaction, use browser."
            ),
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="HTTP or HTTPS URL to fetch.",
                ),
                ToolParameter(
                    name="extractMode",
                    type="string",
                    description='Extraction mode: "markdown" or "text".',
                    required=False,
                    enum=_EXTRACT_MODES,
                ),
                ToolParameter(
                    name="maxChars",
                    type="integer",
                    description="Maximum characters to return. Default 12000, max 50000.",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        url = str(kwargs.get("url", "")).strip()
        extract_mode = str(kwargs.get("extractMode", "markdown")).strip().lower() or "markdown"
        max_chars = max(100, min(int(kwargs.get("maxChars", 12_000)), _MAX_CHARS))
        if not url:
            return ToolResult(success=False, output="", error="url 不能为空")
        if extract_mode not in _EXTRACT_MODES:
            return ToolResult(success=False, output="", error="extractMode 仅支持 markdown 或 text")
        if not (url.startswith("http://") or url.startswith("https://")):
            return ToolResult(success=False, output="", error="仅支持 http/https URL")
        try:
            async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/122.0.0.0 Safari/537.36"
                        )
                    },
                )
        except httpx.HTTPError as exc:
            return ToolResult(success=False, output="", error=f"抓取失败: {exc}")
        if response.status_code >= 400:
            return ToolResult(
                success=False,
                output="",
                error=f"抓取失败: HTTP {response.status_code}",
            )
        title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", response.text)
        title = _strip_tags(title_match.group(1)) if title_match else ""
        body = (
            _strip_tags(response.text)
            if extract_mode == "text"
            else _html_to_markdown_like(response.text)
        )
        output = "\n".join(
            [
                *( [f"Title: {title}"] if title else [] ),
                f"URL: {str(response.url)}",
                f"Extractor: {'basic-text' if extract_mode == 'text' else 'basic-markdown'}",
                "",
                body[:max_chars],
            ]
        ).strip()
        return ToolResult(success=True, output=output)
