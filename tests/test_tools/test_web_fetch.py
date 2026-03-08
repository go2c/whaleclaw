"""Tests for web fetch tool."""

from __future__ import annotations

import pytest

from whaleclaw.tools.web_fetch import WebFetchTool


@pytest.fixture()
def tool() -> WebFetchTool:
    return WebFetchTool()


@pytest.mark.asyncio
async def test_web_fetch_rejects_non_http_url(tool: WebFetchTool) -> None:
    result = await tool.execute(url="file:///tmp/demo.txt")
    assert not result.success
    assert result.error == "仅支持 http/https URL"


@pytest.mark.asyncio
async def test_web_fetch_extracts_title_and_text(monkeypatch, tool: WebFetchTool) -> None:  # noqa: ANN001
    class _Resp:
        status_code = 200
        text = (
            "<html><head><title>Demo</title></head>"
            "<body><article><h1>Hello</h1><p>World</p></article></body></html>"
        )
        url = "https://example.com/demo"

    class _Client:
        async def __aenter__(self):  # noqa: ANN204
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ANN204
            return False

        async def get(self, url: str, headers: dict[str, str]) -> _Resp:  # noqa: ARG002
            return _Resp()

    monkeypatch.setattr("whaleclaw.tools.web_fetch.httpx.AsyncClient", lambda **kwargs: _Client())
    result = await tool.execute(url="https://example.com/demo", extractMode="text")
    assert result.success
    assert "Title: Demo" in result.output
    assert "Extractor: basic-text" in result.output
    assert "Hello" in result.output
    assert "World" in result.output


@pytest.mark.asyncio
async def test_web_fetch_extracts_markdown_mode(monkeypatch, tool: WebFetchTool) -> None:  # noqa: ANN001
    class _Resp:
        status_code = 200
        text = (
            "<html><head><title>Demo</title></head>"
            "<body><h1>Hello</h1><p><a href='https://a.com'>Link</a></p></body></html>"
        )
        url = "https://example.com/demo"

    class _Client:
        async def __aenter__(self):  # noqa: ANN204
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ANN204
            return False

        async def get(self, url: str, headers: dict[str, str]) -> _Resp:  # noqa: ARG002
            return _Resp()

    monkeypatch.setattr("whaleclaw.tools.web_fetch.httpx.AsyncClient", lambda **kwargs: _Client())
    result = await tool.execute(url="https://example.com/demo", extractMode="markdown")
    assert result.success
    assert "Extractor: basic-markdown" in result.output
    assert "Hello" in result.output
