"""Tests for the Feishu bot message handler."""

from __future__ import annotations

import json

from whaleclaw.channels.feishu.bot import FeishuBot, _format_exception_text


class TestExtractText:
    def test_text_message(self) -> None:
        message = {
            "message_type": "text",
            "content": json.dumps({"text": "hello world"}),
        }
        assert FeishuBot.extract_text(message) == "hello world"

    def test_post_message(self) -> None:
        message = {
            "message_type": "post",
            "content": json.dumps({
                "content": [
                    [
                        {"tag": "text", "text": "line one"},
                        {"tag": "text", "text": "line two"},
                    ]
                ]
            }),
        }
        assert "line one" in FeishuBot.extract_text(message)

    def test_empty_content(self) -> None:
        message = {"message_type": "text", "content": "{}"}
        assert FeishuBot.extract_text(message) == ""

    def test_invalid_json(self) -> None:
        message = {"message_type": "text", "content": "not json"}
        assert FeishuBot.extract_text(message) == ""


class TestFormatExceptionText:
    def test_empty_exception_message(self) -> None:
        assert _format_exception_text(TimeoutError()) == "TimeoutError"

    def test_non_empty_exception_message(self) -> None:
        assert _format_exception_text(RuntimeError("boom")) == "boom"
