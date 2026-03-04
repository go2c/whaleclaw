from __future__ import annotations

import pytest

from whaleclaw.tools.browser import _normalize_image_query


def test_search_images_rejects_multi_intent_query() -> None:
    with pytest.raises(ValueError):
        _normalize_image_query("花、瓶子、苹果")


def test_search_images_accepts_single_intent_query() -> None:
    q = _normalize_image_query("花瓶 高清")
    assert q
