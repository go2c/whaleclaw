"""Image-search planning helpers for agent runtime."""

from __future__ import annotations

import re

from whaleclaw.providers.base import ToolCall

_ZH_NUM_MAP = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def is_search_images_call(tc: ToolCall) -> bool:
    if tc.name != "browser":
        return False
    return str(tc.arguments.get("action", "")).strip().lower() == "search_images"


def normalize_search_images_query(tc: ToolCall) -> str:
    if not is_search_images_call(tc):
        return ""
    text = str(tc.arguments.get("text", "")).strip().lower()
    return re.sub(r"\s+", " ", text)


def parse_simple_zh_number(text: str) -> int | None:
    raw = text.strip()
    if not raw:
        return None
    if raw == "十":
        return 10
    if "十" in raw:
        left, _, right = raw.partition("十")
        tens = 1 if not left else _ZH_NUM_MAP.get(left)
        ones = 0 if not right else _ZH_NUM_MAP.get(right)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    if raw in _ZH_NUM_MAP:
        return _ZH_NUM_MAP[raw]
    return None


def extract_planned_image_count(text: str) -> int | None:
    if not text:
        return None
    m = re.search(r"共\s*([0-9]{1,3})\s*张配图", text)
    if m:
        try:
            value = int(m.group(1))
        except ValueError:
            return None
        return value if value > 0 else None
    m = re.search(r"共\s*([零一二两三四五六七八九十]{1,3})\s*张配图", text)
    if not m:
        return None
    value = parse_simple_zh_number(m.group(1))
    if value is None or value <= 0:
        return None
    return value
