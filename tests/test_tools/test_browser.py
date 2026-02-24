import pytest

from whaleclaw.tools.browser import _normalize_image_query


def test_normalize_image_query_expand_person_name() -> None:
    q = _normalize_image_query("杨幂")
    assert q == "杨幂 近照 高清 人像"


def test_normalize_image_query_keep_specific_query() -> None:
    q = _normalize_image_query("杨幂 2025 机场 近照 高清")
    assert q == "杨幂 2025 机场 近照 高清"


def test_normalize_image_query_reject_generic() -> None:
    with pytest.raises(ValueError) as exc:
        _normalize_image_query("2")
    assert "无效" in str(exc.value) or "泛化" in str(exc.value)


def test_normalize_image_query_strips_control_chars() -> None:
    q = _normalize_image_query("\x10\x10刘亦菲\x10 写真 高清\x10")
    assert q == "刘亦菲 写真 高清"


def test_normalize_image_query_strips_escaped_noise() -> None:
    q = _normalize_image_query("杨幂 \\n0\\n0\\n0 高清 写真")
    assert q == "杨幂 高清 写真"
