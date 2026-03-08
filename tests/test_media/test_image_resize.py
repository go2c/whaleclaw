from __future__ import annotations

import cv2
import numpy as np

from whaleclaw.media.image_resize import resize_image_long_edge


def _encode_jpeg(width: int, height: int) -> bytes:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return encoded.tobytes()


def test_resize_image_long_edge_keeps_small_image() -> None:
    data = _encode_jpeg(1200, 800)
    result = resize_image_long_edge(data, mime="image/jpeg", max_long_edge=1536)
    assert result.resized is False
    assert result.data == data
    assert result.width == 1200
    assert result.height == 800
    assert result.mime == "image/jpeg"


def test_resize_image_long_edge_downscales_large_image() -> None:
    data = _encode_jpeg(3840, 2160)
    result = resize_image_long_edge(data, mime="image/jpeg", max_long_edge=1536)
    assert result.resized is True
    assert max(result.width, result.height) == 1536
    assert result.mime == "image/jpeg"

    arr = np.frombuffer(result.data, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    assert decoded.shape[1] == result.width
    assert decoded.shape[0] == result.height


def test_resize_image_long_edge_returns_original_for_invalid_bytes() -> None:
    data = b"not-an-image"
    result = resize_image_long_edge(data, mime="image/png", max_long_edge=1536)
    assert result.resized is False
    assert result.data == data
    assert result.mime == "image/png"
