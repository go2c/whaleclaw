"""Image resize helpers for inbound user images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_DEFAULT_MAX_LONG_EDGE = 1536


@dataclass(frozen=True)
class ImageResizeResult:
    """Result of image resize decision."""

    data: bytes
    mime: str | None
    resized: bool
    width: int
    height: int


def resize_image_long_edge(
    data: bytes,
    *,
    mime: str | None = None,
    max_long_edge: int = _DEFAULT_MAX_LONG_EDGE,
) -> ImageResizeResult:
    """Resize image so its long edge is at most ``max_long_edge``.

    Small images are returned unchanged.
    """
    if max_long_edge < 1:
        return ImageResizeResult(data=data, mime=mime, resized=False, width=0, height=0)

    raw = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None:
        return ImageResizeResult(data=data, mime=mime, resized=False, width=0, height=0)

    height, width = image.shape[:2]
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        out_mime = mime or _guess_mime_from_bytes(data)
        return ImageResizeResult(
            data=data,
            mime=out_mime,
            resized=False,
            width=width,
            height=height,
        )

    scale = max_long_edge / float(long_edge)
    target_w = max(1, int(round(width * scale)))
    target_h = max(1, int(round(height * scale)))
    resized_img = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    ext, out_mime, params = _encoding_for_mime(mime)
    ok, encoded = cv2.imencode(ext, resized_img, params)
    if not ok:
        ok, encoded = cv2.imencode(".png", resized_img)
        if not ok:
            return ImageResizeResult(
                data=data,
                mime=mime or _guess_mime_from_bytes(data),
                resized=False,
                width=width,
                height=height,
            )
        out_mime = "image/png"

    return ImageResizeResult(
        data=encoded.tobytes(),
        mime=out_mime,
        resized=True,
        width=target_w,
        height=target_h,
    )


def _encoding_for_mime(mime: str | None) -> tuple[str, str, list[int]]:
    normalized = (mime or "").strip().lower()
    if normalized == "image/png":
        return ".png", "image/png", [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if normalized == "image/webp":
        return ".webp", "image/webp", [cv2.IMWRITE_WEBP_QUALITY, 90]
    return ".jpg", "image/jpeg", [cv2.IMWRITE_JPEG_QUALITY, 90]


def _guess_mime_from_bytes(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return None


__all__ = ["ImageResizeResult", "resize_image_long_edge"]
