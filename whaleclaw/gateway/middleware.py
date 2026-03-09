"""Authentication middleware for the Gateway."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from base64 import b64decode, b64encode
from typing import Any, cast

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)

from whaleclaw.config.schema import AuthConfig
from whaleclaw.utils.log import get_logger

log = get_logger(__name__)

_PUBLIC_PATHS = {"/", "/api/status", "/api/auth/login"}
_PUBLIC_PREFIXES = ("/static/", "/assets/")


def _make_jwt(secret: str, payload: dict[str, Any], ttl: int) -> str:
    """Minimal HMAC-SHA256 JWT (no external dependency)."""
    header = b64encode(
        json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
    ).decode().rstrip("=")
    payload["exp"] = int(time.time()) + ttl
    body = b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    msg = f"{header}.{body}"
    sig = hmac.new(
        secret.encode(), msg.encode(), hashlib.sha256
    ).digest()
    sig_b64 = b64encode(sig).decode().rstrip("=")
    return f"{msg}.{sig_b64}"


def _verify_jwt(secret: str, token: str) -> dict[str, Any] | None:
    """Verify and decode a JWT. Returns payload or None."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    msg = f"{parts[0]}.{parts[1]}"
    expected_sig = hmac.new(
        secret.encode(), msg.encode(), hashlib.sha256
    ).digest()
    padding = 4 - len(parts[2]) % 4
    try:
        actual_sig = b64decode(parts[2] + "=" * padding)
    except Exception:
        return None
    if not hmac.compare_digest(expected_sig, actual_sig):
        return None
    padding = 4 - len(parts[1]) % 4
    try:
        payload_raw = json.loads(b64decode(parts[1] + "=" * padding))
    except Exception:
        return None
    if not isinstance(payload_raw, dict):
        return None
    payload = cast(dict[str, Any], payload_raw)
    if payload.get("exp", 0) < time.time():
        return None
    return payload


def create_jwt(config: AuthConfig) -> str:
    """Create a JWT with the configured secret and TTL."""
    return _make_jwt(
        config.jwt_secret,
        {"sub": "webchat"},
        config.jwt_expire_hours * 3600,
    )


def verify_jwt(config: AuthConfig, token: str) -> bool:
    """Verify a JWT against the configured secret."""
    return _verify_jwt(config.jwt_secret, token) is not None


class AuthMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that enforces authentication."""

    def __init__(self, app: Any, auth_config: AuthConfig) -> None:  # noqa: ANN401
        super().__init__(app)
        self._config = auth_config

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if self._config.mode == "none":
            response = await call_next(request)
            return cast(Response, response)

        path = request.url.path
        if path in _PUBLIC_PATHS:
            response = await call_next(request)
            return cast(Response, response)
        for prefix in _PUBLIC_PREFIXES:
            if path.startswith(prefix):
                response = await call_next(request)
                return cast(Response, response)

        if request.scope.get("type") == "websocket":
            token = request.query_params.get("token", "")
        else:
            auth_header = request.headers.get("authorization", "")
            token = auth_header[7:] if auth_header.startswith("Bearer ") else ""
            if not token:
                token = request.query_params.get("token", "")

        if self._config.mode == "token":
            if token == self._config.token:
                response = await call_next(request)
                return cast(Response, response)
            return JSONResponse(
                status_code=401,
                content={"error": "认证失败"},
            )

        if self._config.mode == "password":
            if verify_jwt(self._config, token):
                response = await call_next(request)
                return cast(Response, response)
            return JSONResponse(
                status_code=401,
                content={"error": "认证失败，请先登录"},
            )

        response = await call_next(request)
        return cast(Response, response)
