"""OpenAI ChatGPT OAuth 2.0 + PKCE login flow.

Allows users to authenticate with their ChatGPT Plus/Pro account
instead of providing an API key. Uses the same OAuth parameters
as the official Codex CLI.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from whaleclaw.utils.log import get_logger

log = get_logger(__name__)

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"
JWT_CLAIM_PATH = "https://api.openai.com/auth"

_CALLBACK_PORT = 1455
_CALLBACK_TIMEOUT = 120


class OAuthResult:
    """Holds the result of a successful OAuth flow."""

    def __init__(
        self,
        access: str,
        refresh: str,
        expires: int,
        account_id: str,
    ) -> None:
        self.access = access
        self.refresh = refresh
        self.expires = expires
        self.account_id = account_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "access": self.access,
            "refresh": self.refresh,
            "expires": self.expires,
            "account_id": self.account_id,
        }


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE verifier and S256 challenge."""
    verifier_bytes = secrets.token_bytes(32)
    verifier = _base64url(verifier_bytes)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = _base64url(digest)
    return verifier, challenge


def _decode_jwt_payload(token: str) -> dict[str, Any] | None:
    """Decode JWT payload without verification (for extracting claims)."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    padding = 4 - len(payload) % 4
    if padding != 4:
        payload += "=" * padding
    try:
        decoded = base64.urlsafe_b64decode(payload)
        return json.loads(decoded)
    except (ValueError, json.JSONDecodeError):
        return None


def get_account_id(access_token: str) -> str | None:
    """Extract ChatGPT account_id from the JWT access token."""
    payload = _decode_jwt_payload(access_token)
    if not payload:
        return None
    auth = payload.get(JWT_CLAIM_PATH)
    if isinstance(auth, dict):
        aid = auth.get("chatgpt_account_id")
        if isinstance(aid, str) and aid:
            return aid
    return None


def build_authorize_url(state: str, challenge: str) -> str:
    """Build the full OpenAI OAuth authorize URL."""
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "whaleclaw",
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def _http_post_form(url: str, data: dict[str, str]) -> dict[str, Any]:
    """POST form-urlencoded data and return JSON response."""
    encoded = urlencode(data).encode("utf-8")
    req = Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def exchange_code(code: str, verifier: str) -> OAuthResult:
    """Exchange an authorization code for access + refresh tokens."""
    data = {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "code_verifier": verifier,
        "redirect_uri": REDIRECT_URI,
    }
    result = _http_post_form(TOKEN_URL, data)
    access = result.get("access_token", "")
    refresh = result.get("refresh_token", "")
    expires_in = result.get("expires_in", 0)
    if not access or not refresh:
        raise RuntimeError(f"Token 交换失败: {result}")
    account_id = get_account_id(access)
    if not account_id:
        raise RuntimeError("无法从 token 中提取 account_id")
    return OAuthResult(
        access=access,
        refresh=refresh,
        expires=int(time.time()) + int(expires_in),
        account_id=account_id,
    )


def refresh_access_token(refresh_token: str) -> OAuthResult:
    """Refresh an expired access token."""
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID,
    }
    result = _http_post_form(TOKEN_URL, data)
    access = result.get("access_token", "")
    new_refresh = result.get("refresh_token", "")
    expires_in = result.get("expires_in", 0)
    if not access or not new_refresh:
        raise RuntimeError(f"Token 刷新失败: {result}")
    account_id = get_account_id(access)
    if not account_id:
        raise RuntimeError("无法从刷新后的 token 中提取 account_id")
    return OAuthResult(
        access=access,
        refresh=new_refresh,
        expires=int(time.time()) + int(expires_in),
        account_id=account_id,
    )


class _CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures the OAuth callback code."""

    code: str | None = None
    expected_state: str = ""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return

        qs = parse_qs(parsed.query)
        state = qs.get("state", [""])[0]
        if state != self.expected_state:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"State mismatch")
            return

        code = qs.get("code", [""])[0]
        if not code:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing authorization code")
            return

        _CallbackHandler.code = code
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            b"<html><body><p>"
            b"\xe8\xae\xa4\xe8\xaf\x81\xe6\x88\x90\xe5\x8a\x9f\xef\xbc\x81"
            b"\xe8\xaf\xb7\xe8\xbf\x94\xe5\x9b\x9e\xe7\xbb\x88\xe7\xab\xaf\xe7\xbb\xa7\xe7\xbb\xad\xe3\x80\x82"
            b"</p></body></html>"
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def login() -> OAuthResult:
    """Run the full OAuth login flow (blocking).

    1. Generate PKCE verifier/challenge
    2. Start local HTTP callback server on :1455
    3. Open browser to OpenAI authorize URL
    4. Wait for callback with authorization code
    5. Exchange code for tokens
    """
    verifier, challenge = generate_pkce()
    state = secrets.token_hex(16)

    _CallbackHandler.code = None
    _CallbackHandler.expected_state = state

    server = HTTPServer(("127.0.0.1", _CALLBACK_PORT), _CallbackHandler)
    server.timeout = 1

    auth_url = build_authorize_url(state, challenge)

    print(f"\n  正在打开浏览器进行 ChatGPT 授权…")
    print(f"  如果浏览器未自动打开，请手动访问:")
    print(f"  {auth_url}\n")

    webbrowser.open(auth_url)

    deadline = time.time() + _CALLBACK_TIMEOUT
    try:
        while time.time() < deadline and _CallbackHandler.code is None:
            server.handle_request()
    finally:
        server.server_close()

    code = _CallbackHandler.code
    if not code:
        raise RuntimeError("授权超时，未收到回调。请重试。")

    print("  ✅ 授权码已接收，正在交换 Token…")
    result = exchange_code(code, verifier)
    print("  ✅ 登录成功！")
    return result


def save_oauth_to_config(result: OAuthResult, config_path: Path | None = None) -> None:
    """Save OAuth credentials into the whaleclaw.json config file."""
    if config_path is None:
        config_path = Path(os.path.expanduser("~/.whaleclaw/whaleclaw.json"))

    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        cfg = {}

    openai_cfg = cfg.setdefault("models", {}).setdefault("openai", {})
    openai_cfg["auth_mode"] = "oauth"
    openai_cfg["oauth_access"] = result.access
    openai_cfg["oauth_refresh"] = result.refresh
    openai_cfg["oauth_expires"] = result.expires
    openai_cfg["oauth_account_id"] = result.account_id

    config_path.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("openai_oauth.saved", account_id=result.account_id)


def ensure_valid_token(config_path: Path | None = None) -> OAuthResult | None:
    """Check if the stored OAuth token is valid; refresh if expired.

    Returns the (possibly refreshed) OAuthResult, or None if no
    OAuth credentials are configured.
    """
    if config_path is None:
        config_path = Path(os.path.expanduser("~/.whaleclaw/whaleclaw.json"))

    if not config_path.exists():
        return None

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    openai_cfg = cfg.get("models", {}).get("openai", {})

    if openai_cfg.get("auth_mode") != "oauth":
        return None

    access = openai_cfg.get("oauth_access", "")
    refresh = openai_cfg.get("oauth_refresh", "")
    expires = openai_cfg.get("oauth_expires", 0)
    account_id = openai_cfg.get("oauth_account_id", "")

    if not access or not refresh:
        return None

    now = int(time.time())
    if now < expires - 60:
        return OAuthResult(
            access=access,
            refresh=refresh,
            expires=expires,
            account_id=account_id,
        )

    log.info("openai_oauth.refreshing", reason="token expired")
    try:
        result = refresh_access_token(refresh)
        save_oauth_to_config(result, config_path)
        return result
    except Exception:
        log.warning("openai_oauth.refresh_failed")
        return None
