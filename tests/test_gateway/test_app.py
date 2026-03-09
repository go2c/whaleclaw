"""Tests for the Gateway FastAPI application."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from whaleclaw.config.schema import ProviderModelEntry, WhaleclawConfig
from whaleclaw.gateway.app import create_app


@pytest.fixture()
def app():  # noqa: ANN201
    config = WhaleclawConfig()
    return create_app(config)


@pytest.mark.asyncio
async def test_status_endpoint(app) -> None:  # noqa: ANN001
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["gateway"]["port"] == 18666
    assert "compression_ready" in data
    assert "compression_running" in data


@pytest.mark.asyncio
async def test_index_returns_html_or_json(app) -> None:  # noqa: ANN001
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_sessions_crud(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """Test session CRUD via REST API (with lifespan)."""
    import whaleclaw.gateway.app as app_mod
    import whaleclaw.sessions.store as store_mod

    monkeypatch.setattr(store_mod, "_DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(app_mod, "_UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_mod, "_CRON_DB_PATH", tmp_path / "cron.db")

    config = WhaleclawConfig()
    test_app = create_app(config)
    transport = ASGITransport(app=test_app)  # type: ignore[arg-type]
    async with (
        AsyncClient(transport=transport, base_url="http://test") as client,
        test_app.router.lifespan_context(test_app),
    ):
        resp = await client.post("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        sid = data["id"]
        assert sid

        resp = await client.get("/api/sessions")
        assert resp.status_code == 200
        assert any(s["id"] == sid for s in resp.json())

        resp = await client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == sid

        resp = await client.delete(f"/api/sessions/{sid}")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_login_no_password_mode(app) -> None:  # noqa: ANN001
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/auth/login",
            json={"password": "test"},
        )
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_models_oauth_openai_only_shows_gpt52() -> None:
    config = WhaleclawConfig()
    config.models.openai.auth_mode = "oauth"
    config.models.openai.oauth_access = "token"
    config.models.openai.configured_models = [
        ProviderModelEntry(id="gpt-5.2", name="GPT-5.2", verified=True),
        ProviderModelEntry(id="gpt-5.2-codex", name="GPT-5.2 Codex", verified=True),
    ]

    app = create_app(config)
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/models")
    assert resp.status_code == 200
    models = resp.json()["models"]
    ids = [m["id"] for m in models if m["provider"] == "openai"]
    assert ids == ["openai/gpt-5.2", "openai/gpt-5.2-codex"]


@pytest.mark.asyncio
async def test_multi_agent_config_keeps_scenario_and_custom_list(  # noqa: ANN001
    tmp_path,
    monkeypatch,
) -> None:
    import whaleclaw.gateway.app as app_mod
    import whaleclaw.sessions.store as store_mod

    monkeypatch.setattr(store_mod, "_DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(app_mod, "_UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_mod, "_CRON_DB_PATH", tmp_path / "cron.db")
    monkeypatch.setattr(app_mod, "CONFIG_FILE", tmp_path / "whaleclaw.json")

    test_app = create_app(WhaleclawConfig())
    transport = ASGITransport(app=test_app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/plugins/multi-agent",
            json={
                "enabled": True,
                "scenario": "custom::%E8%AF%BB%E4%B9%A6%E4%BC%9A%E8%AE%AE",
                "custom_scenarios": ["读书会议"],
                "mode": "parallel",
                "max_rounds": 4,
                "roles": [],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["scenario"] == "custom::%E8%AF%BB%E4%B9%A6%E4%BC%9A%E8%AE%AE"
        assert data["custom_scenarios"] == ["读书会议"]

        resp = await client.get("/api/plugins/multi-agent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["scenario"] == "custom::%E8%AF%BB%E4%B9%A6%E4%BC%9A%E8%AE%AE"
        assert data["custom_scenarios"] == ["读书会议"]


@pytest.mark.asyncio
async def test_memory_style_rest_api(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    import whaleclaw.gateway.app as app_mod
    import whaleclaw.sessions.store as store_mod

    monkeypatch.setattr(store_mod, "_DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(app_mod, "_UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_mod, "_CRON_DB_PATH", tmp_path / "cron.db")
    monkeypatch.setattr(app_mod, "MEMORY_DIR", tmp_path / "memory")

    config = WhaleclawConfig()
    test_app = create_app(config)
    transport = ASGITransport(app=test_app)  # type: ignore[arg-type]
    async with (
        AsyncClient(transport=transport, base_url="http://test") as client,
        test_app.router.lifespan_context(test_app),
    ):
        resp = await client.get("/api/memory/style")
        assert resp.status_code == 200
        assert resp.json()["has_style"] is False

        resp = await client.post("/api/memory/style", json={"style_directive": "回答简洁明了"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        resp = await client.get("/api/memory/style")
        assert resp.status_code == 200
        assert "简洁明了" in resp.json()["style_directive"]
        assert resp.json()["source"] == "manual"

        resp = await client.delete("/api/memory/style")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        resp = await client.get("/api/memory/style")
        assert resp.status_code == 200
        assert resp.json()["style_directive"] == ""
        assert resp.json()["source"] == "cleared"


@pytest.mark.asyncio
async def test_memory_style_rest_api_falls_back_to_l1_profile(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    import whaleclaw.gateway.app as app_mod
    import whaleclaw.sessions.store as store_mod

    monkeypatch.setattr(store_mod, "_DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(app_mod, "_UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_mod, "_CRON_DB_PATH", tmp_path / "cron.db")
    monkeypatch.setattr(app_mod, "MEMORY_DIR", tmp_path / "memory")

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "memory.json").write_text(
        (
            '{"entries":[{"id":"p1","content":"普通问答默认简洁紧凑，避免冗余客套和过多空行；'
            '制作PPT时图片仅允许裁剪和等比缩放","source":"memory_organizer",'
            '"tags":["memory_profile","level:L1","curated"],"importance":0.5,'
            '"created_at":"2026-03-09T00:00:00+00:00","last_accessed":"2026-03-09T00:00:00+00:00",'
            '"access_count":0,"embedding":null}]}'
        ),
        encoding="utf-8",
    )

    config = WhaleclawConfig()
    test_app = create_app(config)
    transport = ASGITransport(app=test_app)  # type: ignore[arg-type]
    async with (
        AsyncClient(transport=transport, base_url="http://test") as client,
        test_app.router.lifespan_context(test_app),
    ):
        resp = await client.get("/api/memory/style")
        assert resp.status_code == 200
        assert resp.json()["has_style"] is True
        assert "普通问答默认简洁紧凑" in resp.json()["style_directive"]
        assert "制作PPT" not in resp.json()["style_directive"]
        assert resp.json()["source"] == "derived"


@pytest.mark.asyncio
async def test_clawhub_config_and_search_rest_api(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    import whaleclaw.gateway.app as app_mod
    import whaleclaw.sessions.store as store_mod

    monkeypatch.setattr(store_mod, "_DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(app_mod, "_UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_mod, "_CRON_DB_PATH", tmp_path / "cron.db")
    monkeypatch.setattr(app_mod, "CONFIG_FILE", tmp_path / "whaleclaw.json")
    monkeypatch.setattr(app_mod, "WORKSPACE_DIR", tmp_path / "workspace")

    monkeypatch.setattr(app_mod, "is_clawhub_cli_available", lambda: True)
    monkeypatch.setattr(
        app_mod,
        "clawhub_search_skills",
        lambda **_: [
            {
                "slug": "excel-helper",
                "name": "Excel Helper",
                "summary": "表格处理",
                "version": "1.0.0",
            }
        ],
    )

    config = WhaleclawConfig()
    test_app = create_app(config)
    transport = ASGITransport(app=test_app)  # type: ignore[arg-type]
    async with (
        AsyncClient(transport=transport, base_url="http://test") as client,
        test_app.router.lifespan_context(test_app),
    ):
        resp = await client.post(
            "/api/plugins/clawhub",
            json={"enabled": True, "registry_url": "https://clawhub.ai"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert resp.json()["enabled"] is True

        resp = await client.get("/api/clawhub/search", params={"q": "excel"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["slug"] == "excel-helper"


@pytest.mark.asyncio
async def test_clawhub_install_cli_rest_api(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    import whaleclaw.gateway.app as app_mod
    import whaleclaw.sessions.store as store_mod

    monkeypatch.setattr(store_mod, "_DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(app_mod, "_UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_mod, "_CRON_DB_PATH", tmp_path / "cron.db")
    monkeypatch.setattr(
        app_mod,
        "install_clawhub_cli",
        lambda: {"path": "/tmp/clawhub", "version": "0.7.0", "output": "ok"},
    )

    config = WhaleclawConfig()
    test_app = create_app(config)
    transport = ASGITransport(app=test_app)  # type: ignore[arg-type]
    async with (
        AsyncClient(transport=transport, base_url="http://test") as client,
        test_app.router.lifespan_context(test_app),
    ):
        resp = await client.post("/api/clawhub/install-cli")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["cli_available"] is True
        assert data["cli_version"] == "0.7.0"


@pytest.mark.asyncio
async def test_clawhub_auth_status_rest_api(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    import whaleclaw.gateway.app as app_mod
    import whaleclaw.sessions.store as store_mod

    monkeypatch.setattr(store_mod, "_DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(app_mod, "_UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_mod, "_CRON_DB_PATH", tmp_path / "cron.db")
    monkeypatch.setattr(
        app_mod,
        "get_clawhub_auth_status",
        lambda **_: {"logged_in": True, "message": "ok"},
    )

    config = WhaleclawConfig()
    test_app = create_app(config)
    transport = ASGITransport(app=test_app)  # type: ignore[arg-type]
    async with (
        AsyncClient(transport=transport, base_url="http://test") as client,
        test_app.router.lifespan_context(test_app),
    ):
        resp = await client.get("/api/clawhub/auth-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["logged_in"] is True
        assert data["message"] == "ok"


@pytest.mark.asyncio
async def test_clawhub_login_rest_api(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    import whaleclaw.gateway.app as app_mod
    import whaleclaw.sessions.store as store_mod

    monkeypatch.setattr(store_mod, "_DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(app_mod, "_UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_mod, "_CRON_DB_PATH", tmp_path / "cron.db")
    monkeypatch.setattr(
        app_mod,
        "login_clawhub_cli",
        lambda **_: {"ok": True, "message": "已登录", "output": "ok"},
    )

    config = WhaleclawConfig()
    test_app = create_app(config)
    transport = ASGITransport(app=test_app)  # type: ignore[arg-type]
    async with (
        AsyncClient(transport=transport, base_url="http://test") as client,
        test_app.router.lifespan_context(test_app),
    ):
        resp = await client.post("/api/clawhub/login")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["message"] == "已登录"
