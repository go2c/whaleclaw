"""Tests for ClawHub skill search enrichment logic (CLI only)."""

from __future__ import annotations

from pathlib import Path

from whaleclaw.skills import clawhub


def test_search_skills_enriches_stats_via_cli_inspect(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(clawhub, "is_clawhub_cli_available", lambda: True)
    monkeypatch.setattr(clawhub, "_build_env", lambda **_: {})
    monkeypatch.setattr(clawhub, "_enrich_results_with_cli_explore_stats", lambda **_: None)
    monkeypatch.setattr(
        clawhub,
        "_enrich_results_with_cli_inspect_stats",
        lambda **kwargs: kwargs["items"][0].update(
            {"stars": 12, "downloads": 34, "current_installs": 5}
        ),
    )

    def _fake_run(args: list[str], *, env: dict[str, str]) -> str:  # noqa: ARG001
        if "search" in args and "--json" in args:
            return (
                '[{"slug":"ppt","name":"Ai Ppt Generator (3.696)",'
                '"summary":"demo","version":"1.0.0"}]'
            )
        raise clawhub.ClawHubCliError("no extra stats")

    monkeypatch.setattr(clawhub, "_run", _fake_run)
    monkeypatch.setattr(
        clawhub,
        "_search_via_http",
        lambda **_: (_ for _ in ()).throw(RuntimeError("should not call http")),
    )

    items = clawhub.search_skills(
        query="ppt",
        registry_url="https://clawhub.ai",
        workspace_dir=Path.cwd(),
        limit=20,
    )
    assert len(items) == 1
    assert items[0]["stars"] == 12
    assert items[0]["downloads"] == 34
    assert items[0]["current_installs"] == 5
    assert items[0]["all_time_installs"] == 3696


def test_search_skills_guesses_install_count_from_name_when_stats_unavailable(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(clawhub, "is_clawhub_cli_available", lambda: True)
    monkeypatch.setattr(clawhub, "_build_env", lambda **_: {})
    monkeypatch.setattr(clawhub, "_enrich_results_with_cli_explore_stats", lambda **_: None)
    monkeypatch.setattr(clawhub, "_enrich_results_with_cli_inspect_stats", lambda **_: None)

    def _fake_run(args: list[str], *, env: dict[str, str]) -> str:  # noqa: ARG001
        if "search" in args and "--json" in args:
            return (
                '[{"slug":"ppt-generator","name":"Ppt Generator (3.481)",'
                '"summary":"demo","version":"1.0.0"}]'
            )
        raise clawhub.ClawHubCliError("no extra stats")

    monkeypatch.setattr(clawhub, "_run", _fake_run)
    monkeypatch.setattr(
        clawhub,
        "_search_via_http",
        lambda **_: (_ for _ in ()).throw(RuntimeError),
    )

    items = clawhub.search_skills(
        query="ppt",
        registry_url="https://clawhub.ai",
        workspace_dir=Path.cwd(),
        limit=20,
    )
    assert len(items) == 1
    assert items[0]["all_time_installs"] == 3481
    assert items[0].get("stars") is None
    assert items[0].get("downloads") is None


def test_search_skills_text_fallback_also_enriches_stats(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(clawhub, "is_clawhub_cli_available", lambda: True)
    monkeypatch.setattr(clawhub, "_build_env", lambda **_: {})
    monkeypatch.setattr(clawhub, "_enrich_results_with_cli_explore_stats", lambda **_: None)
    monkeypatch.setattr(
        clawhub,
        "_enrich_results_with_cli_inspect_stats",
        lambda **kwargs: kwargs["items"][0].update(
            {"stars": 99, "downloads": 888, "current_installs": 18}
        ),
    )

    def _fake_run(args: list[str], *, env: dict[str, str]) -> str:  # noqa: ARG001
        if "search" in args and "--json" in args:
            raise clawhub.ClawHubCliError("invalid json")
        if "search" in args:
            return "ai-ppt-generator  Ai Ppt Generator (3.696)"
        raise clawhub.ClawHubCliError("unexpected")

    monkeypatch.setattr(clawhub, "_run", _fake_run)
    monkeypatch.setattr(
        clawhub,
        "_search_via_http",
        lambda **_: (_ for _ in ()).throw(RuntimeError("should not call http")),
    )

    items = clawhub.search_skills(
        query="ppt",
        registry_url="https://clawhub.ai",
        workspace_dir=Path.cwd(),
        limit=20,
    )
    assert len(items) == 1
    assert items[0]["slug"] == "ai-ppt-generator"
    assert items[0]["stars"] == 99
    assert items[0]["downloads"] == 888
    assert items[0]["current_installs"] == 18
