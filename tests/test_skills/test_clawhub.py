"""Tests for ClawHub skill search behavior."""

from __future__ import annotations

from pathlib import Path

from whaleclaw.skills import clawhub


def test_search_skills_uses_http_search_and_http_detail(monkeypatch) -> None:  # noqa: ANN001
    clawhub._search_cache.clear()
    clawhub._detail_cache.clear()

    monkeypatch.setattr(
        clawhub,
        "_search_via_http",
        lambda **_: [
            {
                "slug": "python-executor",
                "name": "Python Executor",
                "summary": "demo",
                "version": "1.0.0",
                "stars": 10,
                "downloads": 20,
                "current_installs": 3,
                "all_time_installs": 50,
                "detail_url": "https://clawhub.ai/skills/python-executor",
                "repo_url": "",
            }
        ],
    )
    monkeypatch.setattr(
        clawhub,
        "_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cli should not run")),
    )

    items = clawhub.search_skills(
        query="python",
        registry_url="https://clawhub.ai",
        workspace_dir=Path.cwd(),
        limit=20,
    )
    assert len(items) == 1
    assert items[0]["slug"] == "python-executor"
    assert items[0]["stars"] == 10
    assert items[0]["downloads"] == 20


def test_search_via_http_does_not_re_filter_results(monkeypatch) -> None:  # noqa: ANN001
    clawhub._detail_cache.clear()
    monkeypatch.setattr(clawhub, "_enrich_results_with_skill_details", lambda **kwargs: None)

    class _Resp:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {
                "results": [
                    {
                        "slug": "x402-creative-resources",
                        "displayName": "x402 Creative Resources",
                        "summary": (
                            "Access Xona's x402 creative resource APIs "
                            "on api.xona-agent.com."
                        ),
                        "version": "1.0.0",
                    },
                    {
                        "slug": "nano-banana-2",
                        "displayName": "Nano Banana 2",
                        "summary": "Generate images with Gemini 3.1 Flash Image Preview.",
                        "version": "0.1.1",
                    },
                ]
            }

    class _Client:
        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ANN204
            return False

        def get(self, url: str, params: dict[str, object], headers: dict[str, str]) -> _Resp:  # noqa: ARG002
            return _Resp()

    monkeypatch.setattr(clawhub.httpx, "Client", lambda **kwargs: _Client())

    items = clawhub._search_via_http(
        query="banana",
        registry_url="https://clawhub.ai",
        api_token=None,
        limit=20,
    )
    assert [str(item["slug"]) for item in items] == [
        "nano-banana-2",
        "x402-creative-resources",
    ]


def test_enrich_results_with_skill_details_covers_more_than_first_page(monkeypatch) -> None:  # noqa: ANN001
    clawhub._detail_cache.clear()
    rows = [
        {"slug": f"skill-{i}", "detail_url": "", "stars": None, "downloads": None}
        for i in range(10)
    ]

    def _fake_fetch(**kwargs) -> dict[str, object]:  # noqa: ANN001
        slug = str(kwargs["slug"])
        index = int(slug.split("-")[-1])
        return {"stars": index, "downloads": index * 10}

    monkeypatch.setattr(clawhub, "_fetch_skill_detail_enrichment", _fake_fetch)
    monkeypatch.setattr(clawhub, "_inspect_skill_stats_via_cli", lambda **_: {})

    clawhub._enrich_results_with_skill_details(
        items=rows,
        registry_url="https://clawhub.ai",
        api_token=None,
        max_items=10,
    )
    assert all(row["stars"] is not None for row in rows)
    assert all(row["downloads"] is not None for row in rows)


def test_enrich_results_skips_detail_when_sort_keys_are_ready(monkeypatch) -> None:  # noqa: ANN001
    clawhub._detail_cache.clear()
    rows = [
        {
            "slug": "ready",
            "detail_url": "",
            "stars": 5,
            "downloads": 7,
            "current_installs": None,
            "all_time_installs": 9,
        },
        {
            "slug": "need-detail",
            "detail_url": "",
            "stars": None,
            "downloads": None,
            "current_installs": None,
            "all_time_installs": None,
        },
    ]
    called: list[str] = []

    def _fake_fetch(**kwargs) -> dict[str, object]:  # noqa: ANN001
        called.append(str(kwargs["slug"]))
        return {
            "stars": 1,
            "downloads": 2,
            "current_installs": 3,
            "all_time_installs": 4,
        }

    monkeypatch.setattr(clawhub, "_fetch_skill_detail_enrichment", _fake_fetch)
    monkeypatch.setattr(clawhub, "_inspect_skill_stats_via_cli", lambda **_: {})

    clawhub._enrich_results_with_skill_details(
        items=rows,
        registry_url="https://clawhub.ai",
        api_token=None,
        max_items=10,
    )
    assert called == ["need-detail"]
    assert rows[0]["downloads"] == 7
    assert rows[1]["downloads"] == 2
