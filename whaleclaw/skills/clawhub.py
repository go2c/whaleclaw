"""ClawHub CLI integration for searching and installing skills."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import httpx

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL | re.MULTILINE)


class ClawHubCliError(RuntimeError):
    """Raised when ClawHub CLI command execution fails."""


def is_clawhub_cli_available() -> bool:
    """Return whether ``clawhub`` executable exists in PATH."""
    return _resolve_clawhub_bin() is not None


def get_clawhub_cli_status() -> dict[str, str | bool]:
    """Return CLI availability, path and version (if available)."""
    bin_path = _resolve_clawhub_bin()
    if not bin_path:
        return {"available": False, "path": "", "version": ""}
    version = ""
    with contextlib.suppress(Exception):
        version = _run([bin_path, "--cli-version"], env=os.environ.copy()).strip()
    return {"available": True, "path": bin_path, "version": version}


def _resolve_clawhub_bin() -> str | None:
    env_bin = os.environ.get("CLAWHUB_BIN", "").strip()
    if env_bin:
        p = Path(env_bin).expanduser()
        if p.is_file():
            return str(p)
    path_bin = shutil.which("clawhub")
    if path_bin:
        return path_bin
    roots = [
        Path.cwd(),
        Path(__file__).resolve().parents[2],
    ]
    for root in roots:
        local_bin = root / ".local" / "npm-global" / "bin" / "clawhub"
        if local_bin.is_file():
            return str(local_bin)
    return None


def _resolve_npm_bin() -> str | None:
    env_npm = os.environ.get("CLAWHUB_NPM_BIN", "").strip()
    if env_npm:
        p = Path(env_npm).expanduser()
        if p.is_file():
            return str(p)
    path_npm = shutil.which("npm")
    if path_npm:
        return path_npm
    roots = [
        Path.cwd(),
        Path(__file__).resolve().parents[2],
    ]
    for root in roots:
        local_npm = root / ".local" / "node" / "bin" / "npm"
        if local_npm.is_file():
            return str(local_npm)
    return None


def _build_env(
    *,
    registry_url: str,
    workspace_dir: Path,
    api_token: str | None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["CLAWHUB_REGISTRY"] = registry_url
    env["CLAWHUB_WORKDIR"] = str(workspace_dir)
    if api_token:
        env["CLAWHUB_TOKEN"] = api_token
    repo_root = Path(__file__).resolve().parents[2]
    local_node_bin = repo_root / ".local" / "node" / "bin"
    local_npm_bin = repo_root / ".local" / "npm-global" / "bin"
    path_parts = [str(local_npm_bin), str(local_node_bin), env.get("PATH", "")]
    env["PATH"] = ":".join(part for part in path_parts if part)
    return env


def _run(args: list[str], *, env: dict[str, str]) -> str:
    bin_path = _resolve_clawhub_bin()
    if not bin_path:
        raise ClawHubCliError("未检测到 clawhub CLI")
    if args and args[0] == "clawhub":
        args = [bin_path, *args[1:]]
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=45,
        env=env,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip() or "未知错误"
        raise ClawHubCliError(err)
    return (proc.stdout or "").strip()


def install_clawhub_cli() -> dict[str, str]:
    """Install clawhub CLI into project-local prefix via npm."""
    npm_bin = _resolve_npm_bin()
    if not npm_bin:
        raise ClawHubCliError("未检测到 npm，请先安装 Node.js")

    repo_root = Path(__file__).resolve().parents[2]
    prefix_dir = repo_root / ".local" / "npm-global"
    prefix_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    local_node_bin = repo_root / ".local" / "node" / "bin"
    path_parts = [str(prefix_dir / "bin"), str(local_node_bin), env.get("PATH", "")]
    env["PATH"] = ":".join(p for p in path_parts if p)

    proc = subprocess.run(
        [npm_bin, "i", "-g", "clawhub", "--prefix", str(prefix_dir)],
        capture_output=True,
        text=True,
        timeout=240,
        env=env,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip() or "未知错误"
        raise ClawHubCliError(f"安装失败: {err}")

    status = get_clawhub_cli_status()
    if not bool(status["available"]):
        raise ClawHubCliError("安装完成但未找到 clawhub 可执行文件")
    return {
        "path": str(status["path"]),
        "version": str(status["version"]),
        "output": (proc.stdout or "").strip(),
    }


def get_clawhub_auth_status(
    *,
    registry_url: str,
    workspace_dir: Path,
    api_token: str | None = None,
) -> dict[str, str | bool]:
    """Return CLI auth status by running ``clawhub whoami``."""
    if not is_clawhub_cli_available():
        return {"logged_in": False, "message": "CLI 未安装"}
    env = _build_env(
        registry_url=registry_url,
        workspace_dir=workspace_dir,
        api_token=api_token,
    )
    try:
        out = _run(["clawhub", "whoami"], env=env)
        msg = out.strip() or "已登录"
        return {"logged_in": True, "message": msg}
    except Exception as exc:
        raw = str(exc)
        cleaned = _clean_cli_error(raw)
        return {"logged_in": False, "message": cleaned}


def login_clawhub_cli(
    *,
    registry_url: str,
    workspace_dir: Path,
    api_token: str | None = None,
) -> dict[str, str | bool]:
    """Run ``clawhub login`` and return post-login status."""
    if not is_clawhub_cli_available():
        raise ClawHubCliError("CLI 未安装，请先点击“安装 CLI”")
    env = _build_env(
        registry_url=registry_url,
        workspace_dir=workspace_dir,
        api_token=api_token,
    )
    proc = subprocess.run(
        [_resolve_clawhub_bin() or "clawhub", "login"],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip() or "登录失败"
        raise ClawHubCliError(_clean_cli_error(err))
    status = get_clawhub_auth_status(
        registry_url=registry_url,
        workspace_dir=workspace_dir,
        api_token=api_token,
    )
    return {
        "ok": bool(status["logged_in"]),
        "message": str(status["message"]),
        "output": (proc.stdout or "").strip(),
    }


def logout_clawhub_cli(
    *,
    registry_url: str,
    workspace_dir: Path,
    api_token: str | None = None,
) -> dict[str, str | bool]:
    """Run ``clawhub logout`` and return post-logout status."""
    if not is_clawhub_cli_available():
        raise ClawHubCliError("CLI 未安装，请先点击“安装 CLI”")
    env = _build_env(
        registry_url=registry_url,
        workspace_dir=workspace_dir,
        api_token=api_token,
    )
    proc = subprocess.run(
        [_resolve_clawhub_bin() or "clawhub", "logout"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    if proc.returncode != 0:
        err = _clean_cli_error((proc.stderr or proc.stdout).strip() or "退出登录失败")
        if "未登录" not in err and "Not logged in" not in err:
            raise ClawHubCliError(err)

    status = get_clawhub_auth_status(
        registry_url=registry_url,
        workspace_dir=workspace_dir,
        api_token=api_token,
    )
    return {
        "ok": not bool(status["logged_in"]),
        "message": "已退出登录" if not bool(status["logged_in"]) else str(status["message"]),
        "output": (proc.stdout or "").strip(),
    }


def _clean_cli_error(message: str) -> str:
    lines = [ln.strip() for ln in message.splitlines() if ln.strip()]
    filtered = [
        ln
        for ln in lines
        if "ExperimentalWarning" not in ln and "--trace-warnings" not in ln
    ]
    text = "\n".join(filtered) if filtered else message.strip()
    if "Not logged in" in text or "Run: clawhub login" in text:
        return "未登录，请在终端执行: clawhub login"
    return text


def _parse_search_text(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("slug") or set(line) <= {"-", " "}:
            continue

        # typical line style: "my-skill  A short summary..."
        m = re.match(r"^([a-zA-Z0-9_.-]{2,})\s+(.*)$", line)
        if not m:
            continue
        slug = m.group(1).strip()
        summary = m.group(2).strip()
        if not slug:
            continue
        rows.append(
            {
                "slug": slug,
                "name": slug,
                "summary": summary,
                "version": "",
                "detail_url": "",
                "repo_url": "",
            }
        )
    return rows


def _decorate_results(
    *,
    items: list[dict[str, object]],
    registry_url: str,
) -> list[dict[str, object]]:
    base = registry_url.rstrip("/")
    out: list[dict[str, object]] = []
    for item in items:
        slug = str(item.get("slug", "")).strip()
        if not slug:
            continue
        detail_url = str(item.get("detail_url", "")).strip() or f"{base}/skills/{slug}"
        row = dict(item)
        row["detail_url"] = detail_url
        if "repo_url" not in row:
            row["repo_url"] = ""
        out.append(row)
    return out


def _apply_search_filter_and_sort(
    *,
    items: list[dict[str, object]],
    query: str,
) -> list[dict[str, object]]:
    """Filter by keyword match first, then sort by requested priority."""
    terms = [t for t in re.split(r"\s+", query.strip().lower()) if t]
    if not terms:
        return []

    matched: list[tuple[int, dict[str, object]]] = []
    for row in items:
        slug = str(row.get("slug", "")).strip().lower()
        name = str(row.get("name", "")).strip().lower()
        summary = str(row.get("summary", "")).strip().lower()

        title_hit = all(term in f"{slug} {name}" for term in terms)
        summary_hit = all(term in summary for term in terms)
        if not title_hit and not summary_hit:
            continue
        rank = 0 if title_hit else 1
        matched.append((rank, row))

    matched.sort(
        key=lambda item: (
            item[0],
            -_to_int(item[1].get("stars", 0)),
            -_to_int(item[1].get("all_time_installs", 0)),
            str(item[1].get("slug", "")),
        )
    )
    return [row for _, row in matched]


def _is_empty_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _merge_result_rows(
    *,
    base_items: list[dict[str, object]],
    extra_items: list[dict[str, object]],
) -> None:
    if not base_items or not extra_items:
        return
    by_slug: dict[str, dict[str, object]] = {}
    for row in extra_items:
        slug = str(row.get("slug", "")).strip()
        if slug:
            by_slug[slug] = row
    if not by_slug:
        return
    for row in base_items:
        slug = str(row.get("slug", "")).strip()
        if not slug:
            continue
        extra = by_slug.get(slug)
        if not extra:
            continue
        for key in (
            "name",
            "summary",
            "version",
            "detail_url",
            "repo_url",
            "stars",
            "downloads",
            "current_installs",
            "all_time_installs",
        ):
            if _is_empty_value(row.get(key)) and not _is_empty_value(extra.get(key)):
                row[key] = extra[key]


def _fill_guessed_install_stats(items: list[dict[str, object]]) -> None:
    for row in items:
        if _to_int_or_none(row.get("all_time_installs")) is not None:
            continue
        name = str(row.get("name", "")).strip()
        summary = str(row.get("summary", "")).strip()
        guessed = _parse_number_from_name(name) or _parse_number_from_text(summary)
        if guessed > 0:
            row["all_time_installs"] = guessed


def _enrich_search_results_stats(
    *,
    results: list[dict[str, object]],
    search_limit: int,
    registry_url: str,
    workspace_dir: Path,
    api_token: str | None,
) -> None:
    if not results:
        return
    _enrich_results_with_cli_explore_stats(
        items=results,
        registry_url=registry_url,
        workspace_dir=workspace_dir,
        api_token=api_token,
    )
    _enrich_results_with_cli_inspect_stats(
        items=results,
        registry_url=registry_url,
        api_token=api_token,
        max_items=min(search_limit, 24),
    )
    _fill_guessed_install_stats(results)


def _to_int(value: object) -> int:
    try:
        if value is None:
            return 0
        if isinstance(value, bool):
            return int(value)
        return int(float(str(value)))
    except Exception:
        return 0


def _to_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return _to_int(value)
    raw = str(value).strip()
    if not raw:
        return None
    scale = 1.0
    suffix = raw[-1].lower()
    if suffix in {"k", "m", "b"}:
        raw = raw[:-1].strip()
        scale = {"k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}[suffix]
    raw = raw.replace(",", "")
    if raw.count(".") > 1:
        raw = raw.replace(".", "")
    try:
        return int(float(raw) * scale)
    except Exception:
        return None


def _pick_first_int(*values: object) -> int | None:
    for value in values:
        parsed = _to_int_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _walk_dict_values(value: object) -> list[tuple[list[str], object]]:
    out: list[tuple[list[str], object]] = []

    def _walk(node: object, path: list[str]) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str):
                    _walk(v, [*path, k])
            return
        if isinstance(node, list):
            for idx, v in enumerate(node):
                _walk(v, [*path, str(idx)])
            return
        out.append((path, node))

    _walk(value, [])
    return out


def _walk_dict_nodes(value: object) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            out.append(node)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(value)
    return out


def _find_int_in_tree(source: dict[str, object], keys: set[str]) -> int | None:
    for path, value in _walk_dict_values(source):
        if not path:
            continue
        last = _norm_key(path[-1])
        if last in keys:
            parsed = _to_int_or_none(value)
            if parsed is not None:
                return parsed
    return None


def _extract_stats_from_metric_nodes(source: dict[str, object]) -> dict[str, int | None]:
    out: dict[str, int | None] = {
        "stars": None,
        "downloads": None,
        "current_installs": None,
        "all_time_installs": None,
    }
    for node in _walk_dict_nodes(source):
        keys = {str(k): v for k, v in node.items() if isinstance(k, str)}
        if not keys:
            continue
        marker_parts: list[str] = []
        for marker_key in ("name", "key", "metric", "label", "title", "type"):
            marker_val = keys.get(marker_key)
            if isinstance(marker_val, str):
                marker_parts.append(marker_val.lower())
        marker = " ".join(marker_parts)
        if not marker:
            continue
        candidate = _pick_first_int(
            keys.get("value"),
            keys.get("count"),
            keys.get("total"),
            keys.get("current"),
            keys.get("allTime"),
            keys.get("all_time"),
            keys.get("installs"),
            keys.get("downloads"),
            keys.get("stars"),
        )
        if candidate is None:
            continue
        if "star" in marker and out["stars"] is None:
            out["stars"] = candidate
            continue
        if "download" in marker and out["downloads"] is None:
            out["downloads"] = candidate
            continue
        if "current" in marker and "install" in marker and out["current_installs"] is None:
            out["current_installs"] = candidate
            continue
        if ("alltime" in _norm_key(marker) or "all-time" in marker or "lifetime" in marker) and (
            "install" in marker
        ):
            if out["all_time_installs"] is None:
                out["all_time_installs"] = candidate
            continue
        if "install" in marker and out["all_time_installs"] is None:
            out["all_time_installs"] = candidate
    return out


def _extract_stats(source: dict[str, object]) -> dict[str, int | None]:
    stats_raw = source.get("stats")
    stats = stats_raw if isinstance(stats_raw, dict) else {}
    installs = _pick_first_int(
        source.get("installs"),
        stats.get("installs"),
        _find_int_in_tree(
            stats,
            {
                "installs",
                "installcount",
                "totalinstalls",
                "installstotal",
                "installsalltime",
            },
        ),
    )
    stars = _pick_first_int(
        source.get("stars"),
        source.get("starCount"),
        source.get("starsCount"),
        source.get("stars_count"),
        stats.get("stars"),
        stats.get("starCount"),
        stats.get("starsCount"),
        _find_int_in_tree(
            stats,
            {"stars", "starcount", "starscount", "ratingcount"},
        ),
    )
    downloads = _pick_first_int(
        source.get("downloads"),
        source.get("downloadCount"),
        source.get("downloadsCount"),
        source.get("allDownloads"),
        stats.get("downloads"),
        stats.get("downloadCount"),
        stats.get("downloadsCount"),
        stats.get("allDownloads"),
        _find_int_in_tree(
            stats,
            {"downloads", "downloadcount", "downloadsalltime", "alldownloads"},
        ),
    )
    current_installs = _pick_first_int(
        source.get("current_installs"),
        source.get("currentInstalls"),
        source.get("currentInstallCount"),
        source.get("currentInstall"),
        source.get("activeInstalls"),
        stats.get("current_installs"),
        stats.get("currentInstalls"),
        stats.get("currentInstallCount"),
        stats.get("currentInstall"),
        stats.get("activeInstalls"),
        _find_int_in_tree(
            stats,
            {
                "currentinstalls",
                "installscurrent",
                "currentinstallcount",
                "activeinstalls",
                "current",
            },
        ),
        installs,
    )
    all_time_installs = _pick_first_int(
        source.get("all_time_installs"),
        source.get("allTimeInstalls"),
        source.get("allTimeInstallCount"),
        source.get("installsTotal"),
        source.get("totalInstalls"),
        source.get("totalInstallCount"),
        source.get("lifetimeInstalls"),
        stats.get("all_time_installs"),
        stats.get("allTimeInstalls"),
        stats.get("allTimeInstallCount"),
        stats.get("installsTotal"),
        stats.get("totalInstalls"),
        stats.get("totalInstallCount"),
        stats.get("lifetimeInstalls"),
        _find_int_in_tree(
            stats,
            {
                "alltimeinstalls",
                "installsalltime",
                "alltimeinstallcount",
                "totalinstalls",
                "installstotal",
                "lifetimeinstalls",
                "alltime",
                "total",
            },
        ),
        installs,
    )
    node_stats = _extract_stats_from_metric_nodes(source)
    return {
        "stars": stars if stars is not None else node_stats["stars"],
        "downloads": downloads if downloads is not None else node_stats["downloads"],
        "current_installs": (
            current_installs if current_installs is not None else node_stats["current_installs"]
        ),
        "all_time_installs": (
            all_time_installs if all_time_installs is not None else node_stats["all_time_installs"]
        ),
    }


def _has_any_stats(item: dict[str, object]) -> bool:
    for key in ("stars", "downloads", "current_installs", "all_time_installs"):
        if _to_int_or_none(item.get(key)) is not None:
            return True
    return False


def _has_primary_stats(item: dict[str, object]) -> bool:
    stars = _to_int_or_none(item.get("stars"))
    downloads = _to_int_or_none(item.get("downloads"))
    return stars is not None and downloads is not None


def _extract_stats_from_html(html: str) -> dict[str, int | None]:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return {
            "stars": None,
            "downloads": None,
            "current_installs": None,
            "all_time_installs": None,
        }
    m = re.search(
        (
            r"([0-9][0-9.,kKmMbB]*)\s*[·•]\s*([0-9][0-9.,kKmMbB]*)\s*[·•]\s*"
            r"([0-9][0-9.,kKmMbB]*)\s*current installs\s*[·•]\s*"
            r"([0-9][0-9.,kKmMbB]*)\s*all-time installs"
        ),
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return {
            "stars": _to_int_or_none(m.group(1)),
            "downloads": _to_int_or_none(m.group(2)),
            "current_installs": _to_int_or_none(m.group(3)),
            "all_time_installs": _to_int_or_none(m.group(4)),
        }
    m_zh = re.search(
        r"([0-9][0-9.,kKmMbB]*)\s*[·•]\s*([0-9][0-9.,kKmMbB]*)\s*[·•]\s*([0-9][0-9.,kKmMbB]*)\s*当前安装\s*[·•]\s*([0-9][0-9.,kKmMbB]*)\s*总安装",
        text,
        flags=re.IGNORECASE,
    )
    if m_zh:
        return {
            "stars": _to_int_or_none(m_zh.group(1)),
            "downloads": _to_int_or_none(m_zh.group(2)),
            "current_installs": _to_int_or_none(m_zh.group(3)),
            "all_time_installs": _to_int_or_none(m_zh.group(4)),
        }
    return {
        "stars": None,
        "downloads": None,
        "current_installs": None,
        "all_time_installs": None,
    }


def _parse_number_from_name(name: str) -> int:
    """Extract trailing '(3.671)' style counts from display name."""
    m = re.search(r"\(([\d.,]+)\)\s*$", name.strip())
    if not m:
        return 0
    raw = m.group(1).strip()
    # Handle common locale thousands separators used by registry labels.
    if "." in raw and "," not in raw and raw.count(".") >= 1:
        raw = raw.replace(".", "")
    raw = raw.replace(",", "")
    return _to_int(raw)


def _parse_number_from_text(text: str) -> int:
    """Extract '(3.671)' style count from any position in text."""
    m = re.search(r"\(([\d.,]+)\)", text)
    if not m:
        return 0
    raw = m.group(1).strip()
    if "." in raw and "," not in raw and raw.count(".") >= 1:
        raw = raw.replace(".", "")
    raw = raw.replace(",", "")
    return _to_int(raw)


def search_skills(
    *,
    query: str,
    registry_url: str,
    workspace_dir: Path,
    api_token: str | None = None,
    limit: int = 100,
) -> list[dict[str, object]]:
    """Search ClawHub skills by query via CLI only."""
    if not is_clawhub_cli_available():
        raise ClawHubCliError("未检测到 clawhub CLI，无法执行搜索")
    search_limit = max(1, min(limit, 200))

    env = _build_env(
        registry_url=registry_url,
        workspace_dir=workspace_dir,
        api_token=api_token,
    )

    json_search_ok = False
    try:
        raw_json = _run(
            ["clawhub", "search", query, "--json", "--limit", str(search_limit)],
            env=env,
        )
        json_search_ok = True
        data = json.loads(raw_json)
        items = data if isinstance(data, list) else data.get("items", [])
        results: list[dict[str, object]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            slug = str(item.get("slug", "")).strip()
            if not slug:
                continue
            results.append(
                {
                    "slug": slug,
                    "name": str(item.get("name", slug)).strip() or slug,
                    "summary": str(item.get("summary", "")).strip(),
                    "version": str(item.get("version", "")).strip(),
                }
            )
        if results:
            _enrich_search_results_stats(
                results=results,
                search_limit=search_limit,
                registry_url=registry_url,
                workspace_dir=workspace_dir,
                api_token=api_token,
            )
            filtered_sorted = _apply_search_filter_and_sort(items=results, query=query)
            decorated = _decorate_results(items=filtered_sorted, registry_url=registry_url)
            return decorated[:limit]
        # CLI 执行成功但无结果：视为正常“0 条”
        return []
    except Exception:
        pass

    text_search_ok = False
    with contextlib.suppress(Exception):
        text = _run(
            ["clawhub", "search", query, "--limit", str(search_limit)],
            env=env,
        )
        text_search_ok = True
        parsed = _parse_search_text(text)
        if parsed:
            _enrich_search_results_stats(
                results=parsed,
                search_limit=search_limit,
                registry_url=registry_url,
                workspace_dir=workspace_dir,
                api_token=api_token,
            )
            filtered_sorted = _apply_search_filter_and_sort(items=parsed, query=query)
            return _decorate_results(items=filtered_sorted, registry_url=registry_url)[:limit]
        # CLI 执行成功但文本为空/无法解析：按“无结果”处理
        return []
    if json_search_ok or text_search_ok:
        return []
    raise ClawHubCliError("CLI 搜索失败，请重试或检查 clawhub 登录状态")


def _enrich_results_with_cli_explore_stats(
    *,
    items: list[dict[str, object]],
    registry_url: str,
    workspace_dir: Path,
    api_token: str | None,
) -> None:
    if not items or not is_clawhub_cli_available():
        return
    env = _build_env(
        registry_url=registry_url,
        workspace_dir=workspace_dir,
        api_token=api_token,
    )
    try:
        raw = _run(
            ["clawhub", "explore", "--json", "--limit", "200", "--sort", "installsAllTime"],
            env=env,
        )
        payload = json.loads(raw)
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        return
    by_slug: dict[str, dict[str, int | None]] = {}
    for entry in raw_items:
        if not isinstance(entry, dict):
            continue
        slug = str(entry.get("slug", "")).strip()
        if not slug:
            continue
        by_slug[slug] = _extract_stats(entry)
    for row in items:
        slug = str(row.get("slug", "")).strip()
        if not slug:
            continue
        stats = by_slug.get(slug)
        if not stats:
            continue
        for key, value in stats.items():
            if value is not None:
                row[key] = value


def _enrich_results_with_cli_inspect_stats(
    *,
    items: list[dict[str, object]],
    registry_url: str,
    api_token: str | None,
    max_items: int,
) -> None:
    if not items or max_items <= 0 or not is_clawhub_cli_available():
        return
    checked = 0
    for row in items:
        if checked >= max_items:
            break
        checked += 1
        if _has_primary_stats(row):
            continue
        slug = str(row.get("slug", "")).strip()
        if not slug:
            continue
        stats = _inspect_skill_stats_via_cli(
            slug=slug,
            registry_url=registry_url,
            api_token=api_token,
        )
        for key, value in stats.items():
            if value is not None:
                row[key] = value


def _enrich_results_with_http_skill_list(
    *,
    items: list[dict[str, object]],
    registry_url: str,
    api_token: str | None,
) -> None:
    if not items:
        return
    base = registry_url.rstrip("/") + "/"
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    target_slugs = {
        str(x.get("slug", "")).strip()
        for x in items
        if str(x.get("slug", "")).strip()
    }
    if not target_slugs:
        return
    found: dict[str, dict[str, int | None]] = {}
    cursor: str | None = None
    scanned_pages = 0
    with httpx.Client(timeout=12, follow_redirects=True) as client:
        while scanned_pages < 10 and len(found) < len(target_slugs):
            params: dict[str, str | int] = {"limit": 100, "sort": "installsAllTime"}
            if cursor:
                params["cursor"] = cursor
            resp = client.get(f"{base}api/v1/skills", params=params, headers=headers)
            if resp.status_code >= 400:
                break
            payload = resp.json()
            if not isinstance(payload, dict):
                break
            raw_items = payload.get("items")
            if not isinstance(raw_items, list):
                break
            for entry in raw_items:
                if not isinstance(entry, dict):
                    continue
                slug = str(entry.get("slug", "")).strip()
                if not slug or slug not in target_slugs:
                    continue
                found[slug] = _extract_stats(entry)
            next_cursor = payload.get("nextCursor")
            cursor = str(next_cursor).strip() if next_cursor is not None else ""
            scanned_pages += 1
            if not cursor:
                break
    if not found:
        return
    for row in items:
        slug = str(row.get("slug", "")).strip()
        if not slug:
            continue
        stats = found.get(slug)
        if not stats:
            continue
        for key, value in stats.items():
            if value is not None:
                row[key] = value


def install_skill(
    *,
    slug: str,
    version: str | None,
    registry_url: str,
    workspace_dir: Path,
    install_dir: Path,
    api_token: str | None = None,
) -> str:
    """Install a ClawHub skill by slug (CLI first, HTTP fallback)."""
    _validate_slug(slug)
    if not is_clawhub_cli_available():
        _install_via_http(
            slug=slug,
            version=version,
            registry_url=registry_url,
            install_dir=install_dir,
            api_token=api_token,
        )
        return f"installed via http: {slug}"

    env = _build_env(
        registry_url=registry_url,
        workspace_dir=workspace_dir,
        api_token=api_token,
    )

    base_args = ["clawhub", "install", slug]
    if version:
        base_args.extend(["--version", version])

    attempts = [
        [*base_args, "--dir", str(install_dir), "--workdir", str(workspace_dir)],
        [*base_args, "--workdir", str(workspace_dir)],
        base_args,
    ]

    last_error = "未知错误"
    for args in attempts:
        try:
            return _run(args, env=env)
        except ClawHubCliError as exc:
            last_error = str(exc)

    _install_via_http(
        slug=slug,
        version=version,
        registry_url=registry_url,
        install_dir=install_dir,
        api_token=api_token,
    )
    return f"installed via http fallback: {slug} (cli_error={last_error})"


def publish_installed_skill(
    *,
    skill_dir: Path,
    skill_slug: str,
    skill_version: str | None,
    registry_url: str,
    workspace_dir: Path,
    api_token: str | None = None,
) -> str:
    """Publish one local installed skill to ClawHub via CLI publish."""
    if not is_clawhub_cli_available():
        raise ClawHubCliError("未检测到 clawhub CLI，无法发布技能")
    target = skill_dir.resolve()
    if not target.is_dir() or not (target / "SKILL.md").is_file():
        raise ClawHubCliError(f"无效技能目录: {target}")
    raw_slug = skill_slug.strip()
    _validate_slug(raw_slug)
    version = _normalize_publish_version(skill_version) or _resolve_publish_version(target)

    env = _build_env(
        registry_url=registry_url,
        workspace_dir=workspace_dir,
        api_token=api_token,
    )

    try:
        return _run(
            [
                "clawhub",
                "publish",
                str(target),
                "--slug",
                raw_slug,
                "--version",
                version,
                "--tags",
                "latest",
            ],
            env=env,
        )
    except ClawHubCliError as exc:
        msg = _clean_cli_error(str(exc))
        if "Only the owner can publish updates" in msg:
            raise ClawHubCliError(
                "发布被拒绝：该技能 slug 已存在且你不是所有者，无法覆盖更新。"
                "请改用新的技能 ID（slug）发布，或切换到该技能所有者账号。"
            ) from exc
        if "--version must be valid semver" in msg:
            raise ClawHubCliError(
                "发布失败：版本号不合法。请在 SKILL.md 的 frontmatter 中使用 "
                "version: x.y.z（例如 0.1.0）。"
            ) from exc
        raise ClawHubCliError(msg) from exc


def _resolve_publish_version(skill_dir: Path) -> str:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return "0.1.0"
    with contextlib.suppress(Exception):
        raw = skill_md.read_text(encoding="utf-8")
        fm_match = _FRONTMATTER_RE.match(raw)
        if not fm_match:
            return "0.1.0"
        frontmatter = fm_match.group(1)
        m = re.search(r"(?mi)^\s*version\s*:\s*['\"]?([^\n\"']+)['\"]?\s*$", frontmatter)
        if not m:
            return "0.1.0"
        ver = m.group(1).strip()
        if ver.lower().startswith("v"):
            ver = ver[1:].strip()
        if _SEMVER_RE.match(ver):
            return ver
    return "0.1.0"


def _normalize_publish_version(version: str | None) -> str | None:
    if version is None:
        return None
    raw = version.strip()
    if not raw:
        return None
    if raw.lower().startswith("v"):
        raw = raw[1:].strip()
    if _SEMVER_RE.match(raw):
        return raw
    raise ClawHubCliError(
        "发布失败：版本号不合法。请输入 x.y.z（例如 0.1.0）。"
    )


def _validate_slug(slug: str) -> None:
    raw = slug.strip()
    if not raw:
        raise ClawHubCliError("slug 不能为空")
    if "/" in raw or "\\" in raw or ".." in raw:
        raise ClawHubCliError(f"非法 slug: {raw}")


def _search_via_http(
    *,
    query: str,
    registry_url: str,
    api_token: str | None,
    limit: int = 100,
) -> list[dict[str, object]]:
    base = registry_url.rstrip("/") + "/"
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    with httpx.Client(timeout=12, follow_redirects=True) as client:
        resp = client.get(
            f"{base}api/v1/search",
            params={"q": query, "limit": max(1, min(limit, 200))},
            headers=headers,
        )
        if resp.status_code >= 400:
            raise ClawHubCliError(f"搜索失败: HTTP {resp.status_code} {resp.text[:200]}")
        data = resp.json()
    raw_results = data.get("results", [])
    results: list[dict[str, object]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug", "")).strip()
        if not slug:
            continue
        name = str(item.get("displayName", slug)).strip() or slug
        summary = str(item.get("summary", "")).strip()
        guessed_total = _parse_number_from_name(name) or _parse_number_from_text(summary)
        stats = _extract_stats(item)
        if stats["all_time_installs"] is None and guessed_total > 0:
            stats["all_time_installs"] = guessed_total
        results.append(
            {
                "slug": slug,
                "name": name,
                "summary": summary,
                "version": str(item.get("version", "")).strip(),
                "stars": stats["stars"],
                "downloads": stats["downloads"],
                "current_installs": stats["current_installs"],
                "all_time_installs": stats["all_time_installs"],
                "detail_url": str(
                    item.get("url", "")
                    or item.get("detailUrl", "")
                    or item.get("pageUrl", "")
                    or item.get("permalink", "")
                ).strip(),
                "repo_url": str(
                    item.get("repoUrl", "")
                    or item.get("repository", "")
                    or item.get("repo", "")
                ).strip(),
            }
        )
    _enrich_results_with_skill_details(
        items=results,
        registry_url=registry_url,
        api_token=api_token,
        max_items=min(limit, 8),
    )
    return _decorate_results(items=results, registry_url=registry_url)


def _enrich_results_with_skill_details(
    *,
    items: list[dict[str, object]],
    registry_url: str,
    api_token: str | None,
    max_items: int,
) -> None:
    if not items or max_items <= 0:
        return
    base = registry_url.rstrip("/") + "/"
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    checked = 0
    with httpx.Client(timeout=12, follow_redirects=True) as client:
        for row in items:
            if checked >= max_items:
                break
            checked += 1
            slug = str(row.get("slug", "")).strip()
            if not slug or _has_any_stats(row):
                continue
            try:
                resp = client.get(
                    f"{base}api/v1/skills/{quote(slug, safe='')}",
                    headers=headers,
                )
                if resp.status_code >= 400:
                    continue
                payload = resp.json()
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            skill = payload.get("skill")
            if isinstance(skill, dict):
                stats = _extract_stats(skill)
                for key, value in stats.items():
                    if value is not None:
                        row[key] = value
            if not row.get("version"):
                latest = payload.get("latestVersion")
                if isinstance(latest, dict):
                    ver = str(latest.get("version", "")).strip()
                    if ver:
                        row["version"] = ver
            if _has_any_stats(row):
                continue
            cli_stats = _inspect_skill_stats_via_cli(
                slug=slug,
                registry_url=registry_url,
                api_token=api_token,
            )
            for key, value in cli_stats.items():
                if value is not None:
                    row[key] = value
            if _has_any_stats(row):
                continue
            detail_url = str(row.get("detail_url", "")).strip()
            if not detail_url:
                detail_url = f"{base}skills/{quote(slug, safe='')}"
            try:
                page_resp = client.get(
                    detail_url,
                    headers={"Accept": "text/html,*/*"},
                )
                if page_resp.status_code < 400:
                    page_stats = _extract_stats_from_html(page_resp.text)
                    for key, value in page_stats.items():
                        if value is not None:
                            row[key] = value
            except Exception:
                continue


def _inspect_skill_stats_via_cli(
    *,
    slug: str,
    registry_url: str,
    api_token: str | None,
) -> dict[str, int | None]:
    if not is_clawhub_cli_available():
        return {
            "stars": None,
            "downloads": None,
            "current_installs": None,
            "all_time_installs": None,
        }
    env = _build_env(
        registry_url=registry_url,
        workspace_dir=Path.cwd(),
        api_token=api_token,
    )
    try:
        raw = _run(["clawhub", "inspect", slug, "--json"], env=env)
        payload = json.loads(raw)
    except Exception:
        return {
            "stars": None,
            "downloads": None,
            "current_installs": None,
            "all_time_installs": None,
        }
    if not isinstance(payload, dict):
        return {
            "stars": None,
            "downloads": None,
            "current_installs": None,
            "all_time_installs": None,
        }
    skill = payload.get("skill")
    if not isinstance(skill, dict):
        return {
            "stars": None,
            "downloads": None,
            "current_installs": None,
            "all_time_installs": None,
        }
    return _extract_stats(skill)


def _install_via_http(
    *,
    slug: str,
    version: str | None,
    registry_url: str,
    install_dir: Path,
    api_token: str | None,
) -> None:
    base = registry_url.rstrip("/") + "/"
    headers: dict[str, str] = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    with httpx.Client(timeout=25, follow_redirects=True) as client:
        params = {"slug": slug}
        if version:
            params["version"] = version
        resp = client.get(f"{base}api/v1/download", params=params, headers=headers)
        if resp.status_code >= 400:
            raise ClawHubCliError(f"下载失败: HTTP {resp.status_code} {resp.text[:200]}")
        payload = resp.content

    install_dir.mkdir(parents=True, exist_ok=True)
    target = install_dir / slug
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(BytesIO(payload)) as zf:
            for member in zf.infolist():
                name = member.filename
                if not name or name.endswith("/"):
                    continue
                rel = Path(name)
                parts = rel.parts
                if any(p in {"", ".", ".."} for p in parts):
                    continue
                # remove optional top-level directory in archive
                rel2 = Path(*parts[1:]) if len(parts) > 1 else rel
                if not rel2.parts:
                    continue
                dst = target / rel2
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src, dst.open("wb") as out:
                    out.write(src.read())
    except zipfile.BadZipFile as exc:
        raise ClawHubCliError("下载内容不是有效 ZIP 包") from exc

    skill_md = target / "SKILL.md"
    if not skill_md.is_file():
        raise ClawHubCliError("安装失败: 压缩包中未找到 SKILL.md")
