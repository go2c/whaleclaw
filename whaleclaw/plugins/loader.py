"""Plugin discovery and loading."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

from pydantic import BaseModel

from whaleclaw.config.paths import PLUGINS_DIR
from whaleclaw.plugins.sdk import WhaleclawPlugin
from whaleclaw.types import ConfigError

PLUGIN_MANIFEST = "whaleclaw_plugin.json"
ENTRY_GROUP = "whaleclaw.plugins"
_BUILTIN_PLUGINS: tuple[tuple[str, str], ...] = (
    ("evomap", "whaleclaw.plugins.evomap.plugin:plugin"),
)


class PluginMeta(BaseModel):
    """Plugin metadata from manifest or entry point."""

    id: str
    name: str
    description: str = ""
    version: str = "0.0.0"
    author: str = ""
    entry: str
    path: str = ""


class PluginLoader:
    """Discovers and loads WhaleClaw plugins."""

    def __init__(
        self,
        user_plugins_dir: Path | None = None,
        local_plugins_dir: Path | None = None,
    ) -> None:
        self._user_dir = user_plugins_dir or PLUGINS_DIR
        self._local_dir = local_plugins_dir or Path.cwd() / "plugins"
        self._discovered: dict[str, PluginMeta] = {}

    def discover(self) -> list[PluginMeta]:
        """Scan plugin paths and entry points, return metadata list."""
        seen_ids: set[str] = set()
        result: list[PluginMeta] = []

        for meta in self._discover_dir(self._user_dir):
            if meta.id not in seen_ids:
                seen_ids.add(meta.id)
                result.append(meta)

        if self._local_dir.is_dir():
            for meta in self._discover_dir(self._local_dir):
                if meta.id not in seen_ids:
                    seen_ids.add(meta.id)
                    result.append(meta)

        for meta in self._discover_entry_points():
            if meta.id not in seen_ids:
                seen_ids.add(meta.id)
                result.append(meta)

        for plugin_id, entry in _BUILTIN_PLUGINS:
            if plugin_id in seen_ids:
                continue
            result.append(
                PluginMeta(
                    id=plugin_id,
                    name=plugin_id,
                    description="",
                    version="0.0.0",
                    author="",
                    entry=entry,
                    path="",
                )
            )
            seen_ids.add(plugin_id)

        self._discovered = {m.id: m for m in result}
        return result

    def _discover_dir(self, root: Path) -> list[PluginMeta]:
        metas: list[PluginMeta] = []
        if not root.is_dir():
            return metas
        for sub in root.iterdir():
            if sub.is_dir():
                manifest = sub / PLUGIN_MANIFEST
                if manifest.is_file():
                    try:
                        meta = self._parse_manifest(manifest, sub)
                        metas.append(meta)
                    except (json.JSONDecodeError, KeyError) as exc:
                        raise ConfigError(f"插件清单解析失败: {manifest} — {exc}") from exc
        return metas

    def _parse_manifest(self, path: Path, plugin_dir: Path) -> PluginMeta:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ConfigError(f"插件清单格式错误 (期望 JSON 对象): {path}")
        required = ("id", "name", "entry")
        for k in required:
            if k not in data:
                raise ConfigError(f"插件清单缺少必填项: {k} — {path}")
        return PluginMeta(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "0.0.0"),
            author=data.get("author", ""),
            entry=data["entry"],
            path=str(plugin_dir),
        )

    def _discover_entry_points(self) -> list[PluginMeta]:
        metas: list[PluginMeta] = []
        try:
            eps = importlib.metadata.entry_points(group=ENTRY_GROUP)
        except TypeError:
            eps = importlib.metadata.entry_points().get(ENTRY_GROUP, [])
        for ep in eps:
            metas.append(
                PluginMeta(
                    id=ep.name,
                    name=ep.name,
                    description="",
                    version="0.0.0",
                    author="",
                    entry=ep.value,
                    path="",
                )
            )
        return metas

    def load(self, plugin_id: str) -> WhaleclawPlugin:
        """Load and instantiate plugin by id."""
        meta = self._discovered.get(plugin_id)
        if not meta:
            self.discover()
            meta = self._discovered.get(plugin_id)
        if not meta:
            raise ConfigError(f"插件未发现: {plugin_id}")

        if meta.path:
            return self._load_from_dir(meta)
        return self._load_from_entry_point(meta)

    def _load_from_dir(self, meta: PluginMeta) -> WhaleclawPlugin:
        path = Path(meta.path)
        entry = meta.entry
        if ":" in entry:
            mod_part, attr = entry.split(":", 1)
        else:
            mod_part, attr = entry, "plugin"
        file_path = path / (mod_part if mod_part.endswith(".py") else f"{mod_part}.py")

        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

        spec = importlib.util.spec_from_file_location(
            f"whaleclaw_plugin_{meta.id}",
            file_path,
        )
        if not spec or not spec.loader:
            raise ConfigError(f"无法加载插件模块: {meta.id} — {meta.entry}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        plugin = getattr(mod, attr, None)
        if plugin is None:
            raise ConfigError(f"插件模块缺少 '{attr}' 属性: {meta.id}")
        return plugin

    def _load_from_entry_point(self, meta: PluginMeta) -> WhaleclawPlugin:
        entry_value = meta.entry
        if ":" not in entry_value:
            raise ConfigError(f"Entry point 格式错误 (需 module:attr): {meta.id}")
        mod_name, attr = entry_value.split(":", 1)
        mod = importlib.import_module(mod_name)
        plugin = getattr(mod, attr, None)
        if plugin is None:
            raise ConfigError(f"Entry point 未找到 '{attr}': {meta.id}")
        if callable(plugin):
            plugin = plugin()
        return plugin

    def load_all(self) -> list[WhaleclawPlugin]:
        """Load all discovered plugins."""
        if not self._discovered:
            self.discover()
        return [self.load(pid) for pid in self._discovered]
