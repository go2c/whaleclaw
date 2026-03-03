"""Tests for SkillParser."""

from __future__ import annotations

from whaleclaw.skills.parser import SkillParser


def test_parse_frontmatter_and_fields(tmp_path) -> None:
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        """---
triggers: ["浏览器", "打开网页", "screenshot"]
max_tokens: 800
lock_session: true
param_guard:
  enabled: true
  params:
    - key: api_key
      label: API Key
      type: api_key
      required: true
      prompt: 请提供 API Key
---
# 浏览器控制

## 触发条件
用户请求打开网页、截图。

## 指令
使用 browser 工具操作网页。

## 工具
- browser

## 示例
用户: 打开 example.com
Agent: [navigate] -> [screenshot]
""",
        encoding="utf-8",
    )

    parser = SkillParser()
    skill = parser.parse(skill_md)

    assert skill.triggers == ["浏览器", "打开网页", "screenshot"]
    assert skill.max_tokens == 800
    assert skill.lock_session is True
    assert skill.param_guard is not None
    assert skill.param_guard.enabled is True
    assert len(skill.param_guard.params) == 1
    assert skill.param_guard.params[0].key == "api_key"
    assert skill.name == "浏览器控制"
    assert "browser" in skill.tools
    assert skill.trigger_description
    assert "使用 browser" in skill.instructions


def test_parse_infers_param_guard_when_not_declared(tmp_path) -> None:
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        """---
triggers: ["生图", "图生图"]
---
# 某生图技能

## 指令
文生图最小必填：提示词。
图生图最小必填：提示词 + 图片。
可选：比例/尺寸。
""",
        encoding="utf-8",
    )

    parser = SkillParser()
    skill = parser.parse(skill_md)

    assert skill.param_guard is not None
    keys = [p.key for p in skill.param_guard.params]
    assert "prompt" in keys
    assert "images" in keys
    assert "ratio" in keys
