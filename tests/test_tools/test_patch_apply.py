"""Tests for the patch apply tool."""

from __future__ import annotations

import pytest

from whaleclaw.tools.patch_apply import PatchApplyTool


@pytest.fixture()
def tool() -> PatchApplyTool:
    return PatchApplyTool()


def _sample_patch() -> str:
    return (
        "--- a/demo.txt\n"
        "+++ b/demo.txt\n"
        "@@ -1,2 +1,2 @@\n"
        " hello\n"
        "-world\n"
        "+earth\n"
    )


@pytest.mark.asyncio
async def test_patch_apply_success(tool: PatchApplyTool, tmp_path) -> None:  # noqa: ANN001
    f = tmp_path / "demo.txt"
    f.write_text("hello\nworld\n", encoding="utf-8")

    result = await tool.execute(
        patch=_sample_patch(),
        cwd=str(tmp_path),
        strip=1,
    )

    assert result.success
    assert "earth" in f.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_patch_apply_check_only(tool: PatchApplyTool, tmp_path) -> None:  # noqa: ANN001
    f = tmp_path / "demo.txt"
    f.write_text("hello\nworld\n", encoding="utf-8")

    result = await tool.execute(
        patch=_sample_patch(),
        cwd=str(tmp_path),
        strip=1,
        check=True,
    )

    assert result.success
    assert f.read_text(encoding="utf-8") == "hello\nworld\n"


@pytest.mark.asyncio
async def test_patch_apply_invalid_patch(tool: PatchApplyTool, tmp_path) -> None:  # noqa: ANN001
    result = await tool.execute(
        patch="not a diff",
        cwd=str(tmp_path),
    )
    assert not result.success
    assert result.error == "补丁应用失败"
