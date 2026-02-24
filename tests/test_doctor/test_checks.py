"""Tests for doctor checks and runner."""

from __future__ import annotations

import pytest

from whaleclaw.doctor.checks import (
    CheckResult,
    DatabaseCheck,
    DiskSpaceCheck,
    PythonVersionCheck,
)
from whaleclaw.doctor.runner import Doctor


@pytest.mark.asyncio
async def test_python_version_check() -> None:
    """PythonVersionCheck returns ok on 3.12+."""
    check = PythonVersionCheck()
    result = await check.check()
    assert result.status == "ok"
    assert "3." in result.message


@pytest.mark.asyncio
async def test_disk_space_check() -> None:
    """DiskSpaceCheck returns ok when disk has space."""
    check = DiskSpaceCheck()
    result = await check.check()
    assert result.status == "ok"
    assert isinstance(result, CheckResult)
    assert result.name == "磁盘空间"


@pytest.mark.asyncio
async def test_doctor_run_all() -> None:
    """Doctor run_all returns list of CheckResult."""
    doctor = Doctor()
    results = await doctor.run_all()
    assert isinstance(results, list)
    assert len(results) >= 5
    for r in results:
        assert isinstance(r, CheckResult)
        assert r.name
        assert r.status in ("ok", "warning", "error")
        assert r.message


def test_format_report() -> None:
    """format_report produces string with status counts."""
    results = [
        CheckResult(name="A", status="ok", message="OK"),
        CheckResult(name="B", status="warning", message="Warn"),
        CheckResult(name="C", status="error", message="Err"),
    ]
    doctor = Doctor()
    report = doctor.format_report(results)
    assert "WhaleClaw Doctor" in report
    assert "✅" in report
    assert "⚠️" in report
    assert "❌" in report
    assert "通过" in report or "1" in report
    assert "警告" in report or "2" in report
    assert "错误" in report or "3" in report


@pytest.mark.asyncio
async def test_port_check_returns_result() -> None:
    """PortCheck returns a CheckResult."""
    from whaleclaw.doctor.checks import PortCheck

    check = PortCheck(port=18666)
    result = await check.check()
    assert isinstance(result, CheckResult)
    assert result.name == "Gateway 端口"
    assert result.status in ("ok", "error")


@pytest.mark.asyncio
async def test_database_check() -> None:
    """DatabaseCheck returns a CheckResult."""
    check = DatabaseCheck()
    result = await check.check()
    assert isinstance(result, CheckResult)
    assert result.name == "会话数据库"
    assert result.status in ("ok", "warning")
