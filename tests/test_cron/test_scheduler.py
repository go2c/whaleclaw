"""Tests for CronScheduler with at/cron/every schedule support."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from whaleclaw.cron.scheduler import (
    CronAction,
    CronJob,
    CronScheduler,
    Schedule,
    _parse_cron_field,
)


@pytest.fixture
def scheduler() -> CronScheduler:
    return CronScheduler()


@pytest.fixture
def sample_job() -> CronJob:
    return CronJob(
        id="job-1",
        name="Test",
        schedule="30 14 * * *",
        action=CronAction(type="message", target="user", payload={}),
        enabled=True,
        created_at=datetime(2025, 2, 22, 12, 0, 0),
    )


# ── Basic CRUD ──

@pytest.mark.asyncio
async def test_add_and_list_jobs(scheduler: CronScheduler, sample_job: CronJob) -> None:
    await scheduler.add_job(sample_job)
    jobs = await scheduler.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == "job-1"
    assert jobs[0].name == "Test"


@pytest.mark.asyncio
async def test_remove_job(scheduler: CronScheduler, sample_job: CronJob) -> None:
    await scheduler.add_job(sample_job)
    await scheduler.remove_job("job-1")
    jobs = await scheduler.list_jobs()
    assert len(jobs) == 0


# ── Legacy cron schedule (5-field string) ──

@pytest.mark.asyncio
async def test_should_run_legacy_cron(scheduler: CronScheduler) -> None:
    job = CronJob(
        id="m",
        name="M",
        schedule="32 14 22 2 5",
        action=CronAction(type="message", target="x", payload={}),
        enabled=True,
        created_at=datetime(2025, 2, 22),
    )
    now = datetime(2025, 2, 22, 14, 32, 0)
    assert now.weekday() == 5
    assert scheduler._should_run(job, now) is True


@pytest.mark.asyncio
async def test_should_not_run_legacy_cron(scheduler: CronScheduler) -> None:
    job = CronJob(
        id="m",
        name="M",
        schedule="35 14 22 2 5",
        action=CronAction(type="message", target="x", payload={}),
        enabled=True,
        created_at=datetime(2025, 2, 22),
    )
    now = datetime(2025, 2, 22, 14, 32, 0)
    assert scheduler._should_run(job, now) is False


# ── _parse_cron_field: extended syntax ──

class TestParseCronField:
    def test_star(self) -> None:
        assert _parse_cron_field("*", 0, 59, 42) is True

    def test_exact_match(self) -> None:
        assert _parse_cron_field("30", 0, 59, 30) is True
        assert _parse_cron_field("30", 0, 59, 31) is False

    def test_step(self) -> None:
        assert _parse_cron_field("*/5", 0, 59, 0) is True
        assert _parse_cron_field("*/5", 0, 59, 15) is True
        assert _parse_cron_field("*/5", 0, 59, 17) is False

    def test_step_with_start(self) -> None:
        assert _parse_cron_field("10/15", 0, 59, 10) is True
        assert _parse_cron_field("10/15", 0, 59, 25) is True
        assert _parse_cron_field("10/15", 0, 59, 20) is False

    def test_range(self) -> None:
        assert _parse_cron_field("1-5", 0, 6, 3) is True
        assert _parse_cron_field("1-5", 0, 6, 0) is False
        assert _parse_cron_field("1-5", 0, 6, 6) is False

    def test_comma_list(self) -> None:
        assert _parse_cron_field("0,15,30,45", 0, 59, 15) is True
        assert _parse_cron_field("0,15,30,45", 0, 59, 16) is False

    def test_mixed(self) -> None:
        assert _parse_cron_field("0,30", 0, 59, 30) is True
        assert _parse_cron_field("1-3,7", 0, 10, 7) is True
        assert _parse_cron_field("1-3,7", 0, 10, 5) is False


# ── Schedule kind: "at" (one-shot) ──

class TestScheduleAt:
    @pytest.mark.asyncio
    async def test_at_fires_when_due(self, scheduler: CronScheduler) -> None:
        target = datetime(2026, 2, 23, 10, 0, 0)
        job = CronJob(
            id="at-1",
            name="One-shot",
            schedule_obj=Schedule(kind="at", at=target),
            action=CronAction(type="message", target="s1", payload={"text": "hi"}),
            enabled=True,
            created_at=datetime(2026, 2, 23, 9, 50, 0),
            one_shot=True,
        )
        assert scheduler._should_run(job, datetime(2026, 2, 23, 9, 59, 0)) is False
        assert scheduler._should_run(job, datetime(2026, 2, 23, 10, 0, 0)) is True
        assert scheduler._should_run(job, datetime(2026, 2, 23, 10, 5, 0)) is True

    @pytest.mark.asyncio
    async def test_at_does_not_refire_after_trigger(self, scheduler: CronScheduler) -> None:
        target = datetime(2026, 2, 23, 10, 0, 0)
        job = CronJob(
            id="at-2",
            name="One-shot fired",
            schedule_obj=Schedule(kind="at", at=target),
            action=CronAction(type="message", target="s1", payload={}),
            enabled=True,
            created_at=datetime(2026, 2, 23, 9, 50, 0),
            last_run=datetime(2026, 2, 23, 10, 0, 30),
            one_shot=True,
        )
        assert scheduler._should_run(job, datetime(2026, 2, 23, 10, 1, 0)) is False

    @pytest.mark.asyncio
    async def test_one_shot_auto_deletes(self, scheduler: CronScheduler) -> None:
        job = CronJob(
            id="at-del",
            name="Delete me",
            schedule_obj=Schedule(kind="at", at=datetime(2026, 1, 1)),
            action=CronAction(type="message", target="s1", payload={}),
            enabled=True,
            created_at=datetime(2026, 1, 1),
            one_shot=True,
        )
        await scheduler.add_job(job)
        await scheduler.trigger_job("at-del")
        assert len(await scheduler.list_jobs()) == 0


# ── Schedule kind: "every" (interval) ──

class TestScheduleEvery:
    @pytest.mark.asyncio
    async def test_every_first_run(self, scheduler: CronScheduler) -> None:
        job = CronJob(
            id="ev-1",
            name="Every 5min",
            schedule_obj=Schedule(kind="every", every_seconds=300),
            action=CronAction(type="message", target="s1", payload={}),
            enabled=True,
            created_at=datetime(2026, 2, 23, 9, 0, 0),
        )
        assert scheduler._should_run(job, datetime(2026, 2, 23, 9, 0, 0)) is True

    @pytest.mark.asyncio
    async def test_every_respects_interval(self, scheduler: CronScheduler) -> None:
        base = datetime(2026, 2, 23, 9, 0, 0)
        job = CronJob(
            id="ev-2",
            name="Every 10min",
            schedule_obj=Schedule(kind="every", every_seconds=600),
            action=CronAction(type="message", target="s1", payload={}),
            enabled=True,
            created_at=base,
            last_run=base,
        )
        assert scheduler._should_run(job, base + timedelta(minutes=5)) is False
        assert scheduler._should_run(job, base + timedelta(minutes=10)) is True
        assert scheduler._should_run(job, base + timedelta(minutes=15)) is True


# ── Schedule kind: "cron" (expression via schedule_obj) ──

class TestScheduleCron:
    @pytest.mark.asyncio
    async def test_cron_obj_matches(self, scheduler: CronScheduler) -> None:
        job = CronJob(
            id="cron-1",
            name="Every day 7am",
            schedule_obj=Schedule(kind="cron", expr="0 7 * * *"),
            action=CronAction(type="message", target="s1", payload={}),
            enabled=True,
            created_at=datetime(2026, 2, 23),
        )
        assert scheduler._should_run(job, datetime(2026, 2, 23, 7, 0, 0)) is True
        assert scheduler._should_run(job, datetime(2026, 2, 23, 7, 1, 0)) is False
        assert scheduler._should_run(job, datetime(2026, 2, 24, 7, 0, 0)) is True

    @pytest.mark.asyncio
    async def test_cron_obj_step_expr(self, scheduler: CronScheduler) -> None:
        job = CronJob(
            id="cron-2",
            name="Every 15 min",
            schedule_obj=Schedule(kind="cron", expr="*/15 * * * *"),
            action=CronAction(type="message", target="s1", payload={}),
            enabled=True,
            created_at=datetime(2026, 2, 23),
        )
        assert scheduler._should_run(job, datetime(2026, 2, 23, 8, 0, 0)) is True
        assert scheduler._should_run(job, datetime(2026, 2, 23, 8, 15, 0)) is True
        assert scheduler._should_run(job, datetime(2026, 2, 23, 8, 7, 0)) is False

    @pytest.mark.asyncio
    async def test_disabled_job_never_runs(self, scheduler: CronScheduler) -> None:
        job = CronJob(
            id="cron-dis",
            name="Disabled",
            schedule_obj=Schedule(kind="cron", expr="* * * * *"),
            action=CronAction(type="message", target="s1", payload={}),
            enabled=False,
            created_at=datetime(2026, 2, 23),
        )
        assert scheduler._should_run(job, datetime(2026, 2, 23, 12, 0, 0)) is False


# ── on_fire callback ──

@pytest.mark.asyncio
async def test_on_fire_callback() -> None:
    fired: list[tuple[str, str]] = []

    async def _cb(job_id: str, action: CronAction) -> None:
        fired.append((job_id, action.payload.get("text", "")))

    scheduler = CronScheduler(on_fire=_cb)
    job = CronJob(
        id="cb-1",
        name="Callback test",
        schedule_obj=Schedule(kind="at", at=datetime(2026, 1, 1)),
        action=CronAction(type="message", target="s1", payload={"text": "hello"}),
        enabled=True,
        created_at=datetime(2026, 1, 1),
        one_shot=True,
    )
    await scheduler.add_job(job)
    await scheduler.trigger_job("cb-1")
    assert fired == [("cb-1", "hello")]
    assert len(await scheduler.list_jobs()) == 0
