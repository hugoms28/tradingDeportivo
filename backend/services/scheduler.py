# -*- coding: utf-8 -*-
"""
APScheduler cron for automatic model execution.
Default: Fridays at 08:00 → run predictions for all 5 leagues.
"""
import asyncio
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["scheduler"])

# Module-level scheduler instance
_scheduler: AsyncIOScheduler | None = None
_enabled: bool = False
_cron_config = {
    "day_of_week": "fri",
    "hour": 8,
    "minute": 0,
}

JOB_ID = "auto_predictions"


async def _cron_job():
    """Cron job: run predictions for all leagues."""
    from services.runner import run_predictions
    print(f"[Scheduler] Cron triggered at {datetime.utcnow().isoformat()}")
    leagues = ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"]
    await run_predictions(leagues)
    print(f"[Scheduler] Cron completed at {datetime.utcnow().isoformat()}")


def start_scheduler() -> AsyncIOScheduler | None:
    """Initialize and start the scheduler (called on app startup)."""
    global _scheduler, _enabled

    _scheduler = AsyncIOScheduler()
    _scheduler.start()

    # Don't add the job by default - user enables via API
    _enabled = False

    return _scheduler


def _add_cron_job():
    """Add the cron job to the scheduler."""
    global _enabled
    if _scheduler is None:
        return

    # Remove existing job if any
    if _scheduler.get_job(JOB_ID):
        _scheduler.remove_job(JOB_ID)

    trigger = CronTrigger(
        day_of_week=_cron_config["day_of_week"],
        hour=_cron_config["hour"],
        minute=_cron_config["minute"],
    )
    _scheduler.add_job(_cron_job, trigger, id=JOB_ID, replace_existing=True)
    _enabled = True


def _remove_cron_job():
    """Remove the cron job from the scheduler."""
    global _enabled
    if _scheduler and _scheduler.get_job(JOB_ID):
        _scheduler.remove_job(JOB_ID)
    _enabled = False


# ─── API Endpoints ──────────────────────────────────────────────────────────

class CronConfig(BaseModel):
    day_of_week: str | None = None  # mon, tue, wed, thu, fri, sat, sun
    hour: int | None = None
    minute: int | None = None


@router.get("/scheduler/status")
async def scheduler_status():
    """Get scheduler status and configuration."""
    next_run = None
    if _scheduler and _enabled:
        job = _scheduler.get_job(JOB_ID)
        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat()

    return {
        "enabled": _enabled,
        "config": _cron_config,
        "next_run": next_run,
        "scheduler_running": _scheduler.running if _scheduler else False,
    }


@router.post("/scheduler/toggle")
async def toggle_scheduler():
    """Enable or disable the cron scheduler."""
    if _enabled:
        _remove_cron_job()
        return {"enabled": False, "message": "Scheduler disabled"}
    else:
        _add_cron_job()
        return {"enabled": True, "message": "Scheduler enabled"}


@router.put("/scheduler/config")
async def update_scheduler_config(config: CronConfig):
    """Update cron schedule configuration."""
    if config.day_of_week is not None:
        _cron_config["day_of_week"] = config.day_of_week
    if config.hour is not None:
        _cron_config["hour"] = config.hour
    if config.minute is not None:
        _cron_config["minute"] = config.minute

    # Re-add job with new config if enabled
    if _enabled:
        _add_cron_job()

    return {"config": _cron_config, "enabled": _enabled}
