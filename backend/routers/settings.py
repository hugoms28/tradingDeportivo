# -*- coding: utf-8 -*-
"""Settings router: discipline settings + bankroll."""
import json

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from models_db import Setting

router = APIRouter(tags=["settings"])


class SettingsUpdate(BaseModel):
    bankroll: float | None = None
    initial_bankroll: float | None = None
    peak_bankroll: float | None = None
    discipline_settings: dict | None = None


@router.get("/settings")
async def get_settings(session: AsyncSession = Depends(get_session)):
    """Get all settings."""
    result = await session.execute(select(Setting))
    rows = result.scalars().all()

    settings = {}
    for row in rows:
        if row.key == "discipline_settings":
            settings[row.key] = json.loads(row.value)
        else:
            settings[row.key] = row.value

    return settings


@router.put("/settings")
async def update_settings(
    data: SettingsUpdate,
    session: AsyncSession = Depends(get_session),
):
    """Update settings. Only provided fields are updated."""
    updates = {}
    if data.bankroll is not None:
        updates["bankroll"] = str(data.bankroll)
    if data.initial_bankroll is not None:
        updates["initial_bankroll"] = str(data.initial_bankroll)
    if data.peak_bankroll is not None:
        updates["peak_bankroll"] = str(data.peak_bankroll)
    if data.discipline_settings is not None:
        # Merge with existing settings
        existing = await session.execute(
            select(Setting).where(Setting.key == "discipline_settings")
        )
        row = existing.scalar_one_or_none()
        if row:
            current = json.loads(row.value)
            current.update(data.discipline_settings)
            updates["discipline_settings"] = json.dumps(current)
        else:
            updates["discipline_settings"] = json.dumps(data.discipline_settings)

    for key, value in updates.items():
        result = await session.execute(
            select(Setting).where(Setting.key == key)
        )
        row = result.scalar_one_or_none()
        if row:
            row.value = value
        else:
            session.add(Setting(key=key, value=value))

    await session.commit()
    return {"status": "ok", "updated": list(updates.keys())}
