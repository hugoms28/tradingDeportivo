# -*- coding: utf-8 -*-
"""Async SQLite database setup with SQLAlchemy."""
import json
import os
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from models_db import Base

BACKEND_DIR = Path(__file__).parent
DB_PATH = BACKEND_DIR / "trading.db"
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# Directories for models and cache (inside backend/)
MODELS_DIR = str(BACKEND_DIR / "models")
CACHE_DIR = str(BACKEND_DIR / "cache")

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


def setup_dirs():
    """Create models/ and cache/ directories, load .env, and set env vars
    so trading_deportivo.config picks them up."""
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ["TRADING_MODELS_DIR"] = MODELS_DIR
    os.environ["TRADING_CACHE_DIR"] = CACHE_DIR


async def init_db():
    """Create dirs, tables, and seed defaults."""
    setup_dirs()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Migrate: add columns added after initial schema
        from sqlalchemy import text
        for stmt in [
            "ALTER TABLE bets ADD COLUMN match_starts_at TEXT",
            "ALTER TABLE bets ADD COLUMN closing_odds REAL",
            "ALTER TABLE bets ADD COLUMN clv REAL",
            "ALTER TABLE bets ADD COLUMN sport TEXT",
        ]:
            try:
                await conn.execute(text(stmt))
            except Exception:
                pass  # Column already exists
    await seed_defaults()


async def seed_defaults():
    """Insert default settings if they don't exist."""
    from models_db import Setting
    from sqlalchemy import select

    defaults = {
        "bankroll": "500",
        "initial_bankroll": "500",
        "peak_bankroll": "500",
        "discipline_settings": json.dumps({
            "dailyStopLoss": -60,
            "weeklyStopLoss": -100,
            "maxDrawdownPct": 0.20,
            "maxDailyBets": 5,
            "cooldownLosses": 2,
            "cooldownHours": 2,
            "kellyFraction": 0.25,
            "maxStakePct": 0.05,
        }),
    }

    async with async_session() as session:
        for key, value in defaults.items():
            existing = await session.execute(
                select(Setting).where(Setting.key == key)
            )
            if existing.scalar_one_or_none() is None:
                session.add(Setting(key=key, value=value))
        await session.commit()


async def get_session():
    """Dependency for FastAPI routes."""
    async with async_session() as session:
        yield session
