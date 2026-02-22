# -*- coding: utf-8 -*-
"""FastAPI application for trading-deportivo backend."""
import asyncio
from contextlib import asynccontextmanager

# Set up dirs and env vars BEFORE any trading_deportivo import
from database import setup_dirs
setup_dirs()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from routers import predictions, bets, models, settings
from services.scheduler import router as scheduler_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    from services.scheduler import start_scheduler
    from services.resolver import auto_resolve_pending_bets
    scheduler = start_scheduler()
    # Auto-resolver en background para no bloquear el arranque
    asyncio.create_task(auto_resolve_pending_bets())
    yield
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)


app = FastAPI(title="Trading Deportivo API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router, prefix="/api")
app.include_router(bets.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(settings.router, prefix="/api")
app.include_router(scheduler_router, prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
