# -*- coding: utf-8 -*-
"""Predictions router: run model and query predictions."""
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from models_db import Prediction
from services.runner import run_predictions, get_run_status

router = APIRouter(tags=["predictions"])


class PredictRequest(BaseModel):
    leagues: list[str]


class PredictResponse(BaseModel):
    run_id: str
    leagues: list[str]
    status: str


@router.post("/predict")
async def start_prediction(req: PredictRequest, background_tasks: BackgroundTasks):
    """Start model predictions for given leagues as a background task."""
    # Validate leagues
    valid = {"EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1", "Eredivisie"}
    leagues = [l for l in req.leagues if l in valid]
    if not leagues:
        return {"error": "No valid leagues provided"}

    # Pre-create the run_id so we can return it immediately
    run_id = str(uuid.uuid4())[:8]

    from services.runner import active_runs, run_predictions_with_id  # noqa: E402
    active_runs[run_id] = {
        "leagues": leagues,
        "status": "running",
        "completed": [],
        "failed": [],
    }

    background_tasks.add_task(run_predictions_with_id, run_id, leagues)  # FastAPI handles async

    return PredictResponse(run_id=run_id, leagues=leagues, status="started")


@router.get("/predict/{run_id}/status")
async def prediction_status(run_id: str):
    """Check status of a prediction run."""
    status = get_run_status(run_id)
    if status is None:
        return {"error": "Run not found", "run_id": run_id}
    return {"run_id": run_id, **status}


@router.get("/predictions")
async def get_predictions(
    league: str | None = Query(None),
    run_id: str | None = Query(None),
    limit: int = Query(100, le=500),
    session: AsyncSession = Depends(get_session),
):
    """Get predictions, optionally filtered by league or run_id."""
    query = select(Prediction).order_by(desc(Prediction.created_at))

    if league:
        query = query.where(Prediction.league == league)
    if run_id:
        query = query.where(Prediction.run_id == run_id)

    query = query.limit(limit)
    result = await session.execute(query)
    predictions = result.scalars().all()

    return [_prediction_to_dict(p) for p in predictions]


@router.get("/predictions/latest")
async def get_latest_predictions(
    session: AsyncSession = Depends(get_session),
):
    """Get the most recent predictions for each league."""
    from sqlalchemy import func

    # Find the latest run_id per league (by max created_at)
    subq = (
        select(
            Prediction.league,
            Prediction.run_id,
            func.max(Prediction.created_at).label("max_date"),
        )
        .group_by(Prediction.league, Prediction.run_id)
        .subquery()
    )

    latest_run_subq = (
        select(
            subq.c.league,
            func.max(subq.c.max_date).label("latest_date"),
        )
        .group_by(subq.c.league)
        .subquery()
    )

    run_id_subq = (
        select(subq.c.league, subq.c.run_id)
        .join(
            latest_run_subq,
            (subq.c.league == latest_run_subq.c.league)
            & (subq.c.max_date == latest_run_subq.c.latest_date),
        )
        .subquery()
    )

    query = (
        select(Prediction)
        .join(
            run_id_subq,
            (Prediction.league == run_id_subq.c.league)
            & (Prediction.run_id == run_id_subq.c.run_id),
        )
        .order_by(Prediction.league, desc(Prediction.created_at))
    )

    result = await session.execute(query)
    predictions = result.scalars().all()

    grouped: dict[str, list] = {}
    for p in predictions:
        grouped.setdefault(p.league, []).append(_prediction_to_dict(p))

    return grouped


def _prediction_to_dict(p: Prediction) -> dict:
    import json as _json
    raw = {}
    if p.raw_data:
        try:
            raw = _json.loads(p.raw_data)
        except Exception:
            pass

    # Rebuild value_bets from raw_data using thresholds
    from services.runner import _find_value_bets_from_raw, EDGE_THRESHOLDS
    value_bets = _find_value_bets_from_raw(raw) if raw else []

    return {
        "id": p.id,
        "league": p.league,
        "run_id": p.run_id,
        "home_team": p.home_team,
        "away_team": p.away_team,
        "p_home": p.p_home,
        "p_draw": p.p_draw,
        "p_away": p.p_away,
        "lambda_home": p.lambda_home,
        "mu_away": p.mu_away,
        "odds_home": p.odds_home,
        "odds_draw": p.odds_draw,
        "odds_away": p.odds_away,
        "edge_home": p.edge_home,
        "edge_draw": p.edge_draw,
        "edge_away": p.edge_away,
        "recommended_bet": p.recommended_bet,
        "kelly_stake": p.kelly_stake,
        "value_bets": value_bets,
        "raw": raw,
        "starts_at": raw.get("_starts"),
        "created_at": p.created_at.isoformat() if p.created_at else None,
    }
