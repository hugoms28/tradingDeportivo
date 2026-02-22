# -*- coding: utf-8 -*-
"""Models router: list, train, and inspect trained models."""
import asyncio
import os
import pickle
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session, async_session
from models_db import ModelRun

router = APIRouter(tags=["models"])

# ─── Training progress (in-memory) ─────────────────────────────────────────

STEPS = [
    {"key": "fetch", "label": "Descargando datos de Understat", "pct": 0},
    {"key": "process", "label": "Procesando shots", "pct": 25},
    {"key": "matrix", "label": "Construyendo matriz xG", "pct": 40},
    {"key": "train", "label": "Entrenando modelo Dixon-Coles", "pct": 55},
    {"key": "save", "label": "Guardando modelo", "pct": 85},
    {"key": "done", "label": "Completado", "pct": 100},
]

# train_id -> {league, step, step_label, pct, status, error, started_at}
training_progress: dict[str, dict] = {}


class TrainRequest(BaseModel):
    leagues: list[str]


# ─── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/models")
async def list_models(league: str | None = Query(None)):
    """List saved model files on disk."""
    from trading_deportivo.config import MODELS_DIR

    if not os.path.exists(MODELS_DIR):
        return []

    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    if league:
        files = [f for f in files if league in f]

    models = []
    for f in sorted(files, reverse=True):
        path = os.path.join(MODELS_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        mtime = datetime.fromtimestamp(os.path.getmtime(path))

        metadata = {}
        try:
            with open(path, "rb") as fh:
                model = pickle.load(fh)
                metadata = model.get("_metadata", {})
                metadata["n_teams"] = len(model.get("teams", []))
                metadata["gamma"] = round(model.get("gamma", 0), 4)
                metadata["rho"] = round(model.get("rho", 0), 4)
                metadata["mse"] = model.get("mse")
                metadata["converged"] = model.get("converged")
        except Exception:
            pass

        models.append({
            "filename": f,
            "size_kb": round(size_kb, 1),
            "modified": mtime.isoformat(),
            **metadata,
        })

    return models


@router.get("/models/{league}/latest")
async def get_latest_model(league: str):
    """Get info about the latest model for a league."""
    from trading_deportivo.config import MODELS_DIR

    if not os.path.exists(MODELS_DIR):
        return {"error": "No models directory found"}

    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl") and league in f]
    if not files:
        return {"error": f"No models found for {league}"}

    files.sort(
        key=lambda x: os.path.getmtime(os.path.join(MODELS_DIR, x)), reverse=True
    )
    latest = files[0]
    path = os.path.join(MODELS_DIR, latest)

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)

        metadata = model.get("_metadata", {})
        return {
            "filename": latest,
            "league": league,
            "saved_at": metadata.get("saved_at"),
            "n_teams": len(model.get("teams", [])),
            "gamma": round(model.get("gamma", 0), 4),
            "rho": round(model.get("rho", 0), 4),
            "mse": model.get("mse"),
            "converged": model.get("converged"),
            "teams": sorted(model.get("teams", [])),
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/train")
async def train_model(req: TrainRequest, background_tasks: BackgroundTasks):
    """Train model for given leagues (background task). Returns train_id for progress polling."""
    valid = {"EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"}
    leagues = [l for l in req.leagues if l in valid]
    if not leagues:
        return {"error": "No valid leagues provided"}

    train_id = str(uuid.uuid4())[:8]

    # Init progress for each league
    for league in leagues:
        key = f"{train_id}_{league}"
        training_progress[key] = {
            "league": league,
            "step": "waiting",
            "step_label": "En cola",
            "pct": 0,
            "status": "pending",
            "error": None,
            "started_at": datetime.utcnow().isoformat(),
        }

    async def _train_all():
        for league in leagues:
            await _train_league(train_id, league)

    background_tasks.add_task(_train_all)
    return {"status": "started", "train_id": train_id, "leagues": leagues}


@router.get("/train/{train_id}/progress")
async def get_training_progress(train_id: str):
    """Get progress of a training run."""
    # Find all entries for this train_id
    entries = {
        k: v for k, v in training_progress.items() if k.startswith(train_id)
    }
    if not entries:
        return {"error": "Training not found"}

    leagues = {}
    for key, val in entries.items():
        leagues[val["league"]] = val

    # Overall status
    statuses = [v["status"] for v in leagues.values()]
    if all(s == "completed" for s in statuses):
        overall = "completed"
    elif any(s == "failed" for s in statuses):
        overall = "partial" if any(s == "completed" for s in statuses) else "failed"
    elif any(s == "running" for s in statuses):
        overall = "running"
    else:
        overall = "pending"

    return {
        "train_id": train_id,
        "status": overall,
        "leagues": leagues,
    }


async def _train_league(train_id: str, league: str):
    """Train a single league model with step-by-step progress."""
    progress_key = f"{train_id}_{league}"
    db_run_id = f"train_{train_id}_{league}"

    def _update(step_key: str, status: str = "running"):
        step = next((s for s in STEPS if s["key"] == step_key), None)
        if step:
            training_progress[progress_key].update({
                "step": step_key,
                "step_label": step["label"],
                "pct": step["pct"],
                "status": status,
            })

    # Create DB record
    async with async_session() as session:
        run = ModelRun(
            id=db_run_id,
            league=league,
            status="running",
            started_at=datetime.utcnow(),
        )
        session.add(run)
        await session.commit()

    _update("fetch")

    try:
        from trading_deportivo import (
            get_league_match_ids, fetch_all_shots, shots_to_df,
            build_matches_with_dates, fit_dixon_coles_xg, save_model,
        )

        # Understat uses start year: "2024" = 2024-25, "2025" = 2025-26
        seasons = ["2024", "2025"]

        def _do_train():
            _update("fetch")
            match_ids, raw_matches = get_league_match_ids(league, seasons, use_cache=True)
            shots = fetch_all_shots(match_ids, league, seasons, use_cache=True)

            _update("process")
            df_shots = shots_to_df(shots)
            df_shots["venue"] = df_shots["h_a"].map({"h": "home", "a": "away"})
            df_shots["xg"] = df_shots["xg_understat"]

            _update("matrix")
            matches_with_dates = build_matches_with_dates(raw_matches, df_shots)

            _update("train")
            model = fit_dixon_coles_xg(matches_with_dates, raw_matches)

            _update("save")
            save_model(model, league=league)

            return model, len(matches_with_dates)

        model, n_matches = await asyncio.to_thread(_do_train)

        _update("done", status="completed")

        async with async_session() as session:
            run = await session.get(ModelRun, db_run_id)
            if run:
                run.status = "completed"
                run.completed_at = datetime.utcnow()
                run.n_matches = n_matches
                run.mse = model.get("mse")
                run.converged = model.get("converged", True)
                await session.commit()

    except Exception as e:
        training_progress[progress_key].update({
            "status": "failed",
            "error": str(e),
        })

        async with async_session() as session:
            run = await session.get(ModelRun, db_run_id)
            if run:
                run.status = "failed"
                run.completed_at = datetime.utcnow()
                run.error = str(e)
                await session.commit()


@router.get("/model-runs")
async def list_model_runs(
    league: str | None = Query(None),
    limit: int = Query(20, le=100),
    session: AsyncSession = Depends(get_session),
):
    """List recent model run history."""
    query = select(ModelRun).order_by(desc(ModelRun.started_at))
    if league:
        query = query.where(ModelRun.league == league)
    query = query.limit(limit)

    result = await session.execute(query)
    runs = result.scalars().all()

    return [
        {
            "id": r.id,
            "league": r.league,
            "status": r.status,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            "n_matches": r.n_matches,
            "mse": r.mse,
            "converged": r.converged,
            "error": r.error,
        }
        for r in runs
    ]
