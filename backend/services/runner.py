# -*- coding: utf-8 -*-
"""
Model execution runner. Imports from trading_deportivo package
and runs predictions for one or more leagues.
"""
import json
import math
import uuid
import traceback
from datetime import datetime

from sqlalchemy import select

from database import async_session
from models_db import Prediction, ModelRun

# Store active runs in memory for quick status checks
active_runs: dict[str, dict] = {}

# ─── Edge thresholds per market type ────────────────────────────────────────
# Valores optimizados via grid search (research/optimize_filters.py, Feb 2026)
# vs Pinnacle closing odds, test set oct 2025 – feb 2026, N>=50 bets.
EDGE_THRESHOLDS = {
    "dc":   0.05,   # Doble Oportunidad >= 5%  (no tested; conservative)
    "ah":   0.05,   # Asian Handicap >= 5%     (grid search Feb-24: sweet spot 3%, +5% margen seguridad)
    "ou":   0.14,   # Over/Under >= 14%        (grid search Feb-24: sweet spot 14%, N=85, yield +11.7%)
    "btts": 0.12,   # BTTS >= 12% (consejo)    (no tested; unchanged)
    "1x2":  0.06,   # 1X2 >= 6% (consejo)      (grid search Feb-24: sweet spot 4%, +2% margen seguridad)
}

# ─── Odds caps per market type ───────────────────────────────────────────────
# Cuotas maximas: por encima de estos valores el P&L es consistentemente negativo.
ODDS_CAPS = {
    "dc":  2.10,    # DC odds > 2.10 fuera de rango cobertura (derivado de AH)
    "ah":  2.30,    # AH odds > 2.30 degradan el yield (confirmado Feb-24 con gamma corregido)
    "ou":  2.00,    # OU odds > 2.00 pierden sistematicamente (confirmado Feb-24)
    "btts": 2.50,   # BTTS odds > 2.50 excesivo para mercado secundario
    "1x2": 3.00,    # 1X2 odds > 3.00 pierden sistematicamente (confirmado Feb-24)
}


async def run_predictions(leagues: list[str]) -> str:
    """Execute model predictions for given leagues. Returns run_id."""
    run_id = str(uuid.uuid4())[:8]
    active_runs[run_id] = {
        "leagues": leagues,
        "status": "running",
        "completed": [],
        "failed": [],
        "started_at": datetime.utcnow().isoformat(),
    }
    await run_predictions_with_id(run_id, leagues)
    return run_id


async def run_predictions_with_id(run_id: str, leagues: list[str]):
    """Execute model predictions with a pre-created run_id."""
    if run_id not in active_runs:
        active_runs[run_id] = {
            "leagues": leagues,
            "status": "running",
            "completed": [],
            "failed": [],
            "started_at": datetime.utcnow().isoformat(),
        }

    async with async_session() as session:
        for league in leagues:
            run = ModelRun(
                id=f"{run_id}_{league}",
                league=league,
                status="running",
                started_at=datetime.utcnow(),
            )
            session.add(run)
        await session.commit()

    for league in leagues:
        await _run_single_league(run_id, league)

    if active_runs[run_id]["failed"]:
        active_runs[run_id]["status"] = "partial"
    else:
        active_runs[run_id]["status"] = "completed"


def _save_fixture_snapshot(odds_dict: dict):
    """Persist event_id → 'Home vs Away' mapping so the resolver can look up
    team names for settled fixtures (PS3838 /v3/fixtures/settled has no names)."""
    import os
    snapshot_path = os.path.join(os.path.dirname(__file__), "..", "fixture_snapshot.json")
    try:
        try:
            with open(snapshot_path) as f:
                snapshot = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            snapshot = {}
        for match_key, data in odds_dict.items():
            if "_event_id" in data:
                snapshot[str(data["_event_id"])] = match_key
        with open(snapshot_path, "w") as f:
            json.dump(snapshot, f)
    except Exception as e:
        print(f"[Runner] Warning: no se pudo guardar fixture snapshot: {e}")


async def _run_single_league(run_id: str, league: str):
    """Run predictions for a single league."""
    db_run_id = f"{run_id}_{league}"

    try:
        import asyncio
        from trading_deportivo import load_model, predict_matchday
        from trading_deportivo.odds import fetch_ps3838_odds

        def _do_predict():
            model = load_model(latest=True, league=league)
            matches, odds_dict, status = fetch_ps3838_odds(league)
            if matches is None:
                raise RuntimeError(f"No odds available: {status}")
            _save_fixture_snapshot(odds_dict)
            df_pred = predict_matchday(matches, model, odds=odds_dict)
            return model, df_pred

        model, df_pred = await asyncio.to_thread(_do_predict)
        n_saved = await _save_predictions(run_id, league, df_pred)

        async with async_session() as session:
            run = await session.get(ModelRun, db_run_id)
            if run:
                run.status = "completed"
                run.completed_at = datetime.utcnow()
                run.n_matches = n_saved
                run.mse = model.get("mse")
                run.converged = model.get("converged", True)
                await session.commit()

        active_runs[run_id]["completed"].append(league)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        print(f"[Runner] Error for {league}: {error_msg}\n{tb}")

        async with async_session() as session:
            run = await session.get(ModelRun, db_run_id)
            if run:
                run.status = "failed"
                run.completed_at = datetime.utcnow()
                run.error = error_msg
                await session.commit()

        active_runs[run_id]["failed"].append({"league": league, "error": error_msg})


async def _save_predictions(run_id: str, league: str, df_pred) -> int:
    """Save prediction DataFrame rows to the database, replacing previous ones for the league."""
    from sqlalchemy import delete
    async with async_session() as session:
        await session.execute(delete(Prediction).where(Prediction.league == league))
        await session.commit()

    async with async_session() as session:
        count = 0
        for _, row in df_pred.iterrows():
            # Build raw_data dict with all columns (NaN-safe)
            raw = {}
            for col in df_pred.columns:
                val = row.get(col)
                if val is not None and not _isnan(val):
                    raw[col] = val if not isinstance(val, float) else round(val, 6)
                # Skip NaN and None

            # Find value bets per market with thresholds
            value_bets = _find_value_bets(row)

            # Best recommendation (first principal bet, else first tip)
            recommended = None
            kelly = None
            principals = [v for v in value_bets if v["type"] == "principal"]
            tips = [v for v in value_bets if v["type"] == "consejo"]

            best = (principals or tips or [None])[0]
            if best:
                recommended = best["label"]
                if best.get("prob") and best.get("odds"):
                    kelly = _calc_kelly(best["prob"], best["odds"])

            pred = Prediction(
                league=league,
                run_id=run_id,
                home_team=row.get("Local", ""),
                away_team=row.get("Visitante", ""),
                p_home=_safe_float(row.get("P_1")),
                p_draw=_safe_float(row.get("P_X")),
                p_away=_safe_float(row.get("P_2")),
                lambda_home=_safe_float(row.get("xG_Local")),
                mu_away=_safe_float(row.get("xG_Visita")),
                best_ou_line=_safe_float(row.get("Best_OU_Line")),
                best_ou_prob=_safe_float(row.get("Best_OU_Prob")),
                best_ah_line=_safe_float(row.get("Best_AH_Line")),
                best_ah_prob=_safe_float(row.get("Best_AH_Prob")),
                odds_home=_safe_float(row.get("Odds_1")),
                odds_draw=_safe_float(row.get("Odds_X")),
                odds_away=_safe_float(row.get("Odds_2")),
                edge_home=_safe_float(row.get("Edge_1")),
                edge_draw=_safe_float(row.get("Edge_X")),
                edge_away=_safe_float(row.get("Edge_2")),
                recommended_bet=recommended,
                kelly_stake=kelly,
                raw_data=json.dumps(raw, default=str),
            )
            session.add(pred)
            count += 1

        await session.commit()
    return count


def _find_value_bets(row) -> list[dict]:
    """
    Find all value bets for a match row, applying per-market edge thresholds.
    Returns list sorted by: principal bets first (by edge desc), then tips (by edge desc).
    """
    bets = []

    # ─── Doble Oportunidad (principal, >= 5%, odds <= 2.10) ───
    for key, label in [("Edge_1X", "1X"), ("Edge_X2", "X2"), ("Edge_12", "12")]:
        edge = _safe_float(row.get(key))
        odds = _safe_float(row.get(f"Odds_{label}"))
        if (edge is not None and edge >= EDGE_THRESHOLDS["dc"]
                and odds is not None and odds <= ODDS_CAPS["dc"]):
            bets.append({
                "market": "Doble Oportunidad",
                "label": f"DC {label}",
                "edge": edge,
                "prob": _safe_float(row.get(f"P_{label}")),
                "odds": odds,
                "type": "principal",
            })

    # ─── Asian Handicap (principal, >= 5%, odds <= 2.30) ───
    ah_edge = _safe_float(row.get("Best_AH_Edge"))
    ah_odds = _safe_float(row.get("Best_AH_Odds"))
    if (ah_edge is not None and ah_edge >= EDGE_THRESHOLDS["ah"]
            and ah_odds is not None and ah_odds <= ODDS_CAPS["ah"]):
        line = row.get("Best_AH_Line", "?")
        side = row.get("Best_AH_Side", "?")
        bets.append({
            "market": "Asian Handicap",
            "label": f"AH {side} {line}",
            "edge": ah_edge,
            "prob": _safe_float(row.get("Best_AH_Prob")),
            "odds": ah_odds,
            "type": "principal",
        })

    # ─── Over/Under (principal, >= 14%, odds <= 2.00) ───
    ou_edge = _safe_float(row.get("Best_OU_Edge"))
    ou_odds = _safe_float(row.get("Best_OU_Odds"))
    if (ou_edge is not None and ou_edge >= EDGE_THRESHOLDS["ou"]
            and ou_odds is not None and ou_odds <= ODDS_CAPS["ou"]):
        line = row.get("Best_OU_Line", "?")
        side = row.get("Best_OU_Side", "?")
        bets.append({
            "market": "Over/Under",
            "label": f"{side} {line}",
            "edge": ou_edge,
            "prob": _safe_float(row.get("Best_OU_Prob")),
            "odds": ou_odds,
            "type": "principal",
        })
    # Also check fixed O/U 2.5
    for key, label, prob_key in [("Edge_O25", "Over 2.5", "P_O25"), ("Edge_U25", "Under 2.5", "P_U25")]:
        edge = _safe_float(row.get(key))
        odds_key = "Odds_O25" if "O25" in key else "Odds_U25"
        odds = _safe_float(row.get(odds_key))
        if (edge is not None and edge >= EDGE_THRESHOLDS["ou"]
                and odds is not None and odds <= ODDS_CAPS["ou"]):
            bets.append({
                "market": "Over/Under",
                "label": label,
                "edge": edge,
                "prob": _safe_float(row.get(prob_key)),
                "odds": odds,
                "type": "principal",
            })

    # ─── BTTS (consejo, >= 12%, odds <= 2.50) ───
    for key, label, prob_key, odds_key in [
        ("Edge_BTTS_Si", "BTTS Sí", "P_BTTS", "Odds_BTTS_Si"),
        ("Edge_BTTS_No", "BTTS No", "P_BTTS_No", "Odds_BTTS_No"),
    ]:
        edge = _safe_float(row.get(key))
        odds = _safe_float(row.get(odds_key))
        if (edge is not None and edge >= EDGE_THRESHOLDS["btts"]
                and odds is not None and odds <= ODDS_CAPS["btts"]):
            bets.append({
                "market": "BTTS",
                "label": label,
                "edge": edge,
                "prob": _safe_float(row.get(prob_key)),
                "odds": odds,
                "type": "consejo",
            })

    # ─── 1X2 (consejo, >= 6%, odds <= 3.00) ───
    for key, label, prob_key, odds_key in [
        ("Edge_1", "1 (Local)", "P_1", "Odds_1"),
        ("Edge_X", "X (Empate)", "P_X", "Odds_X"),
        ("Edge_2", "2 (Visitante)", "P_2", "Odds_2"),
    ]:
        edge = _safe_float(row.get(key))
        odds = _safe_float(row.get(odds_key))
        if (edge is not None and edge >= EDGE_THRESHOLDS["1x2"]
                and odds is not None and odds <= ODDS_CAPS["1x2"]):
            bets.append({
                "market": "1X2",
                "label": label,
                "edge": edge,
                "prob": _safe_float(row.get(prob_key)),
                "odds": odds,
                "type": "consejo",
            })

    # Sort: principals first by edge desc, then tips by edge desc
    bets.sort(key=lambda x: (0 if x["type"] == "principal" else 1, -(x["edge"] or 0)))
    return bets


def _calc_kelly(prob: float, odds: float) -> float | None:
    """Kelly/4 fraction."""
    if not prob or not odds or prob <= 0 or odds <= 1:
        return None
    q = 1 - prob
    b = odds - 1
    k = (prob * b - q) / b
    return max(0, round(k * 0.25, 6))


def _isnan(val) -> bool:
    try:
        return math.isnan(float(val))
    except (ValueError, TypeError):
        return False


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _find_value_bets_from_raw(raw: dict) -> list[dict]:
    """Same as _find_value_bets but works on a plain dict (from raw_data JSON)."""
    return _find_value_bets(raw)


def get_run_status(run_id: str) -> dict | None:
    """Get status of an active or recent run."""
    return active_runs.get(run_id)
