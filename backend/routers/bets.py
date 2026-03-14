# -*- coding: utf-8 -*-
"""Bets router: CRUD + resolve + stats."""
import json
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from models_db import Bet, Setting

router = APIRouter(tags=["bets"])


# ─── Schemas ────────────────────────────────────────────────────────────────

class BetCreate(BaseModel):
    event: str
    league: str = ""
    source: str  # Modelo / Tipster / Propia
    tipster_name: str = ""
    sport: str | None = None
    market: str
    pick: str
    odds: float
    model_prob: float | None = None
    stake: float
    edge: float | None = None
    prediction_id: int | None = None
    match_starts_at: str | None = None
    bookmaker: str | None = None


class Ps3838BetCreate(BaseModel):
    prediction_id: int | None = None   # Para apuestas del modelo
    event_id: int | None = None        # Para apuestas manuales (búsqueda previa)
    sport_id: int = 29                 # sportId PS3838 (29=fútbol, 4=basket, etc.)
    league_id: int | None = None       # leagueId PS3838 (necesario para /v2/line)
    event: str
    league: str = ""
    market: str                        # 1X2 | Asian Handicap | Over/Under
    label: str                         # "1 (Local)" | "AH Home -0.5" | "Over 2.5"
    odds: float
    stake: float
    prob: float | None = None
    edge: float | None = None
    source: str = "Modelo"
    tipster_name: str = ""
    bookmaker: str = "PS3838"


class BetUpdate(BaseModel):
    bookmaker: str | None = None
    tipster_name: str | None = None
    source: str | None = None


class BetResolve(BaseModel):
    result: str  # win / half_win / loss / half_loss / void


# ─── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/bets")
async def list_bets(
    league: str | None = Query(None),
    status: str | None = Query(None),  # pending / resolved
    limit: int = Query(100, le=1000),
    session: AsyncSession = Depends(get_session),
):
    """List bets with optional filters."""
    query = select(Bet).order_by(desc(Bet.created_at))

    if league:
        query = query.where(Bet.league == league)
    if status == "pending":
        query = query.where(Bet.result.is_(None))
    elif status == "resolved":
        query = query.where(Bet.result.isnot(None))

    query = query.limit(limit)
    result = await session.execute(query)
    bets = result.scalars().all()

    return [_bet_to_dict(b) for b in bets]


@router.post("/bets")
async def create_bet(
    data: BetCreate,
    session: AsyncSession = Depends(get_session),
):
    """Create a new bet and deduct stake from bankroll immediately."""
    bet = Bet(
        prediction_id=data.prediction_id,
        event=data.event,
        league=data.league,
        source=data.source,
        tipster_name=data.tipster_name,
        sport=data.sport,
        market=data.market,
        pick=data.pick,
        odds=data.odds,
        model_prob=data.model_prob,
        stake=data.stake,
        edge=data.edge,
        match_starts_at=data.match_starts_at,
        bookmaker=data.bookmaker,
    )
    session.add(bet)

    # Bankroll no cambia al registrar la apuesta — solo se actualiza al resolver.
    # Así todas las apuestas de una misma jornada calculan Kelly sobre el bankroll completo.

    await session.commit()
    await session.refresh(bet)
    return _bet_to_dict(bet)


@router.patch("/bets/{bet_id}/resolve")
async def resolve_bet(
    bet_id: int,
    data: BetResolve,
    session: AsyncSession = Depends(get_session),
):
    """Resolve a bet and calculate PnL. Updates bankroll in settings."""
    bet = await session.get(Bet, bet_id)
    if not bet:
        return {"error": "Bet not found"}
    if bet.result is not None:
        return {"error": "Bet already resolved"}

    # Calculate net PnL (for stats/ROI tracking)
    # half_win / half_loss: stake split in two — half wins/loses, half is pushed back
    if data.result == "win":
        pnl = bet.stake * (bet.odds - 1)
    elif data.result == "half_win":
        pnl = (bet.stake / 2) * (bet.odds - 1)
    elif data.result == "loss":
        pnl = -bet.stake
    elif data.result == "half_loss":
        pnl = -(bet.stake / 2)
    else:  # void
        pnl = 0.0

    bet.result = data.result
    bet.pnl = round(pnl, 2)
    bet.resolved_at = datetime.utcnow()

    # Bankroll se actualiza solo al resolver: delta = pnl neto.
    # La stake nunca se descontó al crear, así que aquí aplicamos el resultado completo.
    bankroll_delta = round(pnl, 2)

    bankroll_setting = await session.execute(
        select(Setting).where(Setting.key == "bankroll")
    )
    bankroll_row = bankroll_setting.scalar_one_or_none()
    if bankroll_row:
        current = float(bankroll_row.value)
        new_bankroll = round(current + bankroll_delta, 2)
        bankroll_row.value = str(new_bankroll)

        # Update peak if new high
        peak_setting = await session.execute(
            select(Setting).where(Setting.key == "peak_bankroll")
        )
        peak_row = peak_setting.scalar_one_or_none()
        if peak_row and new_bankroll > float(peak_row.value):
            peak_row.value = str(new_bankroll)

    await session.commit()
    await session.refresh(bet)
    return _bet_to_dict(bet)


@router.patch("/bets/{bet_id}")
async def update_bet(
    bet_id: int,
    data: BetUpdate,
    session: AsyncSession = Depends(get_session),
):
    """Update editable bet fields (bookmaker, etc)."""
    bet = await session.get(Bet, bet_id)
    if not bet:
        return {"error": "Bet not found"}
    if data.bookmaker is not None:
        bet.bookmaker = data.bookmaker
    if data.tipster_name is not None:
        bet.tipster_name = data.tipster_name
    if data.source is not None:
        bet.source = data.source
    await session.commit()
    await session.refresh(bet)
    return _bet_to_dict(bet)


@router.delete("/bets/{bet_id}")
async def delete_bet(
    bet_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Delete a pending bet."""
    bet = await session.get(Bet, bet_id)
    if not bet:
        return {"error": "Bet not found"}
    if bet.result is not None:
        return {"error": "Cannot delete a resolved bet"}
    await session.delete(bet)
    await session.commit()
    return {"deleted": bet_id}


def _map_to_ps3838_params(market: str, label: str, lines: dict) -> tuple:
    """Traduce market+label a parámetros PS3838: (line_id, bet_type, team, alt_line_id)."""
    line_id = lines.get("line_id")
    if not line_id:
        raise ValueError("No hay line_id disponible para este evento")

    if market == "1X2":
        if "Local" in label or label.startswith("1 "):
            team = "Team1"
        elif "Empate" in label or label.startswith("X"):
            team = "Draw"
        else:
            team = "Team2"
        return line_id, "MONEYLINE", team, None, None

    elif market == "Asian Handicap":
        parts = label.split()  # ["AH", "Home", "-0.5"]
        side = parts[1] if len(parts) >= 3 else "Home"
        team = "Team1" if side == "Home" else "Team2"
        try:
            hdp_raw = float(parts[-1])
            hdp_search = hdp_raw if side == "Home" else -hdp_raw
        except (ValueError, IndexError):
            hdp_raw = None
            hdp_search = None
        alt_line_id = None
        for spread in lines.get("spreads", []):
            if hdp_search is not None and abs(spread.get("hdp", 999) - hdp_search) < 0.13:
                alt_line_id = spread.get("altLineId")
                break
        if alt_line_id is None and lines.get("spreads"):
            alt_line_id = lines["spreads"][0].get("altLineId")
        return line_id, "SPREAD", team, alt_line_id, hdp_search

    elif market == "Over/Under":
        parts = label.split()  # ["Over", "2.5"] o ["Under", "2.75"]
        side = parts[0].lower() if parts else "over"
        team = "Team1" if side == "over" else "Team2"
        try:
            points_search = float(parts[-1])
        except (ValueError, IndexError):
            points_search = None
        alt_line_id = None
        for total in lines.get("totals", []):
            if points_search is not None and abs(total.get("points", 999) - points_search) < 0.13:
                alt_line_id = total.get("altLineId")
                break
        if alt_line_id is None and lines.get("totals"):
            alt_line_id = lines["totals"][0].get("altLineId")
        return line_id, "TOTAL_POINTS", team, alt_line_id, points_search

    else:
        raise ValueError(f"Mercado '{market}' no soportado para colocación en PS3838")


@router.post("/bets/place-ps3838")
async def place_bet_on_ps3838(
    data: Ps3838BetCreate,
    session: AsyncSession = Depends(get_session),
):
    """Coloca una apuesta en PS3838 vía API y la registra localmente."""
    from trading_deportivo.odds import place_ps3838_bet, get_ps3838_event_lines
    from models_db import Prediction

    # Resolver event_id
    event_id = data.event_id
    match_starts_at = None
    if not event_id and data.prediction_id:
        pred = await session.get(Prediction, data.prediction_id)
        if not pred or not pred.raw_data:
            return {"error": "Predicción no encontrada"}
        raw = json.loads(pred.raw_data)
        event_id = raw.get("_event_id")
        match_starts_at = raw.get("_starts")

    if not event_id:
        return {"error": "Se requiere event_id o prediction_id con evento PS3838"}

    # Obtener líneas actuales
    try:
        lines = get_ps3838_event_lines(int(event_id), sport_id=data.sport_id)
    except Exception as e:
        return {"error": f"Error obteniendo líneas de PS3838: {e}"}

    if not lines.get("line_id"):
        return {"error": "No se encontraron líneas para este evento (puede haber cerrado)"}

    # Mapear mercado → parámetros PS3838
    try:
        line_id, bet_type, team, alt_line_id, handicap = _map_to_ps3838_params(data.market, data.label, lines)
    except ValueError as e:
        return {"error": str(e)}

    # Colocar apuesta en PS3838
    try:
        ps3838_result = place_ps3838_bet(
            line_id=line_id,
            bet_type=bet_type,
            team=team,
            stake=data.stake,
            alt_line_id=alt_line_id,
            event_id=int(event_id),
            sport_id=data.sport_id,
        )
    except Exception as e:
        return {"error": f"Error en PS3838 API: {e}"}

    print(f"[PS3838 BET RESPONSE] {ps3838_result}")

    status = ps3838_result.get("status", "")
    if status in ("NOT_ACCEPTED", "PROCESSED_WITH_ERROR"):
        error_code = ps3838_result.get("errorCode", "UNKNOWN")
        return {"error": f"PS3838: {error_code}"}

    ps3838_bet_id = str(ps3838_result.get("betId") or "")

    # Guardar en DB local
    bet = Bet(
        prediction_id=data.prediction_id,
        event=data.event,
        league=data.league,
        source=data.source,
        tipster_name=data.tipster_name,
        market=data.market,
        pick=data.label,
        odds=data.odds,
        model_prob=data.prob,
        stake=data.stake,
        edge=data.edge,
        match_starts_at=match_starts_at,
        bookmaker=data.bookmaker,
        ps3838_bet_id=ps3838_bet_id,
    )
    session.add(bet)
    await session.commit()
    await session.refresh(bet)
    return _bet_to_dict(bet)


@router.post("/bets/auto-resolve")
async def trigger_auto_resolve():
    """Lanza la resolución automática de apuestas pendientes contra Understat."""
    from services.resolver import auto_resolve_pending_bets
    result = await auto_resolve_pending_bets()
    return result


@router.get("/bets/stats")
async def bet_stats(
    session: AsyncSession = Depends(get_session),
):
    """Compute betting statistics (ROI, win rate, drawdown, etc)."""
    result = await session.execute(
        select(Bet).order_by(Bet.created_at)
    )
    all_bets = result.scalars().all()

    resolved = [b for b in all_bets if b.result is not None]
    pending = [b for b in all_bets if b.result is None]

    total_pnl = sum(b.pnl for b in resolved)
    total_staked = sum(b.stake for b in resolved)
    wins = [b for b in resolved if b.result in ("win", "half_win")]
    losses = [b for b in resolved if b.result in ("loss", "half_loss")]

    win_rate = (len(wins) / len(resolved) * 100) if resolved else 0
    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0

    # Today stats
    today = datetime.utcnow().date()
    today_bets = [b for b in all_bets if b.created_at and b.created_at.date() == today]
    today_resolved = [b for b in today_bets if b.result is not None]
    today_pnl = sum(b.pnl for b in today_resolved)

    # Week stats
    week_start = datetime.utcnow() - timedelta(days=datetime.utcnow().weekday())
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    week_bets = [b for b in resolved if b.created_at and b.created_at >= week_start]
    week_pnl = sum(b.pnl for b in week_bets)

    # Consecutive losses (half_loss also counts)
    consecutive_losses = 0
    for b in reversed(resolved):
        if b.result in ("loss", "half_loss"):
            consecutive_losses += 1
        else:
            break

    # Drawdown from peak
    bankroll_res = await session.execute(select(Setting).where(Setting.key == "bankroll"))
    peak_res = await session.execute(select(Setting).where(Setting.key == "peak_bankroll"))
    initial_res = await session.execute(select(Setting).where(Setting.key == "initial_bankroll"))

    bankroll = float((bankroll_res.scalar_one_or_none() or Setting(value="500")).value)
    peak = float((peak_res.scalar_one_or_none() or Setting(value="500")).value)
    initial = float((initial_res.scalar_one_or_none() or Setting(value="500")).value)

    drawdown = (peak - bankroll) / peak if peak > 0 else 0

    # By source
    by_source = {}
    for source in ["Modelo", "Tipster", "Propia"]:
        sb = [b for b in resolved if b.source == source]
        by_source[source] = {
            "count": len(sb),
            "pnl": round(sum(b.pnl for b in sb), 2),
            "winRate": round(
                len([b for b in sb if b.result in ("win", "half_win")]) / len(sb) * 100, 1
            ) if sb else 0,
        }

    # CLV medio (solo apuestas con closing_odds disponible)
    clv_bets = [b for b in resolved if b.clv is not None]
    avg_clv = round(sum(b.clv for b in clv_bets) / len(clv_bets), 2) if clv_bets else None

    return {
        "totalPnL": round(total_pnl, 2),
        "wins": len(wins),
        "losses": len(losses),
        "winRate": round(win_rate, 1),
        "roi": round(roi, 1),
        "todayPnL": round(today_pnl, 2),
        "weekPnL": round(week_pnl, 2),
        "todayBetsCount": len(today_bets),
        "drawdownFromPeak": round(drawdown, 4),
        "consecutiveLosses": consecutive_losses,
        "bySource": by_source,
        "resolvedCount": len(resolved),
        "pendingCount": len(pending),
        "bankroll": bankroll,
        "initialBankroll": initial,
        "peakBankroll": peak,
        "avgClv": avg_clv,
        "clvCount": len(clv_bets),
    }


def _bet_to_dict(b: Bet) -> dict:
    return {
        "id": b.id,
        "predictionId": b.prediction_id,
        "event": b.event,
        "league": b.league,
        "source": b.source,
        "tipsterName": b.tipster_name,
        "sport": b.sport,
        "market": b.market,
        "pick": b.pick,
        "odds": b.odds,
        "modelProb": b.model_prob,
        "stake": b.stake,
        "edge": b.edge,
        "result": b.result,
        "pnl": b.pnl,
        "closingOdds": b.closing_odds,
        "clv": b.clv,
        "matchStartsAt": b.match_starts_at,
        "bookmaker": b.bookmaker,
        "ps3838BetId": b.ps3838_bet_id,
        "timestamp": b.created_at.isoformat() if b.created_at else None,
        "resolvedAt": b.resolved_at.isoformat() if b.resolved_at else None,
    }
