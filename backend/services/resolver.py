# -*- coding: utf-8 -*-
"""
Auto-resolver de apuestas pendientes usando resultados de PS3838.
Al resolver también captura la odd de cierre y calcula el CLV.
"""
import asyncio
from datetime import datetime

from sqlalchemy import select

from database import async_session
from models_db import Bet, Setting


# ─── Lógica de resolución ────────────────────────────────────────────────────

def _classify(diff: float) -> str:
    d = round(diff * 4) / 4
    if d > 0.25:
        return "win"
    elif abs(d - 0.25) < 0.001:
        return "half_win"
    elif abs(d) < 0.001:
        return "void"
    elif abs(d + 0.25) < 0.001:
        return "half_loss"
    else:
        return "loss"


def _calculate_outcome(
    home_goals: int, away_goals: int, market: str, pick: str
) -> str | None:
    try:
        market = market.strip()
        pick = pick.strip()

        if market == "1X2":
            if "Local" in pick or pick.startswith("1"):
                return "win" if home_goals > away_goals else "loss"
            elif "Empate" in pick or pick == "X":
                return "win" if home_goals == away_goals else "loss"
            elif "Visitante" in pick or pick.startswith("2"):
                return "win" if away_goals > home_goals else "loss"

        elif market == "Doble Oportunidad":
            if "1X" in pick:
                return "win" if home_goals >= away_goals else "loss"
            elif "X2" in pick:
                return "win" if away_goals >= home_goals else "loss"
            elif "12" in pick:
                return "win" if home_goals != away_goals else "loss"

        elif market == "BTTS":
            btts = home_goals > 0 and away_goals > 0
            if "Sí" in pick or "Si" in pick or "Yes" in pick:
                return "win" if btts else "loss"
            else:
                return "win" if not btts else "loss"

        elif market == "Asian Handicap":
            parts = pick.split()
            if len(parts) < 3:
                return None
            side = parts[1]
            line = float(parts[2])
            diff = (home_goals + line) - away_goals if side == "Home" else (away_goals + line) - home_goals
            return _classify(diff)

        elif market == "Over/Under":
            parts = pick.split()
            if len(parts) < 2:
                return None
            side = parts[0]
            line = float(parts[1])
            total = home_goals + away_goals
            diff = (total - line) if side == "Over" else (line - total)
            return _classify(diff)

    except Exception as e:
        print(f"[Resolver] Error calculando resultado ({market} / {pick}): {e}")

    return None


def _get_closing_odds(closing: dict, market: str, pick: str) -> float | None:
    """
    Extrae la odd de cierre para el mercado/pick concreto de la apuesta.
    """
    if not closing:
        return None
    try:
        market = market.strip()
        pick = pick.strip()

        if market == "1X2":
            if "Local" in pick or pick.startswith("1"):
                return closing.get("home")
            elif "Empate" in pick or pick == "X":
                return closing.get("draw")
            elif "Visitante" in pick or pick.startswith("2"):
                return closing.get("away")

        elif market == "Doble Oportunidad":
            if "1X" in pick:
                return closing.get("dc_1X")
            elif "X2" in pick:
                return closing.get("dc_X2")
            elif "12" in pick:
                return closing.get("dc_12")

        elif market == "Asian Handicap":
            parts = pick.split()
            if len(parts) >= 3:
                side = parts[1].lower()   # "home" / "away"
                line = parts[2]           # e.g. "-0.25"
                return closing.get(f"ah_{side}_{line}")

        elif market == "Over/Under":
            parts = pick.split()
            if len(parts) >= 2:
                side = parts[0].lower()   # "over" / "under"
                line = parts[1]           # e.g. "2.5"
                return closing.get(f"{side}_{line}")

    except Exception:
        pass

    return None


# ─── Fetch de resultados + closing odds de PS3838 ────────────────────────────

def _fetch_results_sync(league: str) -> list[dict]:
    """
    Obtiene partidos liquidados desde PS3838 con marcadores y odds de cierre.
    Devuelve lista de dicts:
      home_team, away_team, home_goals, away_goals, closing_odds (dict)
    """
    from trading_deportivo.odds import _ps3838_request
    from trading_deportivo.team_mappings import get_ps3838_league_id, normalize_ps3838_name

    league_id = get_ps3838_league_id(league)

    # 1. Eventos liquidados → event_id: (home_goals, away_goals)
    try:
        settled = _ps3838_request("/v3/fixtures/settled", {
            "sportId": 29,
            "leagueIds": league_id,
        })
    except Exception as e:
        print(f"[Resolver] PS3838 settled error ({league}): {e}")
        return []

    if not settled or "fixtures" not in settled:
        return []

    scores: dict[int, tuple[int, int]] = {}
    for fixture in settled["fixtures"]:
        eid = fixture.get("id")
        if not eid:
            continue
        for period in fixture.get("periods", []):
            if period.get("number") == 0 and period.get("status") == 2:
                scores[eid] = (
                    int(period.get("team1Score") or 0),
                    int(period.get("team2Score") or 0),
                )
                break

    if not scores:
        return []

    event_ids_str = ",".join(str(eid) for eid in scores)

    # 2. Nombres de equipos para esos event IDs
    try:
        fixtures = _ps3838_request("/v3/fixtures", {
            "sportId": 29,
            "leagueIds": league_id,
            "eventIds": event_ids_str,
        })
    except Exception as e:
        print(f"[Resolver] PS3838 fixtures error ({league}): {e}")
        return []

    # 3. Odds de cierre para esos event IDs (best-effort)
    closing_by_event: dict[int, dict] = {}
    try:
        settled_odds = _ps3838_request("/v3/odds/settled", {
            "sportId": 29,
            "leagueIds": league_id,
            "oddsFormat": "Decimal",
        })
        if settled_odds and "leagues" in settled_odds:
            for lg in settled_odds["leagues"]:
                for event in lg.get("events", []):
                    eid = event.get("id")
                    if eid not in scores:
                        continue
                    co: dict = {}
                    for period in event.get("periods", []):
                        if period.get("number") != 0:
                            continue
                        # Moneyline
                        ml = period.get("moneyline", {})
                        if ml:
                            co["home"] = ml.get("home")
                            co["draw"] = ml.get("draw")
                            co["away"] = ml.get("away")
                            h, d, a = ml.get("home"), ml.get("draw"), ml.get("away")
                            if h and d:
                                co["dc_1X"] = round(1 / (1/h + 1/d), 4)
                            if d and a:
                                co["dc_X2"] = round(1 / (1/d + 1/a), 4)
                            if h and a:
                                co["dc_12"] = round(1 / (1/h + 1/a), 4)
                        # Spreads (AH)
                        for spread in period.get("spreads", []):
                            hdp = spread.get("hdp")
                            if hdp is not None:
                                if spread.get("home"):
                                    co[f"ah_home_{hdp}"] = spread["home"]
                                if spread.get("away"):
                                    co[f"ah_away_{hdp}"] = spread["away"]
                        # Totals (O/U)
                        for total in period.get("totals", []):
                            pts = total.get("points")
                            if pts is not None:
                                if total.get("over"):
                                    co[f"over_{pts}"] = total["over"]
                                if total.get("under"):
                                    co[f"under_{pts}"] = total["under"]
                        break
                    closing_by_event[eid] = co
    except Exception as e:
        print(f"[Resolver] PS3838 closing odds error ({league}): {e}")
        # No blocking — continue without CLV

    # 4. Combinar todo
    results = []
    if fixtures and "league" in fixtures:
        for league_data in fixtures["league"]:
            for event in league_data.get("events", []):
                eid = event.get("id")
                if eid not in scores:
                    continue
                home = normalize_ps3838_name(event.get("home", ""), league)
                away = normalize_ps3838_name(event.get("away", ""), league)
                home_goals, away_goals = scores[eid]
                results.append({
                    "home_team": home,
                    "away_team": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "closing_odds": closing_by_event.get(eid, {}),
                })

    return results


# ─── Función principal ───────────────────────────────────────────────────────

async def auto_resolve_pending_bets() -> dict:
    """
    Comprueba todas las apuestas pendientes contra resultados de PS3838
    y las resuelve automáticamente. También calcula el CLV cuando hay
    odds de cierre disponibles.
    """
    async with async_session() as session:
        result = await session.execute(select(Bet).where(Bet.result.is_(None)))
        pending = list(result.scalars().all())

    if not pending:
        return {"resolved": 0, "skipped": 0, "errors": 0}

    print(f"[Resolver] Revisando {len(pending)} apuesta(s) pendiente(s)...")

    by_league: dict[str, list[Bet]] = {}
    for bet in pending:
        by_league.setdefault(bet.league or "", []).append(bet)

    resolved = skipped = errors = 0

    for league, bets in by_league.items():
        if not league:
            skipped += len(bets)
            continue

        try:
            match_results = await asyncio.to_thread(_fetch_results_sync, league)
        except Exception as e:
            print(f"[Resolver] Fallo al obtener resultados de {league}: {e}")
            skipped += len(bets)
            continue

        results_index: dict[str, dict] = {
            f"{r['home_team']} vs {r['away_team']}": r
            for r in match_results
        }

        for bet in bets:
            match = results_index.get(bet.event)
            if match is None:
                skipped += 1
                continue

            outcome = _calculate_outcome(
                match["home_goals"], match["away_goals"], bet.market, bet.pick
            )
            if outcome is None:
                skipped += 1
                continue

            try:
                async with async_session() as session:
                    b = await session.get(Bet, bet.id)
                    if b is None or b.result is not None:
                        continue

                    if outcome == "win":
                        pnl = b.stake * (b.odds - 1)
                    elif outcome == "half_win":
                        pnl = (b.stake / 2) * (b.odds - 1)
                    elif outcome == "loss":
                        pnl = -b.stake
                    elif outcome == "half_loss":
                        pnl = -(b.stake / 2)
                    else:  # void
                        pnl = 0.0

                    # Bankroll delta = pnl neto (stake nunca se descontó al crear)
                    bankroll_delta = round(pnl, 2)

                    b.result = outcome
                    b.pnl = round(pnl, 2)
                    b.resolved_at = datetime.utcnow()

                    # CLV: (entry_odds / closing_odds - 1) * 100
                    closing_odds = _get_closing_odds(
                        match.get("closing_odds", {}), b.market, b.pick
                    )
                    if closing_odds and closing_odds > 1:
                        b.closing_odds = round(closing_odds, 4)
                        b.clv = round((b.odds / closing_odds - 1) * 100, 2)

                    # Actualizar bankroll
                    bankroll_res = await session.execute(
                        select(Setting).where(Setting.key == "bankroll")
                    )
                    bankroll_row = bankroll_res.scalar_one_or_none()
                    if bankroll_row:
                        current = float(bankroll_row.value)
                        new_br = round(current + bankroll_delta, 2)
                        bankroll_row.value = str(new_br)

                        peak_res = await session.execute(
                            select(Setting).where(Setting.key == "peak_bankroll")
                        )
                        peak_row = peak_res.scalar_one_or_none()
                        if peak_row and new_br > float(peak_row.value):
                            peak_row.value = str(new_br)

                    await session.commit()
                    resolved += 1

                    clv_str = f" | CLV: {b.clv:+.1f}%" if b.clv is not None else ""
                    score = f"{match['home_goals']}-{match['away_goals']}"
                    print(
                        f"[Resolver] ✓ {bet.event} | {bet.market} {bet.pick} "
                        f"| {score} → {outcome} ({b.pnl:+.2f}€){clv_str}"
                    )

            except Exception as e:
                print(f"[Resolver] Error guardando apuesta {bet.id}: {e}")
                errors += 1

    print(f"[Resolver] Finalizado — {resolved} resueltas, {skipped} sin resultado, {errors} errores")
    return {"resolved": resolved, "skipped": skipped, "errors": errors}
