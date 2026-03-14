# -*- coding: utf-8 -*-
"""
Integracion con PS3838 (Pinnacle) y alertas Telegram.
"""
import base64
from datetime import datetime, timedelta

import requests

from .config import (
    PS3838_USERNAME, PS3838_PASSWORD, PS3838_BASE_URL,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
)
from .team_mappings import get_ps3838_league_id, get_ps3838_map, normalize_ps3838_name


# =============================================================================
# TELEGRAM
# =============================================================================

def send_telegram_alert(message):
    """Envia alerta por Telegram cuando PS3838 falla."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"  [TELEGRAM] Sin configurar. Mensaje: {message}")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": f"JEKE MODEL ALERT\n\n{message}"
        }, timeout=10)
    except Exception as e:
        print(f"  [TELEGRAM] Error enviando alerta: {e}")


# =============================================================================
# PS3838 (PINNACLE) API
# =============================================================================

def _ps3838_auth_header():
    """Genera header de autenticacion Basic para PS3838."""
    auth_str = f"{PS3838_USERNAME}:{PS3838_PASSWORD}"
    b64 = base64.b64encode(auth_str.encode('utf-8')).decode('utf-8')
    return {"Authorization": f"Basic {b64}"}


def _ps3838_request(endpoint, params=None):
    """Hace una peticion GET a la API de PS3838."""
    url = f"{PS3838_BASE_URL}{endpoint}"
    response = requests.get(url, headers=_ps3838_auth_header(), params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def fetch_ps3838_odds(league):
    """
    Obtiene partidos y odds de PS3838 (Pinnacle) via endpoints bulk.

    Hace 2 llamadas API:
      1. /v3/fixtures  -> partidos programados
      2. /v3/odds      -> odds bulk (moneyline, totals, spreads)

    Args:
        league: Codigo de liga (EPL, La_Liga, Bundesliga, Serie_A, Ligue_1)

    Returns:
        (matches, odds_dict, status) o (None, None, error_msg) si falla.
    """
    league_id = get_ps3838_league_id(league)

    try:
        # --- CALL 1: Fixtures ---
        fixtures = _ps3838_request("/v3/fixtures", {
            "sportId": 29,
            "leagueIds": league_id,
        })

        if not fixtures or "league" not in fixtures:
            msg = f"PS3838: No hay fixtures para {league}"
            send_telegram_alert(msg)
            return None, None, msg

        all_open = []
        for league_data in fixtures["league"]:
            for event in league_data.get("events", []):
                if event.get("status") == "O":
                    all_open.append({
                        "id": event["id"],
                        "home_ps": event.get("home", ""),
                        "away_ps": event.get("away", ""),
                        "starts": event.get("starts", ""),
                    })

        if not all_open:
            msg = f"PS3838: No hay partidos abiertos para {league}"
            send_telegram_alert(msg)
            return None, None, msg

        # --- Filter: only next matchday (first cluster of dates within 4 days) ---
        for ev in all_open:
            try:
                ev["_dt"] = datetime.fromisoformat(ev["starts"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                ev["_dt"] = datetime.max

        all_open.sort(key=lambda e: e["_dt"])
        first_date = all_open[0]["_dt"].date()
        cutoff = first_date + timedelta(days=4)
        all_open = [e for e in all_open if e["_dt"].date() <= cutoff]

        event_names = {
            e["id"]: {"home_ps": e["home_ps"], "away_ps": e["away_ps"], "starts": e["starts"]}
            for e in all_open
        }

        # --- CALL 2: Odds bulk ---
        odds_data = _ps3838_request("/v3/odds", {
            "sportId": 29,
            "leagueIds": league_id,
            "oddsFormat": "Decimal",
        })

        if not odds_data or "leagues" not in odds_data:
            msg = f"PS3838: No hay odds para {league}"
            send_telegram_alert(msg)
            return None, None, msg

        # --- Parsear odds ---
        matches = []
        odds_dict = {}

        for league_odds in odds_data["leagues"]:
            for event in league_odds.get("events", []):
                event_id = event.get("id")
                if event_id not in event_names:
                    continue

                info = event_names[event_id]
                home = normalize_ps3838_name(info["home_ps"], league)
                away = normalize_ps3838_name(info["away_ps"], league)

                matches.append((home, away))
                match_key = f"{home} vs {away}"
                odds_dict[match_key] = {
                    "_event_id": event_id,
                    "_starts": info["starts"],
                }

                for period in event.get("periods", []):
                    if period.get("number") != 0:
                        continue

                    odds_dict[match_key]["_line_id"] = period.get("lineId")

                    # Moneyline (1X2)
                    ml = period.get("moneyline", {})
                    if ml:
                        h = ml.get("home")
                        d = ml.get("draw")
                        a = ml.get("away")
                        odds_dict[match_key]["home"] = h
                        odds_dict[match_key]["draw"] = d
                        odds_dict[match_key]["away"] = a

                        if h and d and a:
                            odds_dict[match_key]["1X"] = round(1 / (1 / h + 1 / d), 2)
                            odds_dict[match_key]["X2"] = round(1 / (1 / d + 1 / a), 2)
                            odds_dict[match_key]["12"] = round(1 / (1 / h + 1 / a), 2)

                    # Totals (Over/Under)
                    totals = period.get("totals", [])
                    odds_dict[match_key]["_totals"] = totals
                    for total in totals:
                        points = total.get("points")
                        if points is not None:
                            over_odds = total.get("over")
                            under_odds = total.get("under")
                            if over_odds:
                                odds_dict[match_key][f"over_{points}"] = over_odds
                            if under_odds:
                                odds_dict[match_key][f"under_{points}"] = under_odds

                    # Spreads (Asian Handicap)
                    spreads = period.get("spreads", [])
                    odds_dict[match_key]["_spreads"] = spreads
                    for spread in spreads:
                        hdp = spread.get("hdp")
                        if hdp is not None:
                            home_odds = spread.get("home")
                            away_odds = spread.get("away")
                            if home_odds:
                                odds_dict[match_key][f"ah_home_{hdp}"] = home_odds
                            if away_odds:
                                odds_dict[match_key][f"ah_away_{hdp}"] = away_odds

                    break  # Solo periodo 0

        status = f"OK - {len(matches)} partidos, {len(odds_dict)} con odds"
        return matches, odds_dict, status

    except requests.exceptions.HTTPError as e:
        msg = f"PS3838 HTTP Error ({league}): {e}"
        send_telegram_alert(msg)
        return None, None, msg
    except requests.exceptions.ConnectionError as e:
        msg = f"PS3838 Connection Error ({league}): {e}"
        send_telegram_alert(msg)
        return None, None, msg
    except Exception as e:
        msg = f"PS3838 Error ({league}): {e}"
        send_telegram_alert(msg)
        return None, None, msg


def get_ps3838_min_stake(event_id: int, sport_id: int, league_id: int, bet_type: str,
                         period_number: int = 0, team: str = "Team1",
                         handicap: float | None = None, side: str | None = None) -> dict:
    """
    Consulta /v2/line para obtener minRiskStake y maxRiskStake del mercado concreto.
    Devuelve: {min_stake, max_stake, price, status}
    """
    params: dict = {
        "sportId": sport_id,
        "leagueId": league_id,
        "eventId": event_id,
        "periodNumber": period_number,
        "betType": bet_type,
        "oddsFormat": "DECIMAL",
        "handicap": handicap if handicap is not None else 0,
    }
    if team:
        params["team"] = team
    if side:
        params["side"] = side
    try:
        data = _ps3838_request("/v2/line", params)
        return {
            "min_stake": data.get("minRiskStake"),
            "max_stake": data.get("maxRiskStake"),
            "price": data.get("price"),
            "status": data.get("status"),
        }
    except Exception as e:
        return {"error": str(e)}


def get_ps3838_event_lines(event_id: int, sport_id: int = 29) -> dict:
    """
    Obtiene las líneas actuales de PS3838 para un evento concreto.
    Intenta primero /v3/odds (pre-match) y si no hay resultado usa /v3/odds/inplay (live).
    Devuelve: {line_id, moneyline, spreads, totals, is_live}
    """
    result: dict = {"line_id": None, "moneyline": None, "spreads": [], "totals": [], "is_live": False}

    for endpoint, is_live in [("/v3/odds", False), ("/v3/odds/inplay", True)]:
        try:
            odds_data = _ps3838_request(endpoint, {
                "sportId": sport_id,
                "eventIds": event_id,
                "oddsFormat": "Decimal",
            })
        except Exception:
            continue
        for league_odds in odds_data.get("leagues", []):
            for event in league_odds.get("events", []):
                if event.get("id") != event_id:
                    continue
                for period in event.get("periods", []):
                    if period.get("number") != 0:
                        continue
                    result["line_id"] = period.get("lineId")
                    if period.get("moneyline"):
                        result["moneyline"] = period["moneyline"]
                    result["spreads"] = period.get("spreads", [])
                    result["totals"] = period.get("totals", [])
                    result["is_live"] = is_live
                    return result

    return result


def place_ps3838_bet(line_id: int, bet_type: str, team: str, stake: float,
                     period_number: int = 0, alt_line_id: int | None = None,
                     event_id: int | None = None, sport_id: int = 29) -> dict:
    """
    Coloca una apuesta recta en PS3838 vía API POST /v2/bets/place.
    bet_type: MONEYLINE | SPREAD | TOTAL_POINTS
    team:     Team1 | Team2 | Draw
    """
    import uuid
    payload: dict = {
        "uniqueRequestId": str(uuid.uuid4()),
        "acceptBetterLine": True,
        "fillType": "NORMAL",
        "stake": round(stake, 2),
        "winRiskStake": "RISK",
        "lineId": line_id,
        "betType": bet_type,
        "team": team,
        "periodNumber": period_number,
        "oddsFormat": "DECIMAL",
        "sportId": sport_id,
    }
    if event_id is not None:
        payload["eventId"] = event_id
    if alt_line_id is not None:
        payload["altLineId"] = alt_line_id
    url = f"{PS3838_BASE_URL}/v2/bets/place"
    headers = {**_ps3838_auth_header(), "Content-Type": "application/json"}
    print(f"[PS3838 BET PAYLOAD] {payload}")
    response = requests.post(url, headers=headers, json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


# Deportes que se incluyen en la búsqueda global (nombre → sportId PS3838)
PS3838_SEARCH_SPORTS = {
    "Fútbol": 29,
    "Baloncesto": 4,
    "Tenis": 33,
    "Béisbol": 3,
    "Hockey Hielo": 19,
    "Fútbol Americano": 15,
    "MMA": 11,
    "Voleibol": 34,
    "Cricket": 9,
    "Rugby": 23,
    "Esports": 12,
    "Bádminton": 7,
    "Snooker": 26,
    "Dardos": 28,
}
# Mapa inverso para etiquetar resultados
_SPORT_ID_TO_NAME = {v: k for k, v in PS3838_SEARCH_SPORTS.items()}


def search_ps3838_events(sport_id: int, query: str) -> list[dict]:
    """
    Búsqueda de eventos abiertos en PS3838 para un deporte concreto.
    Devuelve lista de {event_id, home, away, league, starts, sport_id, sport_name}.
    """
    if len(query) < 2:
        return []
    fixtures = _ps3838_request("/v3/fixtures", {"sportId": sport_id})
    query_lower = query.lower()
    results = []
    for league_data in fixtures.get("league", []):
        league_name = league_data.get("name", "")
        for event in league_data.get("events", []):
            if event.get("status") not in ("O", "H", "I"):
                continue
            home = event.get("home", "")
            away = event.get("away", "")
            if query_lower in home.lower() or query_lower in away.lower():
                results.append({
                    "event_id": event["id"],
                    "home": home,
                    "away": away,
                    "league": league_name,
                    "league_id": league_data.get("id"),
                    "starts": event.get("starts", ""),
                    "sport_id": sport_id,
                    "sport_name": _SPORT_ID_TO_NAME.get(sport_id, str(sport_id)),
                    "status": event.get("status", "O"),
                })
    return results


def search_ps3838_events_all(query: str) -> list[dict]:
    """
    Búsqueda libre en todos los deportes configurados en paralelo (threads).
    Devuelve hasta 25 resultados ordenados por fecha.
    """
    if len(query) < 2:
        return []
    import concurrent.futures

    def _fetch(sport_id: int) -> list[dict]:
        try:
            return search_ps3838_events(sport_id, query)
        except Exception:
            return []

    all_results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(PS3838_SEARCH_SPORTS)) as pool:
        futures = {pool.submit(_fetch, sid): sid for sid in PS3838_SEARCH_SPORTS.values()}
        for future in concurrent.futures.as_completed(futures, timeout=10):
            try:
                all_results.extend(future.result())
            except Exception:
                pass

    all_results.sort(key=lambda e: e.get("starts", ""))
    return all_results[:25]


def search_ps3838_event(league: str, home_team: str, away_team: str) -> dict | None:
    """
    Busca un partido en PS3838 por nombres de equipo (fuzzy) y devuelve sus líneas actuales.
    """
    try:
        league_id = get_ps3838_league_id(league)
    except (KeyError, ValueError):
        return None
    fixtures = _ps3838_request("/v3/fixtures", {
        "sportId": 29,
        "leagueIds": league_id,
    })
    target_id = None
    for league_data in fixtures.get("league", []):
        for event in league_data.get("events", []):
            if event.get("status") != "O":
                continue
            h = normalize_ps3838_name(event.get("home", ""), league)
            a = normalize_ps3838_name(event.get("away", ""), league)
            if (home_team.lower() in h.lower() or h.lower() in home_team.lower()) and \
               (away_team.lower() in a.lower() or a.lower() in away_team.lower()):
                target_id = event["id"]
                break
        if target_id:
            break
    if not target_id:
        return None
    lines = get_ps3838_event_lines(target_id)
    lines["event_id"] = target_id
    return lines
