# -*- coding: utf-8 -*-
"""
Integracion con PS3838 (Pinnacle) y alertas Telegram.
"""
import base64

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

        event_names = {}
        for league_data in fixtures["league"]:
            for event in league_data.get("events", []):
                if event.get("status") == "O":
                    event_names[event["id"]] = {
                        "home_ps": event.get("home", ""),
                        "away_ps": event.get("away", ""),
                        "starts": event.get("starts", ""),
                    }

        if not event_names:
            msg = f"PS3838: No hay partidos abiertos para {league}"
            send_telegram_alert(msg)
            return None, None, msg

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
