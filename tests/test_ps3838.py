"""
Test PS3838 API - Validar acceso y descubrir nombres de equipos
Ejecutar: python -m pytest tests/test_ps3838.py -v -s
"""
import os
from dotenv import load_dotenv

load_dotenv()

from trading_deportivo.team_mappings import (
    PS3838_LEAGUE_IDS, get_ps3838_map, normalize_ps3838_name
)

# --- Config ---
config = {
    "ps3838": {
        "username": os.getenv("PS3838_USERNAME"),
        "password": os.getenv("PS3838_PASSWORD"),
    }
}


def test_credentials():
    """Test 1: Verificar que las credenciales funcionan (balance)"""
    from PS3838._PS3838Customer import Customer
    print("=" * 50)
    print("TEST 1: Verificar credenciales (balance)")
    print("=" * 50)
    try:
        customer = Customer(config=config)
        balance = customer.get_balance()
        print(f"  Balance disponible: {balance}")
        assert balance is not None
    except Exception as e:
        print(f"  ERROR: {e}")
        raise


def test_fixtures(league="EPL"):
    """Test 2: Obtener proximos partidos de una liga"""
    from PS3838._PS3838Retrieve import Retrieve
    league_id = PS3838_LEAGUE_IDS[league]
    print(f"\n{'=' * 50}")
    print(f"TEST 2: Fixtures de {league} (league_id={league_id})")
    print("=" * 50)
    try:
        api = Retrieve(config=config)
        fixtures = api.get_fixtures_v3(sport_id=29, league_ids=[league_id])

        assert fixtures and "league" in fixtures, "No fixtures found"

        for league_data in fixtures["league"]:
            print(f"\n  Liga: {league_data.get('name', 'N/A')} (id={league_data['id']})")
            events = league_data.get("events", [])
            print(f"  Partidos encontrados: {len(events)}")

            for match in events[:5]:
                home = match.get("home", "?")
                away = match.get("away", "?")
                print(f"    [{match.get('status', '?')}] {home} vs {away}")

    except Exception as e:
        print(f"  ERROR: {e}")
        raise


def test_odds(league="EPL"):
    """Test 3: Obtener odds de un partido (1X2 moneyline)"""
    from PS3838._PS3838Retrieve import Retrieve
    league_id = PS3838_LEAGUE_IDS[league]
    print(f"\n{'=' * 50}")
    print(f"TEST 3: Odds 1X2 de {league}")
    print("=" * 50)
    try:
        api = Retrieve(config=config)
        fixtures = api.get_fixtures_v3(sport_id=29, league_ids=[league_id])

        assert fixtures and "league" in fixtures

        match = None
        for league_data in fixtures["league"]:
            for event in league_data.get("events", []):
                if event.get("status") == "O":
                    match = event
                    break
            if match:
                break

        assert match, "No open matches"

        print(f"  Partido: {match['home']} vs {match['away']}")

        for team_label, team_code in [("Home", "Team1"), ("Draw", "Draw"), ("Away", "Team2")]:
            odds_resp = api.get_straight_line_v2(
                league_id=league_id,
                handicap=0,
                odds_format="Decimal",
                sport_id=29,
                event_id=match["id"],
                period_number=0,
                bet_type="MONEYLINE",
                team=team_code,
            )
            price = odds_resp.get("price", "N/A")
            print(f"    {team_label}: {price}")

    except Exception as e:
        print(f"  ERROR: {e}")
        raise


def test_discover_teams():
    """Test 4: Descubrir nombres de equipos PS3838 para las 5 ligas."""
    from PS3838._PS3838Retrieve import Retrieve
    print(f"\n{'=' * 60}")
    print("TEST 4: DESCUBRIMIENTO DE EQUIPOS PS3838 (5 ligas)")
    print("=" * 60)

    api = Retrieve(config=config)

    for league, league_id in PS3838_LEAGUE_IDS.items():
        print(f"\n{'-' * 50}")
        print(f"  {league} (league_id={league_id})")
        print(f"{'-' * 50}")

        fixtures = api.get_fixtures_v3(sport_id=29, league_ids=[league_id])

        if not fixtures or "league" not in fixtures:
            print(f"    No hay fixtures para {league}")
            continue

        ps3838_teams = set()
        for league_data in fixtures["league"]:
            for event in league_data.get("events", []):
                ps3838_teams.add(event.get("home", "?"))
                ps3838_teams.add(event.get("away", "?"))

        ps3838_map = get_ps3838_map(league)

        unmapped = []
        mapped = []
        for team in sorted(ps3838_teams):
            understat_name = normalize_ps3838_name(team, league)
            if team == understat_name and team not in ps3838_map:
                unmapped.append(team)
            else:
                mapped.append((team, understat_name))

        print(f"    Equipos: {len(ps3838_teams)} | Mapeados: {len(mapped)}")

        if unmapped:
            print(f"\n    SIN MAPEAR ({len(unmapped)}):")
            for team in unmapped:
                print(f"      FALTA {team}")

        assert len(unmapped) == 0, f"Equipos sin mapear en {league}: {unmapped}"


if __name__ == "__main__":
    print("\n  PS3838 API - Test de Validacion\n")

    if not config["ps3838"]["username"]:
        print("  ERROR: Configura tus credenciales en .env primero!")
        exit(1)

    test_credentials()
    test_fixtures("EPL")
    test_odds("EPL")
    test_discover_teams()
