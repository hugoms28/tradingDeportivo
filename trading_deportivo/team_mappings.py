# -*- coding: utf-8 -*-
"""
Mapeo de nombres de equipos entre fuentes de datos.
Understat → football-data.co.uk

TEMPORADA: 2025-26

Uso:
    from team_mappings import normalize_team_name, get_team_map

    # Normalizar un nombre (Understat → football-data)
    name = normalize_team_name("Manchester United", league="EPL")  # → "Man United"

    # Reverso (football-data → Understat)
    name = normalize_team_name("Man United", league="EPL", reverse=True)  # → "Manchester United"
"""

# =============================================================================
# PREMIER LEAGUE 2025-26 (EPL)
# =============================================================================
# Ascendidos: Leeds United, Burnley, Sunderland
# Descendidos: Leicester City, Ipswich Town, Southampton
EPL_MAP = {
    # Nombres diferentes
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nott'm Forest",
    "Leeds United": "Leeds",
    "West Ham United": "West Ham",
    # Nombres iguales: Arsenal, Aston Villa, Bournemouth, Brentford, Brighton,
    # Burnley, Chelsea, Crystal Palace, Everton, Fulham, Liverpool,
    # Sunderland, Tottenham
}

# =============================================================================
# LA LIGA 2025-26 (La_Liga)
# =============================================================================
# Ascendidos: Levante, Elche, Real Oviedo
# Descendidos: Valladolid, Las Palmas, Leganes
LA_LIGA_MAP = {
    # Nombres diferentes
    "Atletico Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao",
    "Real Betis": "Betis",
    "Celta Vigo": "Celta",
    "Deportivo Alaves": "Alaves",
    "Rayo Vallecano": "Vallecano",
    "Real Sociedad": "Sociedad",
    "Real Oviedo": "Oviedo",
    "Espanyol": "Espanol",
    # Nombres iguales: Barcelona, Real Madrid, Sevilla, Valencia, Villarreal,
    # Getafe, Osasuna, Mallorca, Girona, Levante, Elche
}

# =============================================================================
# BUNDESLIGA 2025-26 (Bundesliga)
# =============================================================================
# Ascendidos: Hamburger SV, 1. FC Koln
# Descendidos: Holstein Kiel, VfL Bochum
BUNDESLIGA_MAP = {
    # Nombres diferentes
    "Bayern Munich": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "Borussia M.Gladbach": "M'gladbach",
    "Bayer Leverkusen": "Leverkusen",
    "RasenBallsport Leipzig": "RB Leipzig",
    "RB Leipzig": "RB Leipzig",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "VfB Stuttgart": "Stuttgart",
    "VfL Wolfsburg": "Wolfsburg",
    "SC Freiburg": "Freiburg",
    "TSG Hoffenheim": "Hoffenheim",
    "1. FC Union Berlin": "Union Berlin",
    "FC Union Berlin": "Union Berlin",
    "1. FC Koln": "FC Koln",
    "FC Koln": "FC Koln",
    "Werder Bremen": "Werder Bremen",
    "FC Augsburg": "Augsburg",
    "1. FSV Mainz 05": "Mainz",
    "Mainz 05": "Mainz",
    "FC Heidenheim": "Heidenheim",
    "St. Pauli": "St Pauli",
    "Hamburger SV": "Hamburg",
    "HSV": "Hamburg",
}

# =============================================================================
# SERIE A 2025-26 (Serie_A)
# =============================================================================
# Ascendidos: Pisa, Spezia, Cesena/Bari (playoff)
# Descendidos: Monza, Venezia, Empoli
SERIE_A_MAP = {
    # Nombres diferentes
    "AC Milan": "Milan",
    "Inter": "Inter",
    "Internazionale": "Inter",
    "Roma": "Roma",
    "AS Roma": "Roma",
    "Hellas Verona": "Verona",
    "Parma Calcio 1913": "Parma",
    "Cremonese": "Cremonese",
    "Sassuolo": "Sassuolo",
    # Nombres iguales: Juventus, Napoli, Lazio, Atalanta, Fiorentina,
    # Bologna, Torino, Udinese, Lecce, Genoa, Cagliari, Como, Pisa, Spezia
}

# =============================================================================
# LIGUE 1 2025-26 (Ligue_1)
# =============================================================================
# Ascendidos: Lorient, Paris FC, Metz
# Descendidos: Montpellier, Saint-Etienne, Reims
LIGUE_1_MAP = {
    # Nombres diferentes
    "Paris Saint Germain": "Paris SG",
    "Paris Saint-Germain": "Paris SG",
    "Olympique Marseille": "Marseille",
    "Marseille": "Marseille",
    "Olympique Lyon": "Lyon",
    "Lyon": "Lyon",
    "AS Monaco": "Monaco",
    "Monaco": "Monaco",
    "LOSC Lille": "Lille",
    "Lille": "Lille",
    "Stade Rennais": "Rennes",
    "Rennes": "Rennes",
    "OGC Nice": "Nice",
    "Nice": "Nice",
    "RC Lens": "Lens",
    "Lens": "Lens",
    "FC Nantes": "Nantes",
    "Nantes": "Nantes",
    "Stade Brestois 29": "Brest",
    "Brest": "Brest",
    "RC Strasbourg": "Strasbourg",
    "Strasbourg": "Strasbourg",
    "FC Lorient": "Lorient",
    "Lorient": "Lorient",
    "Le Havre": "Le Havre",
    "AJ Auxerre": "Auxerre",
    "Auxerre": "Auxerre",
    "Angers SCO": "Angers",
    "Angers": "Angers",
    "Paris FC": "Paris FC",
    "FC Metz": "Metz",
    "Metz": "Metz",
    # Nombres iguales: Toulouse
}

# =============================================================================
# MAPEO PRINCIPAL POR LIGA (Understat → football-data)
# =============================================================================
LEAGUE_MAPS = {
    "EPL": EPL_MAP,
    "La_Liga": LA_LIGA_MAP,
    "Bundesliga": BUNDESLIGA_MAP,
    "Serie_A": SERIE_A_MAP,
    "Ligue_1": LIGUE_1_MAP,
}


# =============================================================================
# THE-ODDS-API → UNDERSTAT (para obtener odds en vivo)
# =============================================================================

# Premier League 2025-26
ODDS_API_EPL_MAP = {
    "Manchester United": "Manchester United",
    "Manchester City": "Manchester City",
    "Liverpool": "Liverpool",
    "Arsenal": "Arsenal",
    "Chelsea": "Chelsea",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle United",
    "Brighton and Hove Albion": "Brighton",
    "Aston Villa": "Aston Villa",
    "West Ham United": "West Ham",
    "Brentford": "Brentford",
    "Fulham": "Fulham",
    "Crystal Palace": "Crystal Palace",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Everton": "Everton",
    "Nottingham Forest": "Nottingham Forest",
    "AFC Bournemouth": "Bournemouth",
    "Leeds United": "Leeds",
    "Burnley": "Burnley",
    "Sunderland AFC": "Sunderland",
}

# La Liga 2025-26
ODDS_API_LA_LIGA_MAP = {
    "Barcelona": "Barcelona",
    "Real Madrid": "Real Madrid",
    "Atletico Madrid": "Atletico Madrid",
    "Atlético Madrid": "Atletico Madrid",
    "Athletic Bilbao": "Athletic Club",
    "Real Sociedad": "Real Sociedad",
    "Real Betis": "Real Betis",
    "Villarreal": "Villarreal",
    "Sevilla": "Sevilla",
    "Valencia": "Valencia",
    "Celta Vigo": "Celta Vigo",
    "Osasuna": "Osasuna",
    "CA Osasuna": "Osasuna",
    "Getafe": "Getafe",
    "Rayo Vallecano": "Rayo Vallecano",
    "Mallorca": "Mallorca",
    "Girona": "Girona",
    "Alaves": "Alaves",
    "Alavés": "Alaves",
    "Deportivo Alaves": "Alaves",
    "Espanyol": "Espanyol",
    "Levante": "Levante",
    "Elche": "Elche",
    "Elche CF": "Elche",
    "Real Oviedo": "Real Oviedo",
    "Oviedo": "Real Oviedo",
}

# Bundesliga 2025-26
# Understat names: Augsburg, Bayer Leverkusen, Bayern Munich, Borussia Dortmund,
# Borussia M.Gladbach, Eintracht Frankfurt, FC Cologne, FC Heidenheim, Freiburg,
# Hamburger SV, Hoffenheim, Mainz 05, RasenBallsport Leipzig, St. Pauli,
# Union Berlin, VfB Stuttgart, Werder Bremen, Wolfsburg
ODDS_API_BUNDESLIGA_MAP = {
    "Bayern Munich": "Bayern Munich",
    "Borussia Dortmund": "Borussia Dortmund",
    "Bayer Leverkusen": "Bayer Leverkusen",
    "RB Leipzig": "RasenBallsport Leipzig",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "VfB Stuttgart": "VfB Stuttgart",
    "VfL Wolfsburg": "Wolfsburg",
    "Borussia Monchengladbach": "Borussia M.Gladbach",
    "SC Freiburg": "Freiburg",
    "TSG Hoffenheim": "Hoffenheim",
    "FC Union Berlin": "Union Berlin",
    "Werder Bremen": "Werder Bremen",
    "FC Augsburg": "Augsburg",
    "FSV Mainz 05": "Mainz 05",
    "1. FC Heidenheim": "FC Heidenheim",
    "St Pauli": "St. Pauli",
    "FC St. Pauli": "St. Pauli",
    "Hamburger SV": "Hamburger SV",
    "1. FC Köln": "FC Cologne",
    "FC Koln": "FC Cologne",
}

# Serie A 2025-26
ODDS_API_SERIE_A_MAP = {
    "Napoli": "Napoli",
    "Inter Milan": "Inter",
    "Internazionale": "Inter",
    "AC Milan": "AC Milan",
    "Juventus": "Juventus",
    "AS Roma": "AS Roma",
    "Roma": "AS Roma",
    "Lazio": "Lazio",
    "Atalanta": "Atalanta",
    "Fiorentina": "Fiorentina",
    "Bologna": "Bologna",
    "Torino": "Torino",
    "Udinese": "Udinese",
    "Genoa": "Genoa",
    "Cagliari": "Cagliari",
    "Lecce": "Lecce",
    "Como": "Como",
    "Parma": "Parma",
    "Hellas Verona": "Hellas Verona",
    "Verona": "Hellas Verona",
    "Pisa": "Pisa",
    "Spezia": "Spezia",
}

# Ligue 1 2025-26
ODDS_API_LIGUE_1_MAP = {
    "Paris Saint-Germain": "Paris Saint Germain",
    "Paris Saint Germain": "Paris Saint Germain",
    "PSG": "Paris Saint Germain",
    "Marseille": "Marseille",
    "Olympique Marseille": "Olympique Marseille",
    "Monaco": "Monaco",
    "AS Monaco": "AS Monaco",
    "Lyon": "Lyon",
    "Olympique Lyon": "Olympique Lyon",
    "Olympique Lyonnais": "Olympique Lyon",
    "Lille": "Lille",
    "LOSC Lille": "LOSC Lille",
    "Nice": "Nice",
    "OGC Nice": "OGC Nice",
    "Lens": "Lens",
    "RC Lens": "RC Lens",
    "Rennes": "Rennes",
    "Stade Rennais": "Stade Rennais",
    "Nantes": "Nantes",
    "FC Nantes": "FC Nantes",
    "Strasbourg": "Strasbourg",
    "RC Strasbourg": "RC Strasbourg",
    "Brest": "Brest",
    "Stade Brestois": "Stade Brestois 29",
    "Toulouse": "Toulouse",
    "Auxerre": "Auxerre",
    "AJ Auxerre": "AJ Auxerre",
    "Angers": "Angers",
    "Angers SCO": "Angers SCO",
    "Le Havre": "Le Havre",
    "Lorient": "Lorient",
    "FC Lorient": "FC Lorient",
    "Paris FC": "Paris FC",
    "Metz": "Metz",
    "FC Metz": "FC Metz",
}

# Mapeo principal the-odds-api
ODDS_API_MAPS = {
    "EPL": ODDS_API_EPL_MAP,
    "La_Liga": ODDS_API_LA_LIGA_MAP,
    "Bundesliga": ODDS_API_BUNDESLIGA_MAP,
    "Serie_A": ODDS_API_SERIE_A_MAP,
    "Ligue_1": ODDS_API_LIGUE_1_MAP,
}

# Codigos de liga para the-odds-api
ODDS_API_SPORT_KEYS = {
    "EPL": "soccer_epl",
    "La_Liga": "soccer_spain_la_liga",
    "Bundesliga": "soccer_germany_bundesliga",
    "Serie_A": "soccer_italy_serie_a",
    "Ligue_1": "soccer_france_ligue_one",
}

# =============================================================================
# PS3838 (PINNACLE) → UNDERSTAT
# =============================================================================

# League IDs de PS3838 para nuestras ligas
PS3838_LEAGUE_IDS = {
    "EPL": 1980,
    "La_Liga": 2196,
    "Bundesliga": 1842,
    "Serie_A": 2436,
    "Ligue_1": 2036,
}

# Premier League 2025-26 (PS3838 → Understat)
PS3838_EPL_MAP = {
    "Manchester United": "Manchester United",
    "Manchester City": "Manchester City",
    "Liverpool": "Liverpool",
    "Arsenal": "Arsenal",
    "Chelsea": "Chelsea",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle United",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Brighton": "Brighton",
    "Aston Villa": "Aston Villa",
    "West Ham United": "West Ham",
    "Brentford": "Brentford",
    "Fulham": "Fulham",
    "Crystal Palace": "Crystal Palace",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Everton": "Everton",
    "Nottingham Forest": "Nottingham Forest",
    "AFC Bournemouth": "Bournemouth",
    "Bournemouth": "Bournemouth",
    "Leeds United": "Leeds",
    "Leeds": "Leeds",
    "Burnley": "Burnley",
    "Sunderland": "Sunderland",
    "Sunderland AFC": "Sunderland",
}

# La Liga 2025-26 (PS3838 → Understat)
PS3838_LA_LIGA_MAP = {
    "Barcelona": "Barcelona",
    "FC Barcelona": "Barcelona",
    "Real Madrid": "Real Madrid",
    "Atletico Madrid": "Atletico Madrid",
    "Atletico de Madrid": "Atletico Madrid",
    "Athletic Bilbao": "Athletic Club",
    "Athletic Club": "Athletic Club",
    "Real Sociedad": "Real Sociedad",
    "Real Betis": "Real Betis",
    "Villarreal": "Villarreal",
    "Villarreal CF": "Villarreal",
    "Sevilla": "Sevilla",
    "Sevilla FC": "Sevilla",
    "Valencia": "Valencia",
    "Valencia CF": "Valencia",
    "Celta Vigo": "Celta Vigo",
    "Celta de Vigo": "Celta Vigo",
    "RC Celta": "Celta Vigo",
    "Osasuna": "Osasuna",
    "CA Osasuna": "Osasuna",
    "Getafe": "Getafe",
    "Getafe CF": "Getafe",
    "Rayo Vallecano": "Rayo Vallecano",
    "Mallorca": "Mallorca",
    "RCD Mallorca": "Mallorca",
    "Girona": "Girona",
    "Girona FC": "Girona",
    "Deportivo Alaves": "Alaves",
    "Alaves": "Alaves",
    "Espanyol": "Espanyol",
    "RCD Espanyol": "Espanyol",
    "Levante": "Levante",
    "Levante UD": "Levante",
    "Elche": "Elche",
    "Elche CF": "Elche",
    "Real Oviedo": "Real Oviedo",
}

# Bundesliga 2025-26 (PS3838 → Understat)
PS3838_BUNDESLIGA_MAP = {
    "Bayern Munich": "Bayern Munich",
    "FC Bayern Munich": "Bayern Munich",
    "Bayern Munchen": "Bayern Munich",
    "Borussia Dortmund": "Borussia Dortmund",
    "Bayer Leverkusen": "Bayer Leverkusen",
    "RB Leipzig": "RasenBallsport Leipzig",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "VfB Stuttgart": "VfB Stuttgart",
    "Stuttgart": "VfB Stuttgart",
    "VfL Wolfsburg": "Wolfsburg",
    "Wolfsburg": "Wolfsburg",
    "Borussia Monchengladbach": "Borussia M.Gladbach",
    "Borussia M'gladbach": "Borussia M.Gladbach",
    "Monchengladbach": "Borussia M.Gladbach",
    "SC Freiburg": "Freiburg",
    "Freiburg": "Freiburg",
    "TSG Hoffenheim": "Hoffenheim",
    "Hoffenheim": "Hoffenheim",
    "1. FC Union Berlin": "Union Berlin",
    "FC Union Berlin": "Union Berlin",
    "Union Berlin": "Union Berlin",
    "Werder Bremen": "Werder Bremen",
    "FC Augsburg": "Augsburg",
    "Augsburg": "Augsburg",
    "1. FSV Mainz 05": "Mainz 05",
    "FSV Mainz 05": "Mainz 05",
    "Mainz 05": "Mainz 05",
    "Mainz": "Mainz 05",
    "1. FC Heidenheim": "FC Heidenheim",
    "FC Heidenheim": "FC Heidenheim",
    "Heidenheim": "FC Heidenheim",
    "FC St. Pauli": "St. Pauli",
    "St. Pauli": "St. Pauli",
    "St Pauli": "St. Pauli",
    "Hamburger SV": "Hamburger SV",
    "Hamburg": "Hamburger SV",
    "1. FC Koln": "FC Cologne",
    "FC Koln": "FC Cologne",
    "1. FC Cologne": "FC Cologne",
}

# Serie A 2025-26 (PS3838 → Understat)
PS3838_SERIE_A_MAP = {
    "Napoli": "Napoli",
    "SSC Napoli": "Napoli",
    "Inter Milan": "Inter",
    "Internazionale": "Inter",
    "FC Internazionale": "Inter",
    "Inter": "Inter",
    "AC Milan": "AC Milan",
    "Milan": "AC Milan",
    "Juventus": "Juventus",
    "AS Roma": "Roma",
    "Roma": "Roma",
    "Lazio": "Lazio",
    "SS Lazio": "Lazio",
    "Atalanta": "Atalanta",
    "Atalanta BC": "Atalanta",
    "Fiorentina": "Fiorentina",
    "ACF Fiorentina": "Fiorentina",
    "Bologna": "Bologna",
    "Bologna FC": "Bologna",
    "Torino": "Torino",
    "Torino FC": "Torino",
    "Udinese": "Udinese",
    "Udinese Calcio": "Udinese",
    "Genoa": "Genoa",
    "Genoa CFC": "Genoa",
    "Cagliari": "Cagliari",
    "Cagliari Calcio": "Cagliari",
    "Lecce": "Lecce",
    "US Lecce": "Lecce",
    "Como": "Como",
    "Como 1907": "Como",
    "Parma": "Parma Calcio 1913",
    "Parma Calcio": "Parma Calcio 1913",
    "Parma Calcio 1913": "Parma Calcio 1913",
    "Hellas Verona": "Hellas Verona",
    "Verona": "Hellas Verona",
    "Hellas Verona FC": "Hellas Verona",
    "Pisa": "Pisa",
    "AC Pisa": "Pisa",
    "Spezia": "Spezia",
    "Spezia Calcio": "Spezia",
    "Cremonese": "Cremonese",
    "US Cremonese": "Cremonese",
    "Sassuolo": "Sassuolo",
    "US Sassuolo": "Sassuolo",
}

# Ligue 1 2025-26 (PS3838 → Understat)
PS3838_LIGUE_1_MAP = {
    "Paris Saint-Germain": "Paris Saint Germain",
    "Paris Saint Germain": "Paris Saint Germain",
    "Paris SG": "Paris Saint Germain",
    "PSG": "Paris Saint Germain",
    "Marseille": "Marseille",
    "Olympique Marseille": "Olympique Marseille",
    "Olympique de Marseille": "Olympique Marseille",
    "Monaco": "Monaco",
    "AS Monaco": "AS Monaco",
    "Lyon": "Lyon",
    "Olympique Lyon": "Olympique Lyon",
    "Olympique Lyonnais": "Olympique Lyon",
    "Lille": "Lille",
    "LOSC Lille": "LOSC Lille",
    "Lille OSC": "LOSC Lille",
    "Nice": "Nice",
    "OGC Nice": "OGC Nice",
    "Lens": "Lens",
    "RC Lens": "RC Lens",
    "Rennes": "Rennes",
    "Stade Rennais": "Stade Rennais",
    "Stade Rennais FC": "Stade Rennais",
    "Nantes": "Nantes",
    "FC Nantes": "FC Nantes",
    "Strasbourg": "Strasbourg",
    "RC Strasbourg": "RC Strasbourg",
    "RC Strasbourg Alsace": "RC Strasbourg",
    "Brest": "Brest",
    "Stade Brestois": "Stade Brestois 29",
    "Stade Brestois 29": "Stade Brestois 29",
    "Toulouse": "Toulouse",
    "Toulouse FC": "Toulouse",
    "Auxerre": "Auxerre",
    "AJ Auxerre": "AJ Auxerre",
    "Angers": "Angers",
    "Angers SCO": "Angers SCO",
    "Le Havre": "Le Havre",
    "Le Havre AC": "Le Havre",
    "Lorient": "Lorient",
    "FC Lorient": "FC Lorient",
    "Paris FC": "Paris FC",
    "Metz": "Metz",
    "FC Metz": "FC Metz",
}

# Mapeo principal PS3838
PS3838_MAPS = {
    "EPL": PS3838_EPL_MAP,
    "La_Liga": PS3838_LA_LIGA_MAP,
    "Bundesliga": PS3838_BUNDESLIGA_MAP,
    "Serie_A": PS3838_SERIE_A_MAP,
    "Ligue_1": PS3838_LIGUE_1_MAP,
}


# Codigos de liga para football-data.co.uk
FOOTBALL_DATA_CODES = {
    "EPL": "E0",
    "La_Liga": "SP1",
    "Bundesliga": "D1",
    "Serie_A": "I1",
    "Ligue_1": "F1",
}


def get_football_data_code(league: str) -> str:
    """
    Obtiene el codigo de liga para football-data.co.uk.

    Args:
        league: Codigo de liga (EPL, La_Liga, Bundesliga, Serie_A, Ligue_1)

    Returns:
        str con el codigo para la URL (E0, SP1, D1, I1, F1)
    """
    if league not in FOOTBALL_DATA_CODES:
        raise ValueError(f"Liga '{league}' no soportada. Opciones: {list(FOOTBALL_DATA_CODES.keys())}")
    return FOOTBALL_DATA_CODES[league]


def get_team_map(league: str) -> dict:
    """
    Obtiene el mapeo de equipos para una liga especifica.

    Args:
        league: Codigo de liga (EPL, La_Liga, Bundesliga, Serie_A, Ligue_1)

    Returns:
        dict con mapeo Understat → football-data
    """
    if league not in LEAGUE_MAPS:
        raise ValueError(f"Liga '{league}' no soportada. Opciones: {list(LEAGUE_MAPS.keys())}")
    return LEAGUE_MAPS[league]


def get_odds_api_map(league: str) -> dict:
    """
    Obtiene el mapeo de equipos para the-odds-api.

    Args:
        league: Codigo de liga (EPL, La_Liga, Bundesliga, Serie_A, Ligue_1)

    Returns:
        dict con mapeo the-odds-api → Understat
    """
    if league not in ODDS_API_MAPS:
        raise ValueError(f"Liga '{league}' no soportada. Opciones: {list(ODDS_API_MAPS.keys())}")
    return ODDS_API_MAPS[league]


def get_odds_api_sport_key(league: str) -> str:
    """
    Obtiene el codigo de deporte para the-odds-api.

    Args:
        league: Codigo de liga (EPL, La_Liga, Bundesliga, Serie_A, Ligue_1)

    Returns:
        str con el sport_key para la API
    """
    if league not in ODDS_API_SPORT_KEYS:
        raise ValueError(f"Liga '{league}' no soportada. Opciones: {list(ODDS_API_SPORT_KEYS.keys())}")
    return ODDS_API_SPORT_KEYS[league]


def normalize_odds_api_name(name: str, league: str = "EPL") -> str:
    """
    Convierte nombre de the-odds-api a nombre de Understat.

    Args:
        name: Nombre del equipo en the-odds-api
        league: Codigo de liga

    Returns:
        Nombre normalizado para Understat

    Examples:
        >>> normalize_odds_api_name("Tottenham Hotspur", "EPL")
        'Tottenham'
        >>> normalize_odds_api_name("Inter Milan", "Serie_A")
        'Inter'
    """
    odds_map = get_odds_api_map(league)
    return odds_map.get(name, name)


def get_ps3838_league_id(league: str) -> int:
    """
    Obtiene el league ID de PS3838 para una liga.

    Args:
        league: Codigo de liga (EPL, La_Liga, Bundesliga, Serie_A, Ligue_1)

    Returns:
        int con el league_id para la API de PS3838
    """
    if league not in PS3838_LEAGUE_IDS:
        raise ValueError(f"Liga '{league}' no soportada. Opciones: {list(PS3838_LEAGUE_IDS.keys())}")
    return PS3838_LEAGUE_IDS[league]


def get_ps3838_map(league: str) -> dict:
    """
    Obtiene el mapeo de equipos para PS3838.

    Args:
        league: Codigo de liga (EPL, La_Liga, Bundesliga, Serie_A, Ligue_1)

    Returns:
        dict con mapeo PS3838 → Understat
    """
    if league not in PS3838_MAPS:
        raise ValueError(f"Liga '{league}' no soportada. Opciones: {list(PS3838_MAPS.keys())}")
    return PS3838_MAPS[league]


def normalize_ps3838_name(name: str, league: str = "EPL") -> str:
    """
    Convierte nombre de PS3838 a nombre de Understat.

    Args:
        name: Nombre del equipo en PS3838
        league: Codigo de liga

    Returns:
        Nombre normalizado para Understat

    Examples:
        >>> normalize_ps3838_name("Tottenham Hotspur", "EPL")
        'Tottenham'
        >>> normalize_ps3838_name("Inter Milan", "Serie_A")
        'Inter'
    """
    ps3838_map = get_ps3838_map(league)
    return ps3838_map.get(name, name)


def normalize_team_name(name: str, league: str = "EPL", reverse: bool = False) -> str:
    """
    Normaliza nombres de equipos entre fuentes.

    Args:
        name: Nombre del equipo a normalizar
        league: Codigo de liga (EPL, La_Liga, Bundesliga, Serie_A, Ligue_1)
        reverse: Si True, convierte football-data → Understat
                 Si False (default), convierte Understat → football-data

    Returns:
        Nombre normalizado

    Examples:
        >>> normalize_team_name("Manchester United", "EPL")
        'Man United'
        >>> normalize_team_name("Man United", "EPL", reverse=True)
        'Manchester United'
        >>> normalize_team_name("Borussia Dortmund", "Bundesliga")
        'Dortmund'
    """
    team_map = get_team_map(league)

    if reverse:
        # football-data → Understat
        reverse_map = {v: k for k, v in team_map.items()}
        return reverse_map.get(name, name)
    else:
        # Understat → football-data
        return team_map.get(name, name)


def list_teams(league: str) -> None:
    """
    Muestra todos los equipos mapeados para una liga.

    Args:
        league: Codigo de liga
    """
    team_map = get_team_map(league)
    print(f"\n{'='*50}")
    print(f"MAPEO DE EQUIPOS: {league} (2025-26)")
    print(f"{'='*50}")
    print(f"{'Understat':<30} -> {'football-data':<20}")
    print("-" * 50)
    for understat, fd in sorted(team_map.items()):
        if understat != fd:
            print(f"{understat:<30} -> {fd:<20}")
    print(f"\nTotal equipos con mapeo diferente: {sum(1 for k, v in team_map.items() if k != v)}")


def validate_mapping(df_understat, df_odds, league: str) -> dict:
    """
    Valida el mapeo comparando equipos de ambas fuentes.
    Util para detectar equipos que faltan en el mapeo.

    Args:
        df_understat: DataFrame con columnas home_team, away_team
        df_odds: DataFrame con columnas HomeTeam, AwayTeam
        league: Codigo de liga

    Returns:
        dict con equipos no mapeados
    """
    # Equipos unicos de cada fuente
    understat_teams = set(df_understat["home_team"].unique()) | set(df_understat["away_team"].unique())
    odds_teams = set(df_odds["HomeTeam"].unique()) | set(df_odds["AwayTeam"].unique())

    # Normalizar nombres de Understat
    normalized_understat = {normalize_team_name(t, league) for t in understat_teams}

    # Encontrar discrepancias
    missing_in_odds = normalized_understat - odds_teams
    missing_in_understat = odds_teams - normalized_understat

    if missing_in_odds or missing_in_understat:
        print(f"\n{'='*50}")
        print(f"VALIDACION DE MAPEO: {league}")
        print(f"{'='*50}")

        if missing_in_odds:
            print(f"\nEquipos de Understat NO encontrados en odds:")
            for team in sorted(missing_in_odds):
                # Buscar nombre original
                original = [t for t in understat_teams if normalize_team_name(t, league) == team]
                print(f"  - {team} (original: {original[0] if original else '?'})")

        if missing_in_understat:
            print(f"\nEquipos de odds NO encontrados en Understat:")
            for team in sorted(missing_in_understat):
                print(f"  - {team}")
    else:
        print(f"OK Mapeo OK para {league}: todos los equipos coinciden")

    return {
        "missing_in_odds": missing_in_odds,
        "missing_in_understat": missing_in_understat
    }


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TEAM MAPPINGS - TEMPORADA 2025-26")
    print("=" * 60)

    # Test basico Understat -> football-data
    print("\n[1] UNDERSTAT -> FOOTBALL-DATA:")
    print(f"  Manchester United (EPL) -> {normalize_team_name('Manchester United', 'EPL')}")
    print(f"  Borussia Dortmund (Bundesliga) -> {normalize_team_name('Borussia Dortmund', 'Bundesliga')}")
    print(f"  Paris Saint Germain (Ligue_1) -> {normalize_team_name('Paris Saint Germain', 'Ligue_1')}")
    print(f"  Atletico Madrid (La_Liga) -> {normalize_team_name('Atletico Madrid', 'La_Liga')}")
    print(f"  AC Milan (Serie_A) -> {normalize_team_name('AC Milan', 'Serie_A')}")

    # Equipos nuevos 2025-26
    print("\n[2] EQUIPOS ASCENDIDOS 2025-26:")
    print(f"  Leeds United (EPL) -> {normalize_team_name('Leeds United', 'EPL')}")
    print(f"  Sunderland (EPL) -> {normalize_team_name('Sunderland', 'EPL')}")
    print(f"  Hamburger SV (Bundesliga) -> {normalize_team_name('Hamburger SV', 'Bundesliga')}")
    print(f"  Real Oviedo (La_Liga) -> {normalize_team_name('Real Oviedo', 'La_Liga')}")
    print(f"  Paris FC (Ligue_1) -> {normalize_team_name('Paris FC', 'Ligue_1')}")

    # Test the-odds-api -> Understat
    print("\n[3] THE-ODDS-API -> UNDERSTAT:")
    print(f"  Tottenham Hotspur (EPL) -> {normalize_odds_api_name('Tottenham Hotspur', 'EPL')}")
    print(f"  Brighton and Hove Albion (EPL) -> {normalize_odds_api_name('Brighton and Hove Albion', 'EPL')}")
    print(f"  Inter Milan (Serie_A) -> {normalize_odds_api_name('Inter Milan', 'Serie_A')}")
    print(f"  Paris Saint-Germain (Ligue_1) -> {normalize_odds_api_name('Paris Saint-Germain', 'Ligue_1')}")
    print(f"  Athletic Bilbao (La_Liga) -> {normalize_odds_api_name('Athletic Bilbao', 'La_Liga')}")

    # Sport keys
    print("\n[4] SPORT KEYS PARA THE-ODDS-API:")
    for league in ODDS_API_SPORT_KEYS:
        print(f"  {league}: {get_odds_api_sport_key(league)}")

    # Test PS3838 -> Understat
    print("\n[5] PS3838 -> UNDERSTAT:")
    print(f"  Tottenham Hotspur (EPL) -> {normalize_ps3838_name('Tottenham Hotspur', 'EPL')}")
    print(f"  Brighton & Hove Albion (EPL) -> {normalize_ps3838_name('Brighton & Hove Albion', 'EPL')}")
    print(f"  Inter Milan (Serie_A) -> {normalize_ps3838_name('Inter Milan', 'Serie_A')}")
    print(f"  Paris Saint-Germain (Ligue_1) -> {normalize_ps3838_name('Paris Saint-Germain', 'Ligue_1')}")
    print(f"  Athletic Bilbao (La_Liga) -> {normalize_ps3838_name('Athletic Bilbao', 'La_Liga')}")
    print(f"  RB Leipzig (Bundesliga) -> {normalize_ps3838_name('RB Leipzig', 'Bundesliga')}")

    # PS3838 league IDs
    print("\n[6] PS3838 LEAGUE IDS:")
    for league in PS3838_LEAGUE_IDS:
        print(f"  {league}: {get_ps3838_league_id(league)}")

    # Listar mapeos Understat -> football-data
    print("\n" + "=" * 60)
    print("MAPEOS UNDERSTAT -> FOOTBALL-DATA")
    print("=" * 60)
    for league in LEAGUE_MAPS.keys():
        list_teams(league)
