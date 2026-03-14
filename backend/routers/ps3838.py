# -*- coding: utf-8 -*-
"""PS3838 utility endpoints: búsqueda de eventos y líneas disponibles."""
from fastapi import APIRouter

router = APIRouter(tags=["ps3838"])

# Deportes soportados: nombre → sportId PS3838
SPORT_IDS = {
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


@router.get("/ps3838/sports")
async def ps3838_sports():
    """Devuelve el mapa de deportes disponibles con sus sportIds."""
    return [{"name": name, "sport_id": sid} for name, sid in SPORT_IDS.items()]


@router.get("/ps3838/events")
async def ps3838_search_events(q: str, sport_id: int | None = None):
    """
    Búsqueda libre de eventos PS3838 por nombre de equipo (substring).
    Si sport_id omitido → busca en todos los deportes en paralelo.
    Requiere mínimo 2 caracteres. Devuelve hasta 25 resultados.
    """
    import asyncio
    from trading_deportivo.odds import search_ps3838_events, search_ps3838_events_all
    if len(q) < 2:
        return []
    try:
        if sport_id is not None:
            return await asyncio.to_thread(search_ps3838_events, sport_id, q)
        else:
            return await asyncio.to_thread(search_ps3838_events_all, q)
    except Exception as e:
        return {"error": str(e)}


@router.get("/ps3838/lines")
async def ps3838_get_lines(event_id: int, sport_id: int = 29):
    """Devuelve las líneas actuales (1X2, AH, O/U) para un evento PS3838 concreto."""
    from trading_deportivo.odds import get_ps3838_event_lines
    try:
        lines = get_ps3838_event_lines(event_id, sport_id)
        lines["event_id"] = event_id
        lines["sport_id"] = sport_id
        return lines
    except Exception as e:
        return {"error": str(e)}


@router.get("/ps3838/search")
async def ps3838_search(league: str, home: str, away: str):
    """
    Búsqueda por liga + equipos (para apuestas del modelo).
    Mantiene compatibilidad con el flujo anterior.
    """
    from trading_deportivo.odds import search_ps3838_event
    try:
        result = search_ps3838_event(league, home, away)
        if not result:
            return {"error": f"Evento no encontrado: {home} vs {away} en {league}"}
        return result
    except Exception as e:
        return {"error": str(e)}
