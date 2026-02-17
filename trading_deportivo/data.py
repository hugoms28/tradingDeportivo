# -*- coding: utf-8 -*-
"""
Carga de datos desde Understat, sistema de cache y procesamiento de tiros.
"""
import os
import json as json_lib
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from understatapi import UnderstatClient
from tqdm.auto import tqdm
from time import sleep

from .config import CACHE_DIR, CACHE_TTL_HOURS


# =============================================================================
# CACHE
# =============================================================================

def get_cache_path(league, season, data_type):
    """Genera path del archivo de cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    season_str = "_".join(season) if isinstance(season, list) else season
    return os.path.join(CACHE_DIR, f"{league}_{season_str}_{data_type}.json")


def is_cache_valid(filepath, ttl_hours=CACHE_TTL_HOURS):
    """Verifica si el cache existe y no ha expirado."""
    if not os.path.exists(filepath):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
    return datetime.now() - mtime < timedelta(hours=ttl_hours)


def save_cache(data, filepath):
    """Guarda datos en cache."""
    with open(filepath, "w", encoding="utf-8") as f:
        json_lib.dump(data, f)


def load_cache(filepath):
    """Carga datos del cache."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json_lib.load(f)


def clear_cache(league=None):
    """Limpia el cache (todo o de una liga especifica)."""
    if not os.path.exists(CACHE_DIR):
        print("No hay cache")
        return

    files = os.listdir(CACHE_DIR)
    if league:
        files = [f for f in files if f.startswith(league)]

    for f in files:
        os.remove(os.path.join(CACHE_DIR, f))
        print(f"  Eliminado: {f}")

    print(f"Cache limpiado: {len(files)} archivos")


# =============================================================================
# UNDERSTAT DATA
# =============================================================================

def get_league_match_ids(league, seasons, use_cache=False):
    """Retorna lista de match_ids y metadatos por liga y temporadas."""
    cache_path = get_cache_path(league, seasons, "matches")

    if use_cache and is_cache_valid(cache_path):
        print(f"[CACHE] Cargando partidos desde cache...")
        data = load_cache(cache_path)
        return data["match_ids"], data["matches"]

    print(f"[API] Descargando partidos de Understat...")
    all_match_ids = []
    all_matches = []
    with UnderstatClient() as us:
        for season in seasons:
            matches = us.league(league=league).get_match_data(season=season)
            all_match_ids.extend([m["id"] for m in matches])
            all_matches.extend(matches)

    if use_cache:
        save_cache({"match_ids": all_match_ids, "matches": all_matches}, cache_path)
        print(f"[CACHE] Guardado en {cache_path}")

    return all_match_ids, all_matches


def fetch_match_shots(match_id, client):
    """Obtiene tiros de un partido."""
    out = []
    md = client.match(match=match_id).get_shot_data()
    for side in ("h", "a"):
        for s in md.get(side, []):
            s = dict(s)
            s["h_a"] = side
            s["match_id"] = match_id
            out.append(s)
    return out


def fetch_all_shots(match_ids, league, seasons, use_cache=False):
    """Descarga todos los tiros con cache."""
    cache_path = get_cache_path(league, seasons, "shots")

    if use_cache and is_cache_valid(cache_path):
        print(f"[CACHE] Cargando tiros desde cache...")
        return load_cache(cache_path)

    print(f"[API] Descargando tiros de {len(match_ids)} partidos...")
    all_shots = []
    with UnderstatClient() as us:
        for mid in tqdm(match_ids, desc=f"Descargando tiros {league}"):
            try:
                all_shots.extend(fetch_match_shots(mid, us))
            except Exception:
                pass
            sleep(0.15)

    if use_cache:
        save_cache(all_shots, cache_path)
        print(f"[CACHE] Guardado en {cache_path}")

    return all_shots


# =============================================================================
# PROCESAMIENTO
# =============================================================================

def shots_to_df(shots):
    """Convierte lista de tiros a DataFrame, incluyendo xG de Understat."""
    rows = []
    for s in shots:
        rows.append({
            "match_id": s.get("match_id"),
            "team": s.get("h_team") if s.get("h_a") == "h" else s.get("a_team"),
            "h_a": s.get("h_a"),
            "X": float(s.get("X")),
            "Y": float(s.get("Y")),
            "is_goal": 1 if s.get("result") == "Goal" else 0,
            "situation": s.get("situation"),
            "xg_understat": float(s.get("xG", 0))
        })
    df = pd.DataFrame(rows)
    return df.dropna(subset=["X", "Y"]).reset_index(drop=True)


def build_match_xg_matrix(df, raw_matches=None):
    """
    Construye matriz de xG por partido, opcionalmente con fechas.

    Args:
        df: DataFrame de tiros con columnas [match_id, team, venue, xg]
        raw_matches: Lista opcional de partidos raw para extraer fechas

    Returns:
        DataFrame con [match_id, home_team, away_team, home_xg, away_xg, datetime*]
    """
    agg = (
        df.groupby(["match_id", "team", "venue"], observed=True, as_index=False)
        .agg(xg=("xg", "sum"))
    )
    home = agg[agg["venue"] == "home"][["match_id", "team", "xg"]].rename(
        columns={"team": "home_team", "xg": "home_xg"}
    )
    away = agg[agg["venue"] == "away"][["match_id", "team", "xg"]].rename(
        columns={"team": "away_team", "xg": "away_xg"}
    )
    result = home.merge(away, on="match_id", how="inner")

    if raw_matches is not None:
        match_dates = {}
        for m in raw_matches:
            if m.get("isResult"):
                match_id = str(m.get("id"))
                datetime_str = m.get("datetime", "")
                if datetime_str:
                    match_dates[match_id] = datetime_str

        result["datetime"] = result["match_id"].map(match_dates)
        result["datetime"] = pd.to_datetime(result["datetime"])
        result = result.dropna(subset=["datetime"])

    return result


def build_matches_with_dates(raw_matches, df_shots):
    """
    Combina datos de partidos (fechas, goles) con xG calculado.
    """
    matches = []
    for m in raw_matches:
        if not m.get("isResult"):
            continue

        match_id = str(m.get("id"))
        home_team = m.get("h", {}).get("title")
        away_team = m.get("a", {}).get("title")

        if not home_team or not away_team:
            continue

        datetime_str = m.get("datetime", "")

        matches.append({
            "match_id": match_id,
            "datetime": datetime_str,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": int(m.get("goals", {}).get("h", 0)),
            "away_goals": int(m.get("goals", {}).get("a", 0))
        })

    df_matches = pd.DataFrame(matches)
    df_matches["datetime"] = pd.to_datetime(df_matches["datetime"])
    df_matches = df_matches.sort_values("datetime").reset_index(drop=True)

    xg_agg = (
        df_shots.groupby(["match_id", "team", "venue"], observed=True)
        .agg(xg=("xg", "sum"))
        .reset_index()
    )

    home_xg = xg_agg[xg_agg["venue"] == "home"][["match_id", "xg"]].rename(columns={"xg": "home_xg"})
    away_xg = xg_agg[xg_agg["venue"] == "away"][["match_id", "xg"]].rename(columns={"xg": "away_xg"})

    df_matches = df_matches.merge(home_xg, on="match_id", how="left")
    df_matches = df_matches.merge(away_xg, on="match_id", how="left")

    df_matches = df_matches.dropna(subset=["home_xg", "away_xg"])

    return df_matches
