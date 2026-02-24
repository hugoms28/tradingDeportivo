# -*- coding: utf-8 -*-
"""
retrain_gamma.py — Reentrena las 5 ligas con γᵢ por equipo (Poisson NLL).
Ejecutar desde la raíz del proyecto: python retrain_gamma.py

No toca la base de datos. Solo genera nuevos .pkl en backend/models/.
"""
import os
import sys
import json
import time

# ── Rutas absolutas (funciona desde cualquier directorio) ──────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["TRADING_CACHE_DIR"]  = os.path.join(ROOT, "backend", "cache")
os.environ["TRADING_MODELS_DIR"] = os.path.join(ROOT, "backend", "models")
os.environ["TRADING_CACHE_TTL"]  = "999999"   # evita que el cache expire

import numpy as np
import pandas as pd

from trading_deportivo.data  import shots_to_df, build_matches_with_dates
from trading_deportivo.model import fit_dixon_coles_xg, save_model
from trading_deportivo.config import CACHE_DIR, MODELS_DIR

LEAGUES = ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"]
SEASON  = "2024_2025"


def load_cache(league: str):
    """Carga matches y shots directamente del JSON (sin TTL)."""
    matches_path = os.path.join(CACHE_DIR, f"{league}_{SEASON}_matches.json")
    shots_path   = os.path.join(CACHE_DIR, f"{league}_{SEASON}_shots.json")

    if not os.path.exists(matches_path):
        raise FileNotFoundError(f"Cache de partidos no encontrado: {matches_path}")
    if not os.path.exists(shots_path):
        raise FileNotFoundError(f"Cache de tiros no encontrado: {shots_path}")

    with open(matches_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_matches = data.get("matches", data)   # soporta lista directa o dict con key "matches"

    with open(shots_path, "r", encoding="utf-8") as f:
        raw_shots = json.load(f)

    return raw_matches, raw_shots


def prepare_shots_df(raw_shots) -> pd.DataFrame:
    """Convierte shots crudos a DataFrame con columnas que espera build_matches_with_dates."""
    df = shots_to_df(raw_shots)
    # shots_to_df produce h_a ("h"/"a") y xg_understat;
    # build_matches_with_dates necesita venue ("home"/"away") y xg
    df["venue"] = df["h_a"].map({"h": "home", "a": "away"})
    df["xg"]    = df["xg_understat"]
    return df


# ─────────────────────────────────────────────────────────────────────────────

summary = []

for league in LEAGUES:
    print(f"\n{'═'*65}")
    print(f"  ENTRENANDO  {league}")
    print(f"{'═'*65}")
    t0 = time.time()

    try:
        raw_matches, raw_shots = load_cache(league)
        print(f"  Cache cargado: {len(raw_matches)} partidos, {len(raw_shots)} tiros")

        df_shots = prepare_shots_df(raw_shots)
        df_matches = build_matches_with_dates(raw_matches, df_shots)

        played = df_matches.dropna(subset=["home_xg", "away_xg"]).copy()
        print(f"  Partidos con xG para entrenar: {len(played)}")

        train_data = played[["match_id", "home_team", "away_team",
                              "home_xg", "away_xg", "datetime"]].copy()

        model = fit_dixon_coles_xg(
            train_data,
            raw_matches=raw_matches,
            reg=0.001,
            use_decay=True,
            half_life=60,
        )

        filepath = save_model(model, league=league)
        elapsed  = time.time() - t0

        g = model["gammas"]
        top3 = model["params_df"].head(3)
        bot3 = model["params_df"].tail(3)

        print(f"\n  Resultado:")
        print(f"    Convergencia : {'✓ OK' if model['converged'] else '✗ NO convergió'}")
        print(f"    γ medio      : {model['gamma']:.4f}  (min={g.min():.3f}  max={g.max():.3f}  std={g.std():.3f})")
        print(f"    Fortines     : {', '.join(f'{r.team}({r.gamma_home:.2f})' for _,r in top3.iterrows())}")
        print(f"    Campo neutro : {', '.join(f'{r.team}({r.gamma_home:.2f})' for _,r in bot3.iterrows())}")
        print(f"    rho          : {model['rho']:.4f}")
        print(f"    NLL medio    : {model['avg_nll']:.4f}")
        print(f"    Tiempo       : {elapsed:.1f}s")
        print(f"    Guardado en  : {filepath}")

        summary.append({
            "liga": league,
            "ok": True,
            "convergió": model["converged"],
            "γ_min": round(g.min(), 3),
            "γ_max": round(g.max(), 3),
            "γ_std": round(g.std(), 3),
            "rho": round(model["rho"], 4),
            "nll": round(model["avg_nll"], 4),
            "pkl": os.path.basename(filepath),
        })

    except Exception as exc:
        import traceback
        print(f"\n  ERROR: {exc}")
        traceback.print_exc()
        summary.append({"liga": league, "ok": False, "error": str(exc)})


# ── Resumen final ─────────────────────────────────────────────────────────────
print(f"\n\n{'═'*65}")
print("  RESUMEN REENTRENAMIENTO")
print(f"{'═'*65}")

df_sum = pd.DataFrame(summary)
print(df_sum.to_string(index=False))
print()

ok_count = sum(1 for r in summary if r.get("ok"))
print(f"  {ok_count}/{len(LEAGUES)} ligas reentrenadas con éxito")
if ok_count == len(LEAGUES):
    print("  ✓ Todos los modelos tienen γᵢ por equipo. Ejecuta eval_gamma.py para comparar.")
