# -*- coding: utf-8 -*-
"""
research/walk_forward_2526.py — Walk-Forward Validation sobre temporada 2024-25
================================================================================
Metodologia:
  - Split temporal: 70% entrenamiento -> 30% test (orden cronologico).
  - Modelo: Dixon-Coles + xG, gamma_i por equipo, time decay half_life=60 dias.
  - Metricas:
      * Brier Score 1X2 (vs baseline naive 1/3-1/3-1/3).
      * P&L simulado: Kelly 0.25 apostando contra las tasas medias de la liga.
        Odds simuladas = (1 - 5% vig) / tasa_historica. Sin odds de mercado
        externas (no disponibles en cache); es una aproximacion cientifica.
      * Max Drawdown sobre el P&L acumulado.
  - Gammas: cargadas del pkl mas reciente (entrenado en temporada completa).

Ejecutar desde la raiz del proyecto:
    python research/walk_forward_2526.py

Salida: resultados en stdout + research/wf_results.json
"""

import os
import sys
import json
import time
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["TRADING_CACHE_DIR"]  = os.path.join(ROOT, "backend", "cache")
os.environ["TRADING_MODELS_DIR"] = os.path.join(ROOT, "backend", "models")
os.environ["TRADING_CACHE_TTL"]  = "999999"
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from trading_deportivo.data  import shots_to_df, build_matches_with_dates
from trading_deportivo.model import fit_dixon_coles_xg, predict_match
from trading_deportivo.config import CACHE_DIR, MODELS_DIR

# =============================================================================
# CONFIGURACION
# =============================================================================
LEAGUES          = ["EPL", "La_Liga", "Bundesliga", "Ligue_1", "Serie_A"]
SEASON           = "2024_2025"
TRAIN_FRAC       = 0.70      # primeros 70% para entrenamiento
MIN_TRAIN        = 100       # minimo de partidos para entrenar
KELLY_FRACTION   = 0.25
MAX_STAKE_PCT    = 0.05
INITIAL_BANKROLL = 500.0
VIG              = 0.05      # margen del "mercado naive" simulado


# =============================================================================
# UTILIDADES
# =============================================================================

def load_cache(league: str):
    """Carga matches y shots desde JSON del cache."""
    m_path = os.path.join(CACHE_DIR, f"{league}_{SEASON}_matches.json")
    s_path = os.path.join(CACHE_DIR, f"{league}_{SEASON}_shots.json")
    with open(m_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_matches = data.get("matches", data) if isinstance(data, dict) else data
    with open(s_path, "r", encoding="utf-8") as f:
        raw_shots = json.load(f)
    return raw_matches, raw_shots


def prepare_df(raw_matches, raw_shots) -> pd.DataFrame:
    """Construye DataFrame de partidos con xG, fechas y goles reales."""
    df_shots = shots_to_df(raw_shots)
    df_shots["venue"] = df_shots["h_a"].map({"h": "home", "a": "away"})
    df_shots["xg"]    = df_shots["xg_understat"]

    df_xg = build_matches_with_dates(raw_matches, df_shots)

    goals_map = {}
    for m in raw_matches:
        if m.get("isResult"):
            mid = str(m.get("id"))
            goals_map[mid] = (
                int(m.get("goals", {}).get("h", 0)),
                int(m.get("goals", {}).get("a", 0)),
            )

    df_xg["hg"] = df_xg["match_id"].map(
        lambda x: goals_map.get(str(x), (None, None))[0]
    )
    df_xg["ag"] = df_xg["match_id"].map(
        lambda x: goals_map.get(str(x), (None, None))[1]
    )
    df_xg = df_xg.dropna(subset=["hg", "ag"])
    df_xg["hg"] = df_xg["hg"].astype(int)
    df_xg["ag"] = df_xg["ag"].astype(int)

    return df_xg.sort_values("datetime").reset_index(drop=True)


def brier_1x2(p1, px, p2, hg, ag) -> float:
    """Brier Score vectorial para 1X2 (suma de 3 errores cuadraticos)."""
    if hg > ag:
        y = (1, 0, 0)
    elif hg == ag:
        y = (0, 1, 0)
    else:
        y = (0, 0, 1)
    return (p1 - y[0])**2 + (px - y[1])**2 + (p2 - y[2])**2


def kelly_stake(prob: float, odds: float, bankroll: float) -> float:
    """Fraccion de Kelly con techo de MAX_STAKE_PCT."""
    b = odds - 1
    if b <= 0:
        return 0.0
    k = (b * prob - (1 - prob)) / b
    if k <= 0:
        return 0.0
    return round(min(k * KELLY_FRACTION * bankroll, MAX_STAKE_PCT * bankroll), 2)


def load_latest_gammas(league: str):
    """Carga los gammas del pkl mas reciente de la liga (temporada completa)."""
    try:
        pkls = [f for f in os.listdir(MODELS_DIR)
                if f.endswith(".pkl") and f.startswith(league)]
        if not pkls:
            return [], []
        pkls.sort(
            key=lambda x: os.path.getmtime(os.path.join(MODELS_DIR, x)),
            reverse=True,
        )
        with open(os.path.join(MODELS_DIR, pkls[0]), "rb") as fh:
            full_model = pickle.load(fh)
        if "gammas" not in full_model or "teams" not in full_model:
            return [], []
        g_df = pd.DataFrame({
            "team":  full_model["teams"],
            "gamma": full_model["gammas"],
        }).sort_values("gamma", ascending=False).reset_index(drop=True)
        top5 = [(r.team, round(float(r.gamma), 4)) for _, r in g_df.head(5).iterrows()]
        bot5 = [(r.team, round(float(r.gamma), 4)) for _, r in g_df.tail(5).iterrows()]
        return top5, bot5
    except Exception as exc:
        print(f"  (no se pudo cargar gammas de {league}: {exc})")
        return [], []


# =============================================================================
# WALK-FORWARD POR LIGA
# =============================================================================

def run_league(league: str) -> dict:
    print(f"\n{'='*62}")
    print(f"  LIGA: {league}")
    print(f"{'='*62}")
    t0 = time.time()

    # 1. Cargar datos
    raw_matches, raw_shots = load_cache(league)
    df = prepare_df(raw_matches, raw_shots)
    n = len(df)

    n_train = max(int(n * TRAIN_FRAC), MIN_TRAIN)
    n_test  = n - n_train

    if n_test < 10:
        return {"error": f"Test set demasiado pequeno: {n_test} partidos"}

    train_df = df.iloc[:n_train].copy()
    test_df  = df.iloc[n_train:].copy()

    print(f"  Total / Train / Test : {n} / {n_train} / {n_test}")
    print(f"  Entrenamiento        : {train_df['datetime'].min().date()} -> {train_df['datetime'].max().date()}")
    print(f"  Test                 : {test_df['datetime'].min().date()}  -> {test_df['datetime'].max().date()}")

    # 2. Tasas historicas de train -> "mercado naive" con vig
    r_hw = (train_df["hg"] > train_df["ag"]).mean()
    r_d  = (train_df["hg"] == train_df["ag"]).mean()
    r_aw = (train_df["ag"] > train_df["hg"]).mean()
    market_odds = {
        "home": (1 - VIG) / r_hw,
        "draw": (1 - VIG) / r_d,
        "away": (1 - VIG) / r_aw,
    }
    print(f"  Tasas HW/D/AW        : {r_hw:.1%} / {r_d:.1%} / {r_aw:.1%}")
    print(f"  Odds naive (-5% vig) : H {market_odds['home']:.2f} / D {market_odds['draw']:.2f} / A {market_odds['away']:.2f}")

    # 3. Entrenar modelo gamma_i en primeros 70%
    train_ids = set(train_df["match_id"].astype(str))
    train_raw = [m for m in raw_matches
                 if m.get("isResult") and str(m.get("id")) in train_ids]

    print(f"\n  Entrenando Dixon-Coles gamma_i en {n_train} partidos...")
    model = fit_dixon_coles_xg(
        train_df[["match_id", "home_team", "away_team", "home_xg", "away_xg", "datetime"]],
        raw_matches=train_raw,
        reg=0.001,
        use_decay=True,
        half_life=60,
    )
    print(f"  Convergencia: {'OK' if model['converged'] else 'NO convergio'}")

    # 4. Predicciones sobre test set
    print(f"\n  Prediciendo {n_test} partidos de test...")
    brier_scores = []
    brier_naive  = []
    bankroll     = INITIAL_BANKROLL
    peak         = INITIAL_BANKROLL
    max_dd       = 0.0
    bets_placed  = 0
    n_skipped    = 0

    for _, row in test_df.iterrows():
        try:
            pred = predict_match(row["home_team"], row["away_team"], model)
        except ValueError:
            n_skipped += 1
            continue

        p1, px, p2 = pred["p_home"], pred["p_draw"], pred["p_away"]
        hg, ag     = row["hg"], row["ag"]

        brier_scores.append(brier_1x2(p1, px, p2, hg, ag))
        brier_naive.append(brier_1x2(1/3, 1/3, 1/3, hg, ag))

        # Simulacion Kelly vs mercado naive
        for outcome, prob, key in [
            ("home", p1, "home"),
            ("draw", px, "draw"),
            ("away", p2, "away"),
        ]:
            odds    = market_odds[key]
            implied = 1.0 / odds
            if prob <= implied:
                continue
            stake = kelly_stake(prob, odds, bankroll)
            if stake <= 0:
                continue
            bets_placed += 1
            if (
                (outcome == "home" and hg > ag) or
                (outcome == "draw" and hg == ag) or
                (outcome == "away" and ag > hg)
            ):
                bankroll += stake * (odds - 1)
            else:
                bankroll -= stake
            peak   = max(peak, bankroll)
            max_dd = max(max_dd, peak - bankroll)

    if n_skipped:
        print(f"  (Equipos fuera del modelo: {n_skipped} partidos ignorados)")

    mean_bs    = float(np.mean(brier_scores)) if brier_scores else 0.0
    mean_naive = float(np.mean(brier_naive))  if brier_naive  else 0.0
    skill      = 1.0 - mean_bs / mean_naive   if mean_naive > 0 else 0.0
    pnl_final  = round(bankroll - INITIAL_BANKROLL, 2)
    roi        = round(pnl_final / INITIAL_BANKROLL * 100, 2) if bets_placed else 0.0

    elapsed = time.time() - t0

    print(f"\n  -- RESULTADOS -----------------------------------------------")
    print(f"  Partidos evaluados   : {len(brier_scores)}")
    print(f"  Brier Score gamma_i  : {mean_bs:.4f}")
    print(f"  Brier Score naive    : {mean_naive:.4f}")
    print(f"  Skill Score          : {skill:+.3f}  (0=naive, 1=perfecto)")
    print(f"  P&L simulado         : EUR {pnl_final:+.2f}")
    print(f"  Max Drawdown         : EUR {max_dd:.2f}")
    print(f"  Apuestas colocadas   : {bets_placed}")
    print(f"  ROI simulado         : {roi:+.1f}%")
    print(f"  Tiempo               : {elapsed:.1f}s")

    # 5. Cargar gammas del modelo completo (temporada entera)
    gamma_top5, gamma_bot5 = load_latest_gammas(league)

    return {
        "league"          : league,
        "n_train"         : n_train,
        "n_test"          : n_test,
        "n_evaluated"     : len(brier_scores),
        "brier_score"     : round(mean_bs, 4),
        "brier_naive"     : round(mean_naive, 4),
        "skill_score"     : round(skill, 4),
        "pnl_total"       : pnl_final,
        "bankroll_final"  : round(bankroll, 2),
        "max_drawdown"    : round(max_dd, 2),
        "bets_placed"     : bets_placed,
        "roi"             : roi,
        "converged"       : model["converged"],
        "elapsed_s"       : round(elapsed, 1),
        "gamma_top5"      : gamma_top5,
        "gamma_bot5"      : gamma_bot5,
    }


# =============================================================================
# MAIN
# =============================================================================

def _json_convert(obj):
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Tipo no serializable: {type(obj)}")


results = {}
for league in LEAGUES:
    try:
        results[league] = run_league(league)
    except Exception as exc:
        import traceback
        print(f"\nERROR en {league}: {exc}")
        traceback.print_exc()
        results[league] = {"error": str(exc)}

# Guardar JSON
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wf_results.json")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2, default=_json_convert)

# Tabla resumen
valid = [r for r in results.values() if "error" not in r]

print(f"\n\n{'='*75}")
print("  RESUMEN WALK-FORWARD 2024-25  (split 70/30 temporal, Kelly 0.25)")
print(f"{'='*75}")
hdr = f"{'Liga':<14} {'N-test':>6} {'BS gamma':>9} {'BS naive':>9} {'Skill':>7} {'P&L':>8} {'MaxDD':>8} {'Bets':>6}"
print(hdr)
print("-" * 75)
for league, r in results.items():
    if "error" in r:
        print(f"{league:<14}  ERROR: {r['error']}")
    else:
        print(
            f"{r['league']:<14} {r['n_evaluated']:>6} "
            f"{r['brier_score']:>9.4f} {r['brier_naive']:>9.4f} "
            f"{r['skill_score']:>+7.3f} "
            f"EUR{r['pnl_total']:>+7.2f} "
            f"EUR{r['max_drawdown']:>7.2f} "
            f"{r['bets_placed']:>5}"
        )

if valid:
    weights = [r["n_evaluated"] for r in valid]
    avg_bs    = float(np.average([r["brier_score"] for r in valid], weights=weights))
    avg_naive = float(np.average([r["brier_naive"] for r in valid], weights=weights))
    tot_pnl   = sum(r["pnl_total"] for r in valid)
    max_dd    = max(r["max_drawdown"] for r in valid)
    tot_eval  = sum(r["n_evaluated"] for r in valid)
    print("-" * 75)
    print(
        f"{'TOTAL / WA':<14} {tot_eval:>6} "
        f"{avg_bs:>9.4f} {avg_naive:>9.4f} "
        f"{1 - avg_bs/avg_naive:>+7.3f} "
        f"EUR{tot_pnl:>+7.2f} "
        f"EUR{max_dd:>7.2f}"
    )

print(f"\nResultados guardados en: {out_path}")
print("\nNOTA: P&L simulado con odds de mercado naive = (1-5%vig)/tasa_historica.")
print("      Sin datos de odds reales de temporada 24-25 en cache.")
