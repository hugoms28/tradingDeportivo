# -*- coding: utf-8 -*-
"""
eval_gamma.py — Comparación científica: modelo LEGACY (γ global) vs NUEVO (γᵢ por equipo).
Ejecutar desde la raíz del proyecto: python eval_gamma.py

Conexión a la DB: READ-ONLY (no escribe nada).
"""
import os
import sys
import json
import glob
import sqlite3

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["TRADING_MODELS_DIR"] = os.path.join(ROOT, "backend", "models")

from trading_deportivo.model  import predict_match, load_model
from trading_deportivo.config import MODELS_DIR

DB_PATH = os.path.join(ROOT, "backend", "trading.db")

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONEXIÓN READ-ONLY
# ─────────────────────────────────────────────────────────────────────────────
print("Conectando a DB (read-only)...")
conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXTRACCIÓN DE APUESTAS RESUELTAS (fin de semana)
# ─────────────────────────────────────────────────────────────────────────────
query = """
    SELECT
        b.id,
        b.event,
        b.league,
        b.market,
        b.pick,
        b.odds,
        b.model_prob      AS old_pick_prob,
        b.result,
        b.pnl,
        b.match_starts_at,
        b.created_at,
        p.p_home          AS old_p_home,
        p.p_draw          AS old_p_draw,
        p.p_away          AS old_p_away,
        p.lambda_home     AS old_lambda_home,
        p.mu_away         AS old_mu_away
    FROM bets b
    LEFT JOIN predictions p ON b.prediction_id = p.id
    WHERE b.result IS NOT NULL
      AND b.result != 'void'
      AND b.league IN ('EPL', 'La_Liga', 'Bundesliga', 'Ligue_1')
      AND b.created_at >= '2026-02-17'
    ORDER BY b.league, b.event, b.market
"""

bets = pd.read_sql_query(query, conn)
conn.close()

print(f"  Apuestas resueltas del fin de semana: {len(bets)}")
if bets.empty:
    print("\n  No hay apuestas resueltas para comparar. ¿Están resueltas en la DB?")
    sys.exit(0)

print(f"  Ligas: {sorted(bets['league'].unique())}")
print(f"  Partidos únicos: {bets['event'].nunique()}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_model_files(league: str):
    """Devuelve (pkl_viejo, pkl_nuevo) ordenados por fecha de modificación."""
    pattern = os.path.join(MODELS_DIR, f"{league}*.pkl")
    files   = sorted(glob.glob(pattern), key=os.path.getmtime)
    if len(files) < 2:
        raise RuntimeError(f"Se necesitan al menos 2 pkl para {league} (viejo y nuevo)")
    return files[0], files[-1]   # más antiguo, más nuevo


def result_to_actual(result: str) -> float:
    """Convierte resultado de apuesta a valor numérico para Brier."""
    mapping = {"win": 1.0, "half_win": 0.75, "half_loss": 0.25, "loss": 0.0}
    return mapping.get(result, np.nan)


def infer_1x2_outcome(group: pd.DataFrame):
    """
    Intenta inferir el resultado 1X2 real a partir de las apuestas del partido.
    Devuelve 'home', 'draw', 'away' o None.
    """
    for _, row in group.iterrows():
        pick   = str(row["pick"]).strip()
        result = str(row["result"]).strip()

        # Apuestas 1X2 directas
        if pick == "1" and result == "win":  return "home"
        if pick == "X" and result == "win":  return "draw"
        if pick == "2" and result == "win":  return "away"

        # DC: si pierde, sabemos qué NO ocurrió → inferencia exacta
        if pick in ("DC 1X", "1X") and result == "loss": return "away"
        if pick in ("DC X2", "X2") and result == "loss": return "home"
        if pick in ("DC 12", "12") and result == "loss": return "draw"

    return None


def get_pick_prob_from_model(pred: dict, pick: str, odds_key: str = None) -> float | None:
    """Extrae la probabilidad del modelo para un pick concreto."""
    pick = str(pick).strip()

    mapping_1x2 = {
        "1": pred["p_home"], "Local": pred["p_home"],
        "X": pred["p_draw"], "Empate": pred["p_draw"],
        "2": pred["p_away"], "Visitante": pred["p_away"],
    }
    if pick in mapping_1x2:
        return mapping_1x2[pick]

    mapping_dc = {
        "1X": pred["p_1X"], "DC 1X": pred["p_1X"],
        "X2": pred["p_X2"], "DC X2": pred["p_X2"],
        "12": pred["p_12"], "DC 12": pred["p_12"],
    }
    if pick in mapping_dc:
        return mapping_dc[pick]

    # Over/Under (e.g. "Over 2.5", "Under 1.75")
    if pick.startswith("Over ") or pick.startswith("Under "):
        parts = pick.split()
        side  = parts[0].lower()   # "over" / "under"
        try:
            line = float(parts[1])
        except (IndexError, ValueError):
            return None
        ou = pred.get("ou_probs", {}).get(line)
        if ou:
            return ou.get(side)

    # Asian Handicap (e.g. "AH Home -0.5", "H -0.5", "Away +1.25")
    if "Home" in pick or "Away" in pick or pick.startswith("H ") or pick.startswith("A "):
        tokens = pick.replace("AH", "").split()
        side_token = tokens[0].lower() if tokens else ""
        side = "home" if side_token in ("home", "h") else "away"
        try:
            line = float(tokens[1]) if len(tokens) > 1 else None
        except ValueError:
            return None
        if line is not None:
            ah = pred.get("ah_probs", {}).get(line)
            if ah:
                return ah.get(side)

    # BTTS
    if "BTTS" in pick or "btts" in pick.lower():
        if "No" in pick:
            return pred.get("p_btts_no")
        return pred.get("p_btts_yes")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREDICCIONES CON AMBOS MODELOS
# ─────────────────────────────────────────────────────────────────────────────
print("Cargando modelos y generando predicciones...\n")

rows_output = []      # tabla comparativa final
brier_old_list = []   # Brier Score 1X2 modelo viejo
brier_new_list = []   # Brier Score 1X2 modelo nuevo
pick_brier_old = []   # Brier Score por pick (todos los mercados)
pick_brier_new = []

leagues_processed = set()
models_cache = {}     # league → (model_old, model_new)

for league in sorted(bets["league"].unique()):
    try:
        pkl_old, pkl_new = get_model_files(league)
    except RuntimeError as e:
        print(f"  [{league}] AVISO: {e} — saltando liga")
        continue

    try:
        model_old = load_model(filepath=pkl_old)
        model_new = load_model(filepath=pkl_new)
    except Exception as e:
        print(f"  [{league}] Error cargando modelos: {e}")
        continue

    has_gammas_old = "gammas" in model_old
    has_gammas_new = "gammas" in model_new
    print(f"  [{league}]")
    print(f"    Viejo  ({os.path.basename(pkl_old)}): {'γᵢ por equipo' if has_gammas_old else 'γ global = ' + str(round(model_old['gamma'], 4))}")
    print(f"    Nuevo  ({os.path.basename(pkl_new)}): {'γᵢ por equipo' if has_gammas_new else 'γ global = ' + str(round(model_new['gamma'], 4))}")
    print()

    models_cache[league] = (model_old, model_new)
    leagues_processed.add(league)

# ── Iterar por partido único ──────────────────────────────────────────────────
for event in sorted(bets["event"].unique()):
    event_bets = bets[bets["event"] == event].copy()
    league = event_bets["league"].iloc[0]

    if league not in models_cache:
        continue

    model_old, model_new = models_cache[league]

    # Parsear nombres
    parts = event.split(" vs ")
    if len(parts) != 2:
        continue
    home_team = parts[0].strip()
    away_team = parts[1].strip()

    # Inferir resultado 1X2 real
    actual_1x2 = infer_1x2_outcome(event_bets)

    # Predicciones con modelo viejo
    try:
        pred_old = predict_match(home_team, away_team, model_old)
    except Exception as e:
        pred_old = None

    # Predicciones con modelo nuevo
    try:
        pred_new = predict_match(home_team, away_team, model_new)
    except Exception as e:
        pred_new = None

    # γᵢ del equipo local en el nuevo modelo
    gamma_local = None
    if pred_new and "gammas" in model_new:
        team_idx = {t: i for i, t in enumerate(model_new["teams"])}
        if home_team in team_idx:
            gamma_local = float(model_new["gammas"][team_idx[home_team]])

    # ── Brier Score 1X2 ──────────────────────────────────────────────────────
    if actual_1x2 in ("home", "draw", "away"):
        outcome_vec = {
            "home": np.array([1.0, 0.0, 0.0]),
            "draw": np.array([0.0, 1.0, 0.0]),
            "away": np.array([0.0, 0.0, 1.0]),
        }[actual_1x2]

        if pred_old:
            probs_old = np.array([pred_old["p_home"], pred_old["p_draw"], pred_old["p_away"]])
            brier_old_list.append(np.sum((probs_old - outcome_vec) ** 2))
        if pred_new:
            probs_new = np.array([pred_new["p_home"], pred_new["p_draw"], pred_new["p_away"]])
            brier_new_list.append(np.sum((probs_new - outcome_vec) ** 2))

    # ── Una fila por apuesta del partido ────────────────────────────────────
    for _, bet in event_bets.iterrows():
        pick   = bet["pick"]
        result = bet["result"]
        actual_pick = result_to_actual(result)

        # Probabilidad vieja para este pick
        # Solo usamos pred_old si el modelo pudo predecir (evita fallback al DB
        # que tiene model_prob guardado como porcentaje en apuestas antiguas)
        if pred_old:
            prob_old_pick = get_pick_prob_from_model(pred_old, pick)
        else:
            prob_old_pick = None   # equipo no estaba en modelo viejo → no comparable

        # Probabilidad nueva para este pick
        prob_new_pick = get_pick_prob_from_model(pred_new, pick) if pred_new else None

        # Edge viejo y nuevo
        implied_prob = 1.0 / bet["odds"] if bet["odds"] else None
        edge_old = round(prob_old_pick - implied_prob, 4) if (prob_old_pick and implied_prob) else None
        edge_new = round(prob_new_pick - implied_prob, 4) if (prob_new_pick and implied_prob) else None

        # Value bet (usando mismos umbrales que runner.py)
        THRESHOLDS = {
            "Doble Oportunidad": 0.05, "DC": 0.05,
            "Asian Handicap": 0.05, "AH": 0.05,
            "Over/Under": 0.12, "O/U": 0.12,
            "BTTS": 0.12,
            "1X2": 0.15,
        }
        thr = THRESHOLDS.get(bet["market"], 0.05)
        value_old = "✓" if (edge_old is not None and edge_old >= thr) else "✗"
        value_new = "✓" if (edge_new is not None and edge_new >= thr) else "✗"

        # Brier Score por pick
        if prob_old_pick is not None and not np.isnan(actual_pick):
            pick_brier_old.append((prob_old_pick - actual_pick) ** 2)
        if prob_new_pick is not None and not np.isnan(actual_pick):
            pick_brier_new.append((prob_new_pick - actual_pick) ** 2)

        # Probs 1X2 viejo (de DB o modelo)
        if pred_old:
            ph_old, pd_old, pa_old = pred_old["p_home"], pred_old["p_draw"], pred_old["p_away"]
            xg_home_old, xg_away_old = pred_old["lambda_home"], pred_old["mu_away"]
        else:
            ph_old = bet["old_p_home"]
            pd_old = bet["old_p_draw"]
            pa_old = bet["old_p_away"]
            xg_home_old = bet["old_lambda_home"]
            xg_away_old = bet["old_mu_away"]

        rows_output.append({
            "Partido"     : event,
            "Liga"        : league,
            "Mercado"     : bet["market"],
            "Pick"        : pick,
            "Cuota"       : round(bet["odds"], 2) if bet["odds"] else None,
            "Resultado"   : result,
            "P1_viejo"    : round(ph_old, 3)  if ph_old  else None,
            "P1_nuevo"    : round(pred_new["p_home"],  3) if pred_new else None,
            "PX_viejo"    : round(pd_old, 3)  if pd_old  else None,
            "PX_nuevo"    : round(pred_new["p_draw"],  3) if pred_new else None,
            "P2_viejo"    : round(pa_old, 3)  if pa_old  else None,
            "P2_nuevo"    : round(pred_new["p_away"],  3) if pred_new else None,
            "xG_H_viejo"  : round(xg_home_old, 2) if xg_home_old else None,
            "xG_H_nuevo"  : round(pred_new["lambda_home"], 2) if pred_new else None,
            "xG_A_viejo"  : round(xg_away_old, 2) if xg_away_old else None,
            "xG_A_nuevo"  : round(pred_new["mu_away"], 2) if pred_new else None,
            "γ_local"     : round(gamma_local, 3) if gamma_local else None,
            "ProbPick_V"  : round(prob_old_pick, 3) if prob_old_pick else None,
            "ProbPick_N"  : round(prob_new_pick, 3) if prob_new_pick else None,
            "Edge_V"      : f"{edge_old*100:+.1f}%" if edge_old is not None else "—",
            "Edge_N"      : f"{edge_new*100:+.1f}%" if edge_new is not None else "—",
            "Value_V"     : value_old,
            "Value_N"     : value_new,
            "Outcome_1X2" : actual_1x2 or "?",
        })


# ─────────────────────────────────────────────────────────────────────────────
# 5. TABLA COMPARATIVA
# ─────────────────────────────────────────────────────────────────────────────
print("\n")
print("═" * 140)
print("  TABLA COMPARATIVA: MODELO LEGACY vs MODELO γᵢ POR EQUIPO")
print("═" * 140)

if not rows_output:
    print("  Sin datos para comparar.")
else:
    df_out = pd.DataFrame(rows_output)

    # ── Subtabla por liga ──────────────────────────────────────────────────
    for league in sorted(df_out["Liga"].unique()):
        sub = df_out[df_out["Liga"] == league].copy()
        print(f"\n  ── {league} ─────────────────────────────────────────────────────────────────")

        # Columnas principales para mostrar
        cols_show = ["Partido", "Mercado", "Pick", "Cuota", "Resultado",
                     "P1_viejo", "P1_nuevo", "PX_viejo", "PX_nuevo", "P2_viejo", "P2_nuevo",
                     "γ_local", "ProbPick_V", "ProbPick_N", "Edge_V", "Edge_N",
                     "Value_V", "Value_N", "Outcome_1X2"]
        cols_show = [c for c in cols_show if c in sub.columns]

        print(sub[cols_show].to_string(index=False))

    # ── Resumen de cambios en value bets ──────────────────────────────────
    print(f"\n\n{'═'*140}")
    print("  RESUMEN VALUE BETS")
    print(f"{'═'*140}")
    df_out["ambos_value"]   = (df_out["Value_V"] == "✓") & (df_out["Value_N"] == "✓")
    df_out["solo_viejo"]    = (df_out["Value_V"] == "✓") & (df_out["Value_N"] == "✗")
    df_out["solo_nuevo"]    = (df_out["Value_V"] == "✗") & (df_out["Value_N"] == "✓")
    df_out["ninguno"]       = (df_out["Value_V"] == "✗") & (df_out["Value_N"] == "✗")

    total = len(df_out)
    print(f"\n  Apuestas totales analizadas : {total}")
    print(f"  Value en ambos modelos      : {df_out['ambos_value'].sum()}")
    print(f"  Solo en modelo LEGACY       : {df_out['solo_viejo'].sum()}")
    print(f"  Solo en modelo NUEVO (γᵢ)   : {df_out['solo_nuevo'].sum()}")
    print(f"  Ninguno detecta value       : {df_out['ninguno'].sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. BRIER SCORE — VEREDICTO FINAL
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n\n{'═'*140}")
print("  MÉTRICAS CIENTÍFICAS — VEREDICTO FINAL")
print(f"{'═'*140}")

# 6a. Brier Score 1X2 (solo partidos con outcome conocido)
print(f"\n  ── Brier Score 1X2 (partidos con resultado inferible: {len(brier_old_list)}) ──")
if brier_old_list and brier_new_list:
    bs_old_1x2 = np.mean(brier_old_list)
    bs_new_1x2 = np.mean(brier_new_list)
    delta_1x2  = bs_old_1x2 - bs_new_1x2
    print(f"    Modelo LEGACY   : {bs_old_1x2:.4f}")
    print(f"    Modelo γᵢ nuevo : {bs_new_1x2:.4f}")
    print(f"    Diferencia      : {delta_1x2:+.4f}")
    if delta_1x2 > 0.001:
        print(f"\n    ✓  El modelo γᵢ MEJORA el Brier Score 1X2 en {abs(delta_1x2):.4f}")
    elif delta_1x2 < -0.001:
        print(f"\n    ✗  El modelo γᵢ EMPEORA el Brier Score 1X2 en {abs(delta_1x2):.4f}")
    else:
        print(f"\n    ≈  Diferencia marginal — más partidos necesarios para ser concluyente")
else:
    print(f"    Insuficientes partidos con resultado 1X2 inferible.")
    print(f"    (Se necesitan apuestas de tipo 1X2 directas o DC con resultado claro.)")

# 6b. Brier Score por pick (todos los mercados)
print(f"\n  ── Brier Score por pick (todos los mercados: {len(pick_brier_old)} apuestas) ──")
if pick_brier_old and pick_brier_new:
    bs_old_pick = np.mean(pick_brier_old)
    bs_new_pick = np.mean(pick_brier_new)
    delta_pick  = bs_old_pick - bs_new_pick
    print(f"    Modelo LEGACY   : {bs_old_pick:.4f}")
    print(f"    Modelo γᵢ nuevo : {bs_new_pick:.4f}")
    print(f"    Diferencia      : {delta_pick:+.4f}")
    if delta_pick > 0.001:
        print(f"\n    ✓  El modelo γᵢ tiene MEJOR calibración en los picks apostados")
    elif delta_pick < -0.001:
        print(f"\n    ✗  El modelo γᵢ tiene PEOR calibración en los picks apostados")
    else:
        print(f"\n    ≈  Sin diferencia significativa en calibración por pick")

# 6c. Diferencias medias de probabilidad
if rows_output:
    df_out_temp = pd.DataFrame(rows_output)
    mask = df_out_temp["P1_viejo"].notna() & df_out_temp["P1_nuevo"].notna()
    if mask.sum() > 0:
        diff_p1 = (df_out_temp.loc[mask, "P1_nuevo"] - df_out_temp.loc[mask, "P1_viejo"]).abs().mean()
        diff_px = (df_out_temp.loc[mask, "PX_nuevo"] - df_out_temp.loc[mask, "PX_viejo"]).abs().mean()
        diff_p2 = (df_out_temp.loc[mask, "P2_nuevo"] - df_out_temp.loc[mask, "P2_viejo"]).abs().mean()
        print(f"\n  ── Cambio medio en probabilidades 1X2 ──")
        print(f"    ΔP(1) medio : {diff_p1:.4f}  ({diff_p1*100:.2f}%)")
        print(f"    ΔP(X) medio : {diff_px:.4f}  ({diff_px*100:.2f}%)")
        print(f"    ΔP(2) medio : {diff_p2:.4f}  ({diff_p2*100:.2f}%)")

print()
