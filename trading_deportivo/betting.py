# -*- coding: utf-8 -*-
"""
Registro de apuestas, tracking de ROI y gestion de resultados.
"""
import os
from datetime import datetime

import pandas as pd

from .config import BETS_FILE
from .model import kelly_fraction


# =============================================================================
# INICIALIZACION
# =============================================================================

def init_bets_file():
    """Crea el archivo de apuestas si no existe."""
    if not os.path.exists(BETS_FILE):
        df = pd.DataFrame(columns=[
            "fecha", "partido", "mercado", "seleccion",
            "prob_modelo", "odds", "stake", "edge",
            "resultado", "profit", "settled"
        ])
        df.to_csv(BETS_FILE, index=False)
        print(f"Archivo creado: {BETS_FILE}")


# =============================================================================
# REGISTRO DE APUESTAS
# =============================================================================

def log_bet(partido, mercado, seleccion, prob_modelo, odds, stake=1.0):
    """
    Registra una nueva apuesta.

    Args:
        partido: "Borussia Dortmund vs Mainz 05"
        mercado: "1X2", "Over2.5", "BTTS", etc.
        seleccion: "1", "X", "2", "Over", "Under", "Si", "No"
        prob_modelo: Probabilidad segun nuestro modelo (0-1)
        odds: Cuota decimal
        stake: Unidades apostadas (default 1)
    """
    init_bets_file()

    edge = prob_modelo - (1 / odds)

    new_bet = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "partido": partido,
        "mercado": mercado,
        "seleccion": seleccion,
        "prob_modelo": round(prob_modelo, 4),
        "odds": odds,
        "stake": stake,
        "edge": round(edge, 4),
        "resultado": "",
        "profit": "",
        "settled": False
    }

    df = pd.read_csv(BETS_FILE)
    df = pd.concat([df, pd.DataFrame([new_bet])], ignore_index=True)
    df.to_csv(BETS_FILE, index=False)

    print(f"Apuesta registrada:")
    print(f"  {partido} | {mercado} {seleccion} @ {odds}")
    print(f"  Prob: {prob_modelo*100:.1f}% | Edge: {edge*100:+.1f}% | Stake: {stake}u")


def log_bets_from_predictions(df_pred, mercado="1X2", min_edge=0.03, stake_mode="flat", base_stake=1.0):
    """
    Registra apuestas automaticamente desde las predicciones.

    Args:
        df_pred: DataFrame de predict_matchday()
        mercado: "1X2", "Over2.5", "Under2.5", "BTTS" o "all" para todos
        min_edge: Edge minimo para apostar
        stake_mode: "flat" o "kelly"
        base_stake: Stake base (para flat) o bankroll % (para kelly)
    """
    init_bets_file()
    count = 0

    market_configs = [
        ("1X2", "1", "P_1", "Odds_1", "Edge_1"),
        ("1X2", "X", "P_X", "Odds_X", "Edge_X"),
        ("1X2", "2", "P_2", "Odds_2", "Edge_2"),
        ("Over2.5", "Over", "P_O25", "Odds_O25", "Edge_O25"),
        ("Under2.5", "Under", "P_U25", "Odds_U25", "Edge_U25"),
        ("BTTS", "Si", "P_BTTS", "Odds_BTTS_Si", "Edge_BTTS_Si"),
        ("BTTS", "No", "P_BTTS_No", "Odds_BTTS_No", "Edge_BTTS_No"),
    ]

    if mercado != "all":
        market_configs = [m for m in market_configs if m[0] == mercado]

    for _, row in df_pred.iterrows():
        for mkt, outcome, p_col, odds_col, edge_col in market_configs:
            if edge_col not in row or pd.isna(row.get(edge_col)):
                continue

            edge = row[edge_col]
            if edge >= min_edge:
                prob = row[p_col]
                odds = row[odds_col]

                if stake_mode == "kelly":
                    stake = kelly_fraction(prob, odds, fraction=0.25) * base_stake
                else:
                    stake = base_stake

                if stake > 0:
                    log_bet(
                        partido=f"{row['Local']} vs {row['Visitante']}",
                        mercado=mkt,
                        seleccion=outcome,
                        prob_modelo=prob,
                        odds=odds,
                        stake=round(stake, 2)
                    )
                    count += 1

    print(f"\nTotal: {count} apuestas registradas")


# =============================================================================
# RESULTADOS
# =============================================================================

def update_result(partido, mercado, resultado_real):
    """
    Actualiza el resultado de un partido.

    Args:
        partido: "Borussia Dortmund vs Mainz 05"
        mercado: "1X2", "Over2.5", etc.
        resultado_real: "1", "X", "2", "Over", "Under", etc.
    """
    df = pd.read_csv(BETS_FILE)

    mask = (df["partido"] == partido) & (df["mercado"] == mercado) & (df["settled"] == False)

    if mask.sum() == 0:
        print(f"No se encontro apuesta pendiente para {partido} ({mercado})")
        return

    for idx in df[mask].index:
        seleccion = df.loc[idx, "seleccion"]
        odds = df.loc[idx, "odds"]
        stake = df.loc[idx, "stake"]

        if seleccion == resultado_real:
            profit = stake * (odds - 1)
            df.loc[idx, "resultado"] = "WIN"
        else:
            profit = -stake
            df.loc[idx, "resultado"] = "LOSS"

        df.loc[idx, "profit"] = round(profit, 2)
        df.loc[idx, "settled"] = True

        print(f"{partido} | {mercado} {seleccion}: {df.loc[idx, 'resultado']} ({profit:+.2f}u)")

    df.to_csv(BETS_FILE, index=False)


# =============================================================================
# ESTADISTICAS
# =============================================================================

def show_roi_stats():
    """Muestra estadisticas de ROI."""
    if not os.path.exists(BETS_FILE):
        print("No hay apuestas registradas")
        return

    df = pd.read_csv(BETS_FILE)

    if len(df) == 0:
        print("No hay apuestas registradas")
        return

    settled = df[df["settled"] == True]
    pending = df[df["settled"] == False]

    print("=" * 60)
    print("ROI TRACKING - ESTADISTICAS")
    print("=" * 60)

    print(f"\nAPUESTAS:")
    print(f"  Total: {len(df)}")
    print(f"  Settled: {len(settled)}")
    print(f"  Pendientes: {len(pending)}")

    if len(settled) > 0:
        wins = len(settled[settled["resultado"] == "WIN"])
        losses = len(settled[settled["resultado"] == "LOSS"])
        win_rate = wins / len(settled) * 100

        total_stake = settled["stake"].sum()
        total_profit = settled["profit"].sum()
        roi = (total_profit / total_stake) * 100 if total_stake > 0 else 0

        avg_odds = settled["odds"].mean()
        avg_edge = settled["edge"].mean() * 100

        print(f"\nRESULTADOS:")
        print(f"  Wins: {wins} | Losses: {losses}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Odds promedio: {avg_odds:.2f}")
        print(f"  Edge promedio: {avg_edge:+.1f}%")

        print(f"\nPROFIT:")
        print(f"  Stake total: {total_stake:.2f}u")
        print(f"  Profit: {total_profit:+.2f}u")
        print(f"  ROI: {roi:+.1f}%")

        print(f"\nPOR MERCADO:")
        for mercado in settled["mercado"].unique():
            m_df = settled[settled["mercado"] == mercado]
            m_profit = m_df["profit"].sum()
            m_stake = m_df["stake"].sum()
            m_roi = (m_profit / m_stake) * 100 if m_stake > 0 else 0
            m_wins = len(m_df[m_df["resultado"] == "WIN"])
            print(f"  {mercado}: {m_wins}/{len(m_df)} wins | {m_profit:+.2f}u | ROI: {m_roi:+.1f}%")

    if len(pending) > 0:
        print(f"\nPENDIENTES:")
        for _, row in pending.iterrows():
            print(f"  {row['partido']} | {row['mercado']} {row['seleccion']} @ {row['odds']}")

    return df


def show_pending_bets():
    """Muestra solo apuestas pendientes."""
    if not os.path.exists(BETS_FILE):
        print("No hay apuestas registradas")
        return

    df = pd.read_csv(BETS_FILE)
    pending = df[df["settled"] == False]

    if len(pending) == 0:
        print("No hay apuestas pendientes")
        return

    print(f"APUESTAS PENDIENTES ({len(pending)}):")
    print("-" * 70)
    for _, row in pending.iterrows():
        print(f"  {row['partido']} | {row['mercado']} {row['seleccion']} @ {row['odds']} | {row['stake']}u")

    return pending
