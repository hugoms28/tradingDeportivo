# -*- coding: utf-8 -*-
"""
Validacion temporal, evaluacion de modelo y backtest contra odds del mercado.
"""
import io
import sys

import numpy as np
import pandas as pd

from .model import predict_match, fit_dixon_coles_xg


# =============================================================================
# EVALUACION DEL MODELO
# =============================================================================

def evaluate_model(model, raw_matches):
    """
    Evalua el modelo con metricas de calibracion sobre partidos jugados.

    Returns:
        dict con log_loss, brier_score y datos de calibracion
    """
    teams = model["teams"]
    team_idx = {team: i for i, team in enumerate(teams)}

    results = []
    for m in raw_matches:
        if not m.get("isResult"):
            continue
        home_team = m.get("h", {}).get("title")
        away_team = m.get("a", {}).get("title")
        if home_team not in team_idx or away_team not in team_idx:
            continue

        home_goals = int(m.get("goals", {}).get("h", 0))
        away_goals = int(m.get("goals", {}).get("a", 0))

        if home_goals > away_goals:
            outcome = 2
        elif home_goals < away_goals:
            outcome = 0
        else:
            outcome = 1

        try:
            pred = predict_match(home_team, away_team, model)
            results.append({
                "home_team": home_team,
                "away_team": away_team,
                "outcome": outcome,
                "p_home": pred["p_home"],
                "p_draw": pred["p_draw"],
                "p_away": pred["p_away"]
            })
        except Exception:
            continue

    if not results:
        print("No hay partidos para evaluar")
        return None

    df = pd.DataFrame(results)

    # Log Loss
    eps = 1e-10
    log_loss_val = 0.0
    for _, row in df.iterrows():
        probs = [row["p_away"], row["p_draw"], row["p_home"]]
        probs = np.clip(probs, eps, 1 - eps)
        probs = probs / np.sum(probs)
        log_loss_val -= np.log(probs[row["outcome"]])
    log_loss_val /= len(df)

    # Brier Score
    brier_score = 0.0
    for _, row in df.iterrows():
        probs = np.array([row["p_away"], row["p_draw"], row["p_home"]])
        true_vec = np.zeros(3)
        true_vec[row["outcome"]] = 1
        brier_score += np.sum((probs - true_vec) ** 2)
    brier_score /= len(df)

    # Accuracy
    correct = 0
    for _, row in df.iterrows():
        probs = [row["p_away"], row["p_draw"], row["p_home"]]
        pred_outcome = np.argmax(probs)
        if pred_outcome == row["outcome"]:
            correct += 1
    accuracy = correct / len(df)

    print("METRICAS DE VALIDACION")
    print("=" * 50)
    print(f"Partidos evaluados: {len(df)}")
    print(f"\nLog Loss: {log_loss_val:.4f}")
    print(f"  (Benchmark aleatorio: {-np.log(1/3):.4f})")
    print(f"\nBrier Score: {brier_score:.4f}")
    print(f"  (Benchmark aleatorio: 0.667)")
    print(f"\nAccuracy: {accuracy:.1%}")
    print(f"  (Benchmark aleatorio: 33.3%)")

    return {
        "log_loss": log_loss_val,
        "brier_score": brier_score,
        "accuracy": accuracy,
        "n_matches": len(df),
        "data": df
    }


# =============================================================================
# VALIDACION TEMPORAL
# =============================================================================

def temporal_validation(df_matches, raw_matches, train_ratio=0.7, reg=0.001):
    """
    Validacion temporal: entrena con partidos antiguos, evalua con recientes.

    Args:
        df_matches: DataFrame con [datetime, home_team, away_team, home_xg, away_xg, home_goals, away_goals, match_id]
        raw_matches: Lista de partidos raw de Understat
        train_ratio: Proporcion de partidos para entrenamiento
        reg: Regularizacion L2
    """
    df = df_matches.sort_values("datetime").reset_index(drop=True)

    n_train = int(len(df) * train_ratio)
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()

    cutoff_date = train_df["datetime"].max()

    print("=" * 60)
    print("VALIDACION TEMPORAL (OUT-OF-SAMPLE)")
    print("=" * 60)
    print(f"\nCutoff: {cutoff_date.date()}")
    print(f"TRAIN: {len(train_df)} partidos ({train_df['datetime'].min().date()} -> {train_df['datetime'].max().date()})")
    print(f"TEST:  {len(test_df)} partidos ({test_df['datetime'].min().date()} -> {test_df['datetime'].max().date()})")

    train_teams = set(train_df["home_team"]) | set(train_df["away_team"])
    test_teams = set(test_df["home_team"]) | set(test_df["away_team"])
    missing_teams = test_teams - train_teams

    if missing_teams:
        print(f"\nEquipos en TEST sin datos en TRAIN: {missing_teams}")
        test_df = test_df[
            test_df["home_team"].isin(train_teams) &
            test_df["away_team"].isin(train_teams)
        ]
        print(f"  Partidos TEST despues de filtrar: {len(test_df)}")

    train_xg = train_df[["match_id", "home_team", "away_team", "home_xg", "away_xg"]].copy()

    print(f"\nEntrenando modelo con {len(train_xg)} partidos...")

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    train_match_ids = set(train_df["match_id"].astype(str))
    train_raw = [m for m in raw_matches if str(m.get("id")) in train_match_ids]

    model_train = fit_dixon_coles_xg(train_xg, raw_matches=train_raw, reg=reg)
    sys.stdout = old_stdout

    print(f"Modelo entrenado (gamma={model_train['gamma']:.3f}, rho={model_train['rho']:.4f})")

    def evaluate_on_matches(model, matches_df, set_name):
        eps = 1e-10
        log_loss = 0.0
        brier = 0.0
        correct = 0
        n = 0

        for _, row in matches_df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            hg, ag = row["home_goals"], row["away_goals"]

            if hg > ag:
                outcome = 2
            elif hg < ag:
                outcome = 0
            else:
                outcome = 1

            try:
                pred = predict_match(ht, at, model)
                probs = np.array([pred["p_away"], pred["p_draw"], pred["p_home"]])
                probs = np.clip(probs, eps, 1 - eps)
                probs = probs / probs.sum()

                log_loss -= np.log(probs[outcome])
                true_vec = np.zeros(3)
                true_vec[outcome] = 1
                brier += np.sum((probs - true_vec) ** 2)

                if np.argmax(probs) == outcome:
                    correct += 1

                n += 1
            except Exception:
                continue

        if n == 0:
            return None

        return {
            "set": set_name,
            "n": n,
            "log_loss": log_loss / n,
            "brier": brier / n,
            "accuracy": correct / n,
        }

    results_train = evaluate_on_matches(model_train, train_df, "TRAIN (in-sample)")
    results_test = evaluate_on_matches(model_train, test_df, "TEST (out-of-sample)")

    benchmark_log_loss = -np.log(1 / 3)

    print(f"\n{'='*60}")
    print("RESULTADOS")
    print(f"{'='*60}")
    print(f"\n{'Metrica':<20} {'TRAIN':<15} {'TEST (OOS)':<15} {'Benchmark':<15}")
    print("-" * 60)

    if results_train and results_test:
        print(f"{'Log Loss':<20} {results_train['log_loss']:<15.4f} {results_test['log_loss']:<15.4f} {benchmark_log_loss:<15.4f}")
        print(f"{'Brier Score':<20} {results_train['brier']:<15.4f} {results_test['brier']:<15.4f} {'0.6667':<15}")
        print(f"{'Accuracy':<20} {results_train['accuracy']*100:<14.1f}% {results_test['accuracy']*100:<14.1f}% {'33.3':<14}%")
        print(f"{'N partidos':<20} {results_train['n']:<15} {results_test['n']:<15}")

        overfit = results_test['log_loss'] - results_train['log_loss']
        print(f"\nOverfitting check: {overfit:+.4f}")
        if overfit > 0.1:
            print(f"  Alta diferencia sugiere overfitting")
        else:
            print(f"  Diferencia aceptable")

    return {
        "model": model_train,
        "train": results_train,
        "test": results_test,
        "train_df": train_df,
        "test_df": test_df,
        "cutoff_date": cutoff_date
    }


# =============================================================================
# BACKTEST VS MERCADO
# =============================================================================

def backtest_vs_market(df_matches, odds_df, model, min_edge=0.03, use_pinnacle=True,
                       league="EPL", normalize_team_name=None):
    """
    Backtesting del modelo contra odds reales del mercado.

    Args:
        df_matches: DataFrame con partidos y xG
        odds_df: DataFrame con odds historicas
        model: Modelo Dixon-Coles entrenado
        min_edge: Edge minimo para apostar
        use_pinnacle: Usar Pinnacle (True) o Bet365 (False)
        league: Liga para normalizar nombres
        normalize_team_name: Funcion de normalizacion (de team_mappings)
    """
    if use_pinnacle and "PSH" in odds_df.columns:
        odds_cols = {"home": "PSH", "draw": "PSD", "away": "PSA"}
        bookmaker = "Pinnacle"
    else:
        odds_cols = {"home": "B365H", "draw": "B365D", "away": "B365A"}
        bookmaker = "Bet365"

    print("=" * 70)
    print(f"BACKTEST VS MERCADO ({bookmaker})")
    print("=" * 70)

    results = []

    for _, match in df_matches.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]
        match_date = match["datetime"].date()

        if normalize_team_name:
            home_norm = normalize_team_name(home_team, league=league)
            away_norm = normalize_team_name(away_team, league=league)
        else:
            home_norm = home_team
            away_norm = away_team

        odds_match = odds_df[
            (odds_df["HomeTeam"] == home_norm) &
            (odds_df["AwayTeam"] == away_norm) &
            (odds_df["Date"].dt.date == match_date)
        ]

        if odds_match.empty:
            odds_match = odds_df[
                (odds_df["HomeTeam"] == home_norm) &
                (odds_df["AwayTeam"] == away_norm) &
                (abs((odds_df["Date"].dt.date - match_date).apply(
                    lambda x: x.days if hasattr(x, 'days') else 999)) <= 1)
            ]

        if odds_match.empty:
            continue

        odds_row = odds_match.iloc[0]

        try:
            market_odds_home = float(odds_row[odds_cols["home"]])
            market_odds_draw = float(odds_row[odds_cols["draw"]])
            market_odds_away = float(odds_row[odds_cols["away"]])
        except (ValueError, KeyError):
            continue

        if pd.isna(market_odds_home) or pd.isna(market_odds_draw) or pd.isna(market_odds_away):
            continue

        try:
            pred = predict_match(home_team, away_team, model)
        except Exception:
            continue

        home_goals = match["home_goals"]
        away_goals = match["away_goals"]
        if home_goals > away_goals:
            result = "home"
        elif home_goals < away_goals:
            result = "away"
        else:
            result = "draw"

        for outcome, prob_key, odds_key in [
            ("home", "p_home", market_odds_home),
            ("draw", "p_draw", market_odds_draw),
            ("away", "p_away", market_odds_away)
        ]:
            prob_modelo = pred[prob_key]
            prob_mercado = 1 / odds_key
            edge = prob_modelo - prob_mercado

            results.append({
                "date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "outcome": outcome,
                "prob_modelo": prob_modelo,
                "odds_mercado": odds_key,
                "prob_mercado": prob_mercado,
                "edge": edge,
                "result": result,
                "won": result == outcome
            })

    if not results:
        print("No se encontraron partidos coincidentes")
        return None

    df_results = pd.DataFrame(results)

    print(f"\nPartidos analizados: {len(df_results) // 3}")

    # Log Loss comparison
    eps = 1e-10
    model_ll = 0
    market_ll = 0
    n_matches = 0

    for match_date in df_results["date"].unique():
        match_data = df_results[df_results["date"] == match_date].iloc[:3]
        if len(match_data) < 3:
            continue

        home_row = match_data[match_data["outcome"] == "home"].iloc[0]
        draw_row = match_data[match_data["outcome"] == "draw"].iloc[0]
        away_row = match_data[match_data["outcome"] == "away"].iloc[0]

        probs_modelo = np.array([home_row["prob_modelo"], draw_row["prob_modelo"], away_row["prob_modelo"]])
        probs_modelo = np.clip(probs_modelo, eps, 1 - eps)
        probs_modelo = probs_modelo / probs_modelo.sum()

        probs_mercado = np.array([home_row["prob_mercado"], draw_row["prob_mercado"], away_row["prob_mercado"]])
        probs_mercado = np.clip(probs_mercado, eps, 1 - eps)
        probs_mercado = probs_mercado / probs_mercado.sum()

        result = home_row["result"]
        outcome_idx = {"home": 0, "draw": 1, "away": 2}[result]

        model_ll -= np.log(probs_modelo[outcome_idx])
        market_ll -= np.log(probs_mercado[outcome_idx])
        n_matches += 1

    if n_matches > 0:
        model_ll /= n_matches
        market_ll /= n_matches

        print(f"\n{'='*70}")
        print("LOG LOSS: MODELO vs MERCADO")
        print(f"{'='*70}")
        print(f"  Modelo:   {model_ll:.4f}")
        print(f"  Mercado:  {market_ll:.4f}")
        print(f"  Benchmark (azar): {-np.log(1/3):.4f}")

        if model_ll < market_ll:
            edge_vs_market = (market_ll - model_ll) / market_ll * 100
            print(f"\n  Modelo SUPERA al mercado por {edge_vs_market:.2f}%")
        else:
            print(f"\n  Modelo NO supera al mercado")

    # Value betting simulation
    print(f"\n{'='*70}")
    print(f"SIMULACION DE VALUE BETTING (edge >= {min_edge*100:.0f}%)")
    print(f"{'='*70}")

    value_bets = df_results[df_results["edge"] >= min_edge].copy()

    if len(value_bets) == 0:
        print(f"\nNo hay apuestas con edge >= {min_edge*100:.0f}%")
        return {"df_results": df_results, "model_ll": model_ll, "market_ll": market_ll}

    value_bets["stake"] = 1.0
    value_bets["pnl"] = value_bets.apply(
        lambda x: x["stake"] * (x["odds_mercado"] - 1) if x["won"] else -x["stake"],
        axis=1
    )

    total_bets = len(value_bets)
    wins = value_bets["won"].sum()
    total_staked = value_bets["stake"].sum()
    total_pnl = value_bets["pnl"].sum()
    roi = total_pnl / total_staked * 100

    print(f"\nApuestas realizadas: {total_bets}")
    print(f"Victorias: {wins} ({wins/total_bets*100:.1f}%)")
    print(f"Total apostado: {total_staked:.0f} unidades")
    print(f"P&L total: {total_pnl:+.2f} unidades")
    print(f"ROI: {roi:+.2f}%")

    print(f"\nDesglose por mercado:")
    for outcome in ["home", "draw", "away"]:
        subset = value_bets[value_bets["outcome"] == outcome]
        if len(subset) > 0:
            pnl = subset["pnl"].sum()
            n = len(subset)
            w = subset["won"].sum()
            print(f"  {outcome:>5}: {n:3d} apuestas, {w:3d} wins ({w/n*100:5.1f}%), P&L: {pnl:+6.2f}")

    value_bets = value_bets.sort_values("date")
    value_bets["cumulative_pnl"] = value_bets["pnl"].cumsum()

    return {
        "df_results": df_results,
        "value_bets": value_bets,
        "model_ll": model_ll if n_matches > 0 else None,
        "market_ll": market_ll if n_matches > 0 else None,
        "total_bets": total_bets,
        "roi": roi,
        "total_pnl": total_pnl
    }
