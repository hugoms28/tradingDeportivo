# -*- coding: utf-8 -*-
"""
Modelo Dixon-Coles con xG: entrenamiento, prediccion y persistencia.
"""
import math
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from .config import MODELS_DIR, DEFAULT_MAX_GOALS


# =============================================================================
# FUNCIONES AUXILIARES DEL MODELO
# =============================================================================

def tau(x, y, lambda_home, mu_away, rho):
    """
    Factor de correlacion Dixon-Coles para marcadores bajos.
    """
    if x == 0 and y == 0:
        return 1 - lambda_home * mu_away * rho
    elif x == 0 and y == 1:
        return 1 + lambda_home * rho
    elif x == 1 and y == 0:
        return 1 + mu_away * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0


def time_decay_weight(match_date, reference_date, half_life_days=60):
    """
    Peso exponencial: partidos recientes pesan mas.
    """
    days_ago = (reference_date - match_date).days
    decay_rate = np.log(2) / half_life_days
    return np.exp(-decay_rate * max(0, days_ago))


def dc_xg_loss(params, match_xg, teams, reg=0.0, use_decay=False, half_life=60, reference_date=None):
    """
    Funcion de perdida para Dixon-Coles con xG.
    Minimiza la log-verosimilitud negativa de Poisson: -(xg*log(lambda) - lambda).
    Estadisticamente optimo para datos que siguen distribucion de Poisson.
    """
    n_teams = len(teams)
    team_idx = {team: i for i, team in enumerate(teams)}

    alphas = params[:n_teams]
    betas  = params[n_teams:2 * n_teams]
    gammas = params[2 * n_teams:3 * n_teams]    # ventaja local por equipo

    if use_decay and reference_date is None and "datetime" in match_xg.columns:
        reference_date = match_xg["datetime"].max()

    total_nll = 0.0

    for _, row in match_xg.iterrows():
        home_i = team_idx[row["home_team"]]
        away_i = team_idx[row["away_team"]]

        xg_home = row["home_xg"]
        xg_away = row["away_xg"]

        lambda_home = alphas[home_i] * betas[away_i] * gammas[home_i]
        mu_away     = alphas[away_i] * betas[home_i]

        # Poisson NLL: -(k * log(lambda) - lambda), con clip para evitar log(0)
        nll = -(xg_home * np.log(lambda_home + 1e-10) - lambda_home) \
              -(xg_away * np.log(mu_away     + 1e-10) - mu_away)

        if use_decay and "datetime" in match_xg.columns and reference_date is not None:
            weight = time_decay_weight(row["datetime"], reference_date, half_life)
            nll *= weight

        total_nll += nll

    if reg > 0:
        # L2 sobre alphas, betas y gammas: todos se encogen hacia 1.0
        penalty = reg * (
            np.sum((alphas - 1) ** 2) +
            np.sum((betas  - 1) ** 2) +
            np.sum((gammas - 1) ** 2)
        )
        total_nll += penalty

    return total_nll


def estimate_rho(match_xg, alphas, betas, gammas, teams, raw_matches):
    """
    Estima rho (correlacion Dixon-Coles) usando goles reales.
    gammas: array de ventaja local por equipo.
    """
    team_idx = {team: i for i, team in enumerate(teams)}

    match_goals = []
    for m in raw_matches:
        if not m.get("isResult"):
            continue
        home_team = m.get("h", {}).get("title")
        away_team = m.get("a", {}).get("title")
        if home_team not in team_idx or away_team not in team_idx:
            continue
        match_goals.append({
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": int(m.get("goals", {}).get("h", 0)),
            "away_goals": int(m.get("goals", {}).get("a", 0))
        })

    if not match_goals:
        return -0.05

    df_goals = pd.DataFrame(match_goals)

    def neg_log_likelihood(rho_val):
        rho = rho_val[0]
        nll = 0.0
        for _, row in df_goals.iterrows():
            home_i = team_idx[row["home_team"]]
            away_i = team_idx[row["away_team"]]

            lambda_h = alphas[home_i] * betas[away_i] * gammas[home_i]
            mu_a = alphas[away_i] * betas[home_i]

            x, y = row["home_goals"], row["away_goals"]

            p_home = poisson.pmf(x, lambda_h)
            p_away = poisson.pmf(y, mu_a)

            tau_val = tau(x, y, lambda_h, mu_a, rho)

            prob = tau_val * p_home * p_away
            if prob > 1e-10:
                nll -= np.log(prob)
            else:
                nll += 20

        return nll

    result = minimize(neg_log_likelihood, [-0.05], bounds=[(-0.3, 0.3)], method="L-BFGS-B")
    return result.x[0]


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def fit_dixon_coles_xg(match_xg, raw_matches=None, max_iter=500, reg=0.001,
                       use_decay=True, half_life=60):
    """
    Ajusta el modelo Dixon-Coles usando xG.

    Args:
        match_xg: DataFrame con [home_team, away_team, home_xg, away_xg, datetime*]
        raw_matches: Lista de partidos raw (para estimar rho con goles reales)
        max_iter: Maximo de iteraciones
        reg: Coeficiente de regularizacion L2
        use_decay: Si True, aplica time decay
        half_life: Dias para que el peso sea 50%

    Returns:
        dict con parametros ajustados y metadata
    """
    teams = sorted(pd.unique(pd.concat([match_xg["home_team"], match_xg["away_team"]])))
    n_teams = len(teams)

    print(f"Equipos: {n_teams}")
    print(f"Partidos: {len(match_xg)}")
    print(f"Parametros a estimar: {3 * n_teams} (alpha + beta + gamma por equipo)")
    print(f"Regularizacion L2: {reg}")

    has_datetime = "datetime" in match_xg.columns
    if use_decay and has_datetime:
        ref_date = match_xg["datetime"].max()
        print(f"Time Decay: ON (half_life={half_life} dias, ref={ref_date.date()})")
    elif use_decay and not has_datetime:
        print(f"Time Decay: OFF (no hay fechas en match_xg)")
        use_decay = False
    else:
        print(f"Time Decay: OFF")

    x0 = np.concatenate([
        np.ones(n_teams),       # alphas
        np.ones(n_teams),       # betas
        np.ones(n_teams) * 1.1, # gammas (ventaja local inicial ligeramente > 1)
    ])

    def constraint_sum_alpha(params):
        return np.sum(params[:n_teams]) - n_teams

    def constraint_sum_gamma(params):
        return np.sum(params[2 * n_teams:3 * n_teams]) - n_teams

    constraints = [
        {"type": "eq", "fun": constraint_sum_alpha},
        {"type": "eq", "fun": constraint_sum_gamma},
    ]

    bounds = (
        [(0.01, 5.0)] * n_teams +   # alphas
        [(0.01, 5.0)] * n_teams +   # betas
        [(0.1,  3.0)] * n_teams     # gammas
    )

    print("\nOptimizando...")
    result = minimize(
        dc_xg_loss,
        x0,
        args=(match_xg, teams, reg, use_decay, half_life),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "disp": True}
    )

    params = result.x
    alphas = params[:n_teams]
    betas  = params[n_teams:2 * n_teams]
    gammas = params[2 * n_teams:3 * n_teams]
    gamma_mean = float(np.mean(gammas))

    if raw_matches is not None:
        print("\nEstimando rho con goles reales...")
        rho = estimate_rho(match_xg, alphas, betas, gammas, teams, raw_matches)
        print(f"rho estimado: {rho:.4f}")
    else:
        rho = -0.05
        print(f"\nUsando rho por defecto: {rho}")

    params_df = pd.DataFrame({
        "team": teams,
        "alpha_attack": alphas,
        "beta_defense": betas,
        "gamma_home": gammas,
    }).sort_values("gamma_home", ascending=False)

    print(f"\nVentaja local (gamma) por equipo:")
    print(f"  Media: {gamma_mean:.4f} | Min: {gammas.min():.4f} | Max: {gammas.max():.4f}")
    top3 = params_df.head(3)
    bot3 = params_df.tail(3)
    print(f"  Fortines: {', '.join(f'{r.team}({r.gamma_home:.2f})' for _, r in top3.iterrows())}")
    print(f"  Campo neutral: {', '.join(f'{r.team}({r.gamma_home:.2f})' for _, r in bot3.iterrows())}")

    return {
        "params_df": params_df,
        "teams": teams,
        "alphas": alphas,
        "betas": betas,
        "gammas": gammas,
        "gamma": gamma_mean,   # compatibilidad con modelos antiguos
        "rho": rho,
        "converged": result.success,
        "message": result.message,
        "avg_nll": result.fun / len(match_xg),
        "reg": reg,
        "use_decay": use_decay,
        "half_life": half_life if use_decay else None
    }


# =============================================================================
# PREDICCION
# =============================================================================

def predict_match(home_team, away_team, model, max_goals=DEFAULT_MAX_GOALS):
    """
    Predice probabilidades para un partido, incluyendo todos los mercados.

    Returns:
        dict con probabilidades para: 1X2, Doble Oportunidad, O/U, AH, BTTS, score_matrix.
    """
    teams = model["teams"]
    team_idx = {team: i for i, team in enumerate(teams)}

    if home_team not in team_idx:
        raise ValueError(f"Equipo '{home_team}' no encontrado")
    if away_team not in team_idx:
        raise ValueError(f"Equipo '{away_team}' no encontrado")

    home_i = team_idx[home_team]
    away_i = team_idx[away_team]

    # Compatibilidad: modelos nuevos tienen gammas por equipo, viejos gamma escalar
    if "gammas" in model:
        gamma_home = model["gammas"][home_i]
    else:
        gamma_home = model["gamma"]
    lambda_home = model["alphas"][home_i] * model["betas"][away_i] * gamma_home
    mu_away = model["alphas"][away_i] * model["betas"][home_i]
    rho = model["rho"]

    # Matriz de probabilidades
    score_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            score_matrix[x, y] = (tau(x, y, lambda_home, mu_away, rho)
                                  * poisson.pmf(x, lambda_home)
                                  * poisson.pmf(y, mu_away))
    score_matrix = score_matrix / score_matrix.sum()

    # 1X2
    p_home_win = np.sum(np.tril(score_matrix, -1))
    p_draw = np.sum(np.diag(score_matrix))
    p_away_win = np.sum(np.triu(score_matrix, 1))

    # BTTS
    p_btts_yes = sum(score_matrix[x, y] for x in range(1, max_goals + 1) for y in range(1, max_goals + 1))

    # Doble Oportunidad
    p_1X = p_home_win + p_draw
    p_X2 = p_draw + p_away_win
    p_12 = p_home_win + p_away_win

    # Distribuciones de total goals y goal difference
    total_goals_prob = {}
    goal_diff_prob = {}
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            t = x + y
            total_goals_prob[t] = total_goals_prob.get(t, 0) + score_matrix[x, y]
            d = x - y
            goal_diff_prob[d] = goal_diff_prob.get(d, 0) + score_matrix[x, y]

    # Over/Under
    ou_lines = [0.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75]
    ou_probs = {}

    def _ou_base(line_val):
        """Compute over/under probs for a whole or half O/U line."""
        if line_val == int(line_val):  # whole line (push possible)
            n = int(line_val)
            p_over = sum(p for g, p in total_goals_prob.items() if g > n)
            p_push = total_goals_prob.get(n, 0)
            p_under = sum(p for g, p in total_goals_prob.items() if g < n)
            return {"over": p_over + 0.5 * p_push, "under": p_under + 0.5 * p_push}
        else:  # half line (no push)
            p_over = sum(p for g, p in total_goals_prob.items() if g > line_val)
            p_under = sum(p for g, p in total_goals_prob.items() if g < line_val)
            return {"over": p_over, "under": p_under}

    for line in ou_lines:
        frac = round(line % 0.5, 2)
        if frac == 0:
            # Whole or half line — compute directly
            ou_probs[line] = _ou_base(line)
        else:
            # Quarter line — average of two adjacent lines
            L1 = round(line - 0.25, 2)
            L2 = round(line + 0.25, 2)
            p1 = _ou_base(L1)
            p2 = _ou_base(L2)
            ou_probs[line] = {
                "over": 0.5 * p1["over"] + 0.5 * p2["over"],
                "under": 0.5 * p1["under"] + 0.5 * p2["under"],
            }

    # Asian Handicap
    ah_lines = [-3.5, -3.25, -3, -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25,
                0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5]
    ah_probs = {}

    def _ah_base(line_val):
        """Compute effective home/away probs for a whole or half AH line."""
        T = -line_val
        if T == int(T):  # whole line (push possible)
            T = int(T)
            p_h = sum(p for d, p in goal_diff_prob.items() if d > T)
            p_push = goal_diff_prob.get(T, 0)
            p_a = sum(p for d, p in goal_diff_prob.items() if d < T)
            return {"home": p_h + 0.5 * p_push, "away": p_a + 0.5 * p_push}
        else:  # half line (no push)
            p_h = sum(p for d, p in goal_diff_prob.items() if d > T)
            p_a = sum(p for d, p in goal_diff_prob.items() if d < T)
            return {"home": p_h, "away": p_a}

    for line in ah_lines:
        frac = round(abs(line) % 0.5, 2)
        if frac == 0:
            # Whole or half line — compute directly
            ah_probs[line] = _ah_base(line)
        else:
            # Quarter line — average of two adjacent lines
            L1 = round(line - 0.25, 2)
            L2 = round(line + 0.25, 2)
            p1 = _ah_base(L1)
            p2 = _ah_base(L2)
            ah_probs[line] = {
                "home": 0.5 * p1["home"] + 0.5 * p2["home"],
                "away": 0.5 * p1["away"] + 0.5 * p2["away"],
            }

    max_idx = np.unravel_index(score_matrix.argmax(), score_matrix.shape)
    most_likely_score = f"{max_idx[0]}-{max_idx[1]}"
    most_likely_prob = score_matrix[max_idx]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "lambda_home": lambda_home,
        "mu_away": mu_away,
        "p_home": p_home_win,
        "p_draw": p_draw,
        "p_away": p_away_win,
        "p_1X": p_1X,
        "p_X2": p_X2,
        "p_12": p_12,
        "p_btts_yes": p_btts_yes,
        "p_btts_no": 1 - p_btts_yes,
        "most_likely_score": most_likely_score,
        "most_likely_prob": most_likely_prob,
        "score_matrix": score_matrix,
        "ou_probs": ou_probs,
        "ah_probs": ah_probs,
        "p_over_15": ou_probs[1.5]["over"],
        "p_under_15": ou_probs[1.5]["under"],
        "p_over_25": ou_probs[2.5]["over"],
        "p_under_25": ou_probs[2.5]["under"],
        "p_over_35": ou_probs[3.5]["over"],
        "p_under_35": ou_probs[3.5]["under"],
    }


def kelly_fraction(prob, odds, fraction=0.25):
    """
    Calcula la fraccion de Kelly para una apuesta.
    """
    implied_prob = 1 / odds
    value = prob - implied_prob

    if value <= 0:
        return 0.0

    b = odds - 1
    f_kelly = (b * prob - (1 - prob)) / b

    return max(0, f_kelly * fraction)


def predict_matchday(matches, model, odds=None, min_edge=0.03):
    """
    Genera predicciones para una lista de partidos con todos los mercados.

    Args:
        matches: Lista de tuplas (home_team, away_team)
        model: Modelo Dixon-Coles entrenado
        odds: Diccionario opcional con odds del mercado (formato PS3838)
        min_edge: Edge minimo para considerar value bet

    Returns:
        DataFrame con predicciones incluyendo todos los mercados
    """
    predictions = []

    for home, away in matches:
        try:
            pred = predict_match(home, away, model)

            row = {
                "Partido": f"{home} vs {away}",
                "Local": home,
                "Visitante": away,
                "xG_Local": pred["lambda_home"],
                "xG_Visita": pred["mu_away"],
                "P_1": pred["p_home"],
                "P_X": pred["p_draw"],
                "P_2": pred["p_away"],
                "P_1X": pred["p_1X"],
                "P_X2": pred["p_X2"],
                "P_12": pred["p_12"],
                "P_O15": pred["p_over_15"],
                "P_U15": pred["p_under_15"],
                "P_O25": pred["p_over_25"],
                "P_U25": pred["p_under_25"],
                "P_O35": pred["p_over_35"],
                "P_U35": pred["p_under_35"],
                "P_BTTS": pred["p_btts_yes"],
                "P_BTTS_No": pred["p_btts_no"],
                "Marcador": pred["most_likely_score"],
                "Prediccion": "1" if pred["p_home"] > max(pred["p_draw"], pred["p_away"])
                             else ("X" if pred["p_draw"] > pred["p_away"] else "2")
            }

            match_key = f"{home} vs {away}"
            if odds and match_key in odds:
                match_odds = odds[match_key]

                # 1X2
                row["Odds_1"] = match_odds.get("home", None)
                row["Odds_X"] = match_odds.get("draw", None)
                row["Odds_2"] = match_odds.get("away", None)

                if row["Odds_1"]:
                    row["Edge_1"] = pred["p_home"] - 1 / row["Odds_1"]
                if row["Odds_X"]:
                    row["Edge_X"] = pred["p_draw"] - 1 / row["Odds_X"]
                if row["Odds_2"]:
                    row["Edge_2"] = pred["p_away"] - 1 / row["Odds_2"]

                # Doble Oportunidad
                for dc_key, p_key in [("1X", "p_1X"), ("X2", "p_X2"), ("12", "p_12")]:
                    dc_odds = match_odds.get(dc_key)
                    if dc_odds:
                        row[f"Odds_{dc_key}"] = dc_odds
                        row[f"Edge_{dc_key}"] = pred[p_key] - 1 / dc_odds

                # Over/Under - mejor linea
                best_ou_edge = -999
                best_ou_line = None
                best_ou_side = None
                best_ou_data = {}

                ou_probs = pred.get("ou_probs", {})

                for line, probs in ou_probs.items():
                    over_key = f"over_{line}"
                    under_key = f"under_{line}"
                    over_odds = match_odds.get(over_key)
                    under_odds = match_odds.get(under_key)

                    if over_odds:
                        edge_over = probs["over"] - 1 / over_odds
                        if edge_over > best_ou_edge:
                            best_ou_edge = edge_over
                            best_ou_line = line
                            best_ou_side = "Over"
                            best_ou_data = {"prob": probs["over"], "odds": over_odds, "edge": edge_over}

                    if under_odds:
                        edge_under = probs["under"] - 1 / under_odds
                        if edge_under > best_ou_edge:
                            best_ou_edge = edge_under
                            best_ou_line = line
                            best_ou_side = "Under"
                            best_ou_data = {"prob": probs["under"], "odds": under_odds, "edge": edge_under}

                if best_ou_line is not None:
                    row["Best_OU_Line"] = best_ou_line
                    row["Best_OU_Side"] = best_ou_side
                    row["Best_OU_Prob"] = best_ou_data["prob"]
                    row["Best_OU_Odds"] = best_ou_data["odds"]
                    row["Best_OU_Edge"] = best_ou_data["edge"]

                row["Odds_O25"] = match_odds.get("over_2.5", None)
                row["Odds_U25"] = match_odds.get("under_2.5", None)
                if row.get("Odds_O25"):
                    row["Edge_O25"] = pred["p_over_25"] - 1 / row["Odds_O25"]
                if row.get("Odds_U25"):
                    row["Edge_U25"] = pred["p_under_25"] - 1 / row["Odds_U25"]

                # Asian Handicap - mejor linea
                best_ah_edge = -999
                best_ah_line = None
                best_ah_side = None
                best_ah_data = {}

                ah_probs = pred.get("ah_probs", {})

                for line, probs in ah_probs.items():
                    ah_home_key = f"ah_home_{line}"
                    ah_away_key = f"ah_away_{line}"
                    ah_home_odds = match_odds.get(ah_home_key)
                    ah_away_odds = match_odds.get(ah_away_key)

                    if ah_home_odds:
                        edge_home = probs["home"] - 1 / ah_home_odds
                        if edge_home > best_ah_edge:
                            best_ah_edge = edge_home
                            best_ah_line = line
                            best_ah_side = "Home"
                            best_ah_data = {"prob": probs["home"], "odds": ah_home_odds, "edge": edge_home}

                    if ah_away_odds:
                        edge_away = probs["away"] - 1 / ah_away_odds
                        if edge_away > best_ah_edge:
                            best_ah_edge = edge_away
                            best_ah_line = line
                            best_ah_side = "Away"
                            best_ah_data = {"prob": probs["away"], "odds": ah_away_odds, "edge": edge_away}

                if best_ah_line is not None:
                    row["Best_AH_Line"] = best_ah_line
                    row["Best_AH_Side"] = best_ah_side
                    row["Best_AH_Prob"] = best_ah_data["prob"]
                    row["Best_AH_Odds"] = best_ah_data["odds"]
                    row["Best_AH_Edge"] = best_ah_data["edge"]

                # BTTS
                row["Odds_BTTS_Si"] = match_odds.get("btts_yes", None)
                row["Odds_BTTS_No"] = match_odds.get("btts_no", None)
                if row.get("Odds_BTTS_Si"):
                    row["Edge_BTTS_Si"] = pred["p_btts_yes"] - 1 / row["Odds_BTTS_Si"]
                if row.get("Odds_BTTS_No"):
                    row["Edge_BTTS_No"] = pred["p_btts_no"] - 1 / row["Odds_BTTS_No"]

                # Metadatos PS3838
                row["_event_id"] = match_odds.get("_event_id")
                row["_line_id"] = match_odds.get("_line_id")
                row["_starts"] = match_odds.get("_starts")

            predictions.append(row)

        except Exception as e:
            print(f"Error en {home} vs {away}: {e}")

    df = pd.DataFrame(predictions)

    # Mostrar predicciones
    print("=" * 120)
    print("PREDICCIONES PROXIMA JORNADA")
    print("=" * 120)

    decay_info = ""
    if model.get("use_decay"):
        decay_info = f", decay={model.get('half_life')}d"
    gamma_info = f"gamma_medio={model['gamma']:.3f}"
    if "gammas" in model:
        g = model["gammas"]
        gamma_info += f" [min={g.min():.2f} max={g.max():.2f}]"
    print(f"\nModelo: Dixon-Coles con xG ({gamma_info}, rho={model['rho']:.4f}{decay_info})")

    print(f"\n{'Partido':<35} {'xG':>9} {'P(1)':>6} {'P(X)':>6} {'P(2)':>6} {'O1.5':>5} {'O2.5':>5} {'O3.5':>5} {'BTTS':>5} {'Score':>5}")
    print("-" * 120)

    for _, row in df.iterrows():
        xg_str = f"{row['xG_Local']:.1f}-{row['xG_Visita']:.1f}"
        print(f"{row['Partido']:<35} {xg_str:>9} "
              f"{row['P_1']*100:>5.1f}% {row['P_X']*100:>5.1f}% {row['P_2']*100:>5.1f}% "
              f"{row['P_O15']*100:>4.0f}% {row['P_O25']*100:>4.0f}% {row['P_O35']*100:>4.0f}% "
              f"{row['P_BTTS']*100:>4.0f}% {row['Marcador']:>5}")

    # Mostrar value bets
    if odds:
        _print_value_bets(df, min_edge)

    return df


def _print_value_bets(df, min_edge):
    """Muestra value bets encontradas en las predicciones."""
    value_bets = []

    for _, row in df.iterrows():
        # 1X2
        for outcome, p_col, odds_col, edge_col in [
            ("1", "P_1", "Odds_1", "Edge_1"),
            ("X", "P_X", "Odds_X", "Edge_X"),
            ("2", "P_2", "Odds_2", "Edge_2"),
        ]:
            if edge_col in row and pd.notna(row.get(edge_col)) and row[edge_col] >= min_edge:
                kelly = kelly_fraction(row[p_col], row[odds_col], fraction=0.25)
                value_bets.append({
                    "Partido": row["Partido"], "Mercado": "1X2", "Apuesta": outcome,
                    "Prob": row[p_col], "Odds": row[odds_col], "Edge": row[edge_col],
                    "Kelly": kelly, "Prioridad": 4,
                })

        # Doble Oportunidad
        for outcome, p_col, odds_col, edge_col in [
            ("1X", "P_1X", "Odds_1X", "Edge_1X"),
            ("X2", "P_X2", "Odds_X2", "Edge_X2"),
            ("12", "P_12", "Odds_12", "Edge_12"),
        ]:
            if edge_col in row and pd.notna(row.get(edge_col)) and row[edge_col] >= min_edge:
                kelly = kelly_fraction(row[p_col], row[odds_col], fraction=0.25)
                value_bets.append({
                    "Partido": row["Partido"], "Mercado": "DC", "Apuesta": outcome,
                    "Prob": row[p_col], "Odds": row[odds_col], "Edge": row[edge_col],
                    "Kelly": kelly, "Prioridad": 3,
                })

        # Over/Under - mejor linea
        if "Best_OU_Edge" in row and pd.notna(row.get("Best_OU_Edge")) and row["Best_OU_Edge"] >= min_edge:
            kelly = kelly_fraction(row["Best_OU_Prob"], row["Best_OU_Odds"], fraction=0.25)
            value_bets.append({
                "Partido": row["Partido"], "Mercado": "O/U",
                "Apuesta": f"{row['Best_OU_Side']} {row['Best_OU_Line']}",
                "Prob": row["Best_OU_Prob"], "Odds": row["Best_OU_Odds"],
                "Edge": row["Best_OU_Edge"], "Kelly": kelly, "Prioridad": 1,
            })

        # Asian Handicap - mejor linea
        if "Best_AH_Edge" in row and pd.notna(row.get("Best_AH_Edge")) and row["Best_AH_Edge"] >= min_edge:
            kelly = kelly_fraction(row["Best_AH_Prob"], row["Best_AH_Odds"], fraction=0.25)
            ah_label = f"{'H' if row['Best_AH_Side']=='Home' else 'A'} {row['Best_AH_Line']:+g}"
            value_bets.append({
                "Partido": row["Partido"], "Mercado": "AH", "Apuesta": ah_label,
                "Prob": row["Best_AH_Prob"], "Odds": row["Best_AH_Odds"],
                "Edge": row["Best_AH_Edge"], "Kelly": kelly, "Prioridad": 2,
            })

        # O/U 2.5 legacy
        best_ou_line = row.get("Best_OU_Line")
        if best_ou_line != 2.5:
            for outcome, p_col, odds_col, edge_col in [
                ("Over 2.5", "P_O25", "Odds_O25", "Edge_O25"),
                ("Under 2.5", "P_U25", "Odds_U25", "Edge_U25"),
            ]:
                if edge_col in row and pd.notna(row.get(edge_col)) and row[edge_col] >= min_edge:
                    kelly = kelly_fraction(row[p_col], row[odds_col], fraction=0.25)
                    value_bets.append({
                        "Partido": row["Partido"], "Mercado": "O/U", "Apuesta": outcome,
                        "Prob": row[p_col], "Odds": row[odds_col], "Edge": row[edge_col],
                        "Kelly": kelly, "Prioridad": 1,
                    })

        # BTTS
        for outcome, p_col, odds_col, edge_col in [
            ("BTTS Si", "P_BTTS", "Odds_BTTS_Si", "Edge_BTTS_Si"),
            ("BTTS No", "P_BTTS_No", "Odds_BTTS_No", "Edge_BTTS_No"),
        ]:
            if edge_col in row and pd.notna(row.get(edge_col)) and row[edge_col] >= min_edge:
                kelly = kelly_fraction(row[p_col], row[odds_col], fraction=0.25)
                value_bets.append({
                    "Partido": row["Partido"], "Mercado": "BTTS", "Apuesta": outcome,
                    "Prob": row[p_col], "Odds": row[odds_col], "Edge": row[edge_col],
                    "Kelly": kelly, "Prioridad": 5,
                })

    if value_bets:
        value_bets = sorted(value_bets, key=lambda x: (x["Prioridad"], -x["Edge"]))

        print(f"\n{'='*90}")
        print(f"VALUE BETS (Edge >= {min_edge*100:.0f}%)")
        print(f"{'='*90}")
        print(f"{'#':<3} {'Partido':<35} {'Mercado':<6} {'Apuesta':<12} {'Prob':>6} {'Odds':>6} {'Edge':>7} {'Kelly':>7}")
        print("-" * 90)

        total_kelly = 0
        for i, vb in enumerate(value_bets, 1):
            total_kelly += vb["Kelly"]
            print(f"{i:<3} {vb['Partido']:<35} {vb['Mercado']:<6} {vb['Apuesta']:<12} "
                  f"{vb['Prob']*100:>5.1f}% {vb['Odds']:>6.2f} {vb['Edge']*100:>+6.1f}% {vb['Kelly']*100:>6.1f}%")

        print(f"\n{len(value_bets)} value bets | Stake total: {total_kelly*100:.1f}% del bankroll")
    else:
        print(f"\nNo hay value bets (edge >= {min_edge*100:.0f}%)")


def export_predictions(df_pred, league, filename=None):
    """Exporta predicciones a CSV formateado con todos los mercados."""
    folder = f"predicciones/{league}"
    os.makedirs(folder, exist_ok=True)

    if filename is None:
        filename = f"{folder}/predicciones_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    else:
        filename = f"{folder}/{filename}"

    df_export = df_pred.copy()

    internal_cols = [c for c in df_export.columns if c.startswith("_")]
    df_export = df_export.drop(columns=internal_cols, errors="ignore")

    for col in ["xG_Local", "xG_Visita"]:
        if col in df_export.columns:
            df_export[col] = pd.to_numeric(df_export[col], errors="coerce").round(2)

    prob_cols = [c for c in df_export.columns if c.startswith("P_")]
    for col in prob_cols:
        df_export[col] = (pd.to_numeric(df_export[col], errors="coerce") * 100).round(1).astype(str) + "%"

    for col in [c for c in df_export.columns if c.endswith("_Prob")]:
        df_export[col] = pd.to_numeric(df_export[col], errors="coerce").apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else ""
        )

    edge_cols = [c for c in df_export.columns if c.startswith("Edge_") or c.endswith("_Edge")]
    for col in edge_cols:
        df_export[col] = pd.to_numeric(df_export[col], errors="coerce").apply(
            lambda x: f"{x*100:+.1f}%" if pd.notna(x) else ""
        )

    odds_cols = [c for c in df_export.columns if c.startswith("Odds_") or c.endswith("_Odds")]
    for col in odds_cols:
        df_export[col] = pd.to_numeric(df_export[col], errors="coerce").round(2)

    col_order = [
        "Local", "Visitante", "xG_Local", "xG_Visita",
        "P_1", "P_X", "P_2",
        "P_1X", "P_X2", "P_12",
        "P_O15", "P_U15", "P_O25", "P_U25", "P_O35", "P_U35",
        "P_BTTS",
        "Prediccion",
    ]

    for col in df_export.columns:
        if col not in col_order:
            if col.startswith("Odds_") or col.startswith("Edge_") or col.startswith("Best_"):
                col_order.append(col)

    col_order = [c for c in col_order if c in df_export.columns]
    df_export = df_export[col_order]

    df_export.to_csv(filename, index=False)
    print(f"Exportado: {filename}")
    print(f"  Partidos: {len(df_export)}")
    print(f"  Columnas: {len(col_order)}")

    return filename


# =============================================================================
# PERSISTENCIA
# =============================================================================

def save_model(model, name=None, league="EPL"):
    """Guarda el modelo entrenado en disco."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    if name is None:
        name = f"{league}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    filepath = os.path.join(MODELS_DIR, f"{name}.pkl")

    model_to_save = model.copy()
    model_to_save["_metadata"] = {
        "saved_at": datetime.now().isoformat(),
        "league": league,
        "n_teams": len(model["teams"]),
    }

    with open(filepath, "wb") as f:
        pickle.dump(model_to_save, f)

    print(f"Modelo guardado: {filepath}")
    print(f"  Equipos: {len(model['teams'])}")
    if "gammas" in model:
        g = model["gammas"]
        print(f"  Gamma medio: {model['gamma']:.4f} | Min: {g.min():.4f} | Max: {g.max():.4f}")
    else:
        print(f"  Gamma: {model['gamma']:.4f}")
    print(f"  Rho: {model['rho']:.4f}")

    return filepath


def load_model(filepath=None, latest=True, league="EPL"):
    """Carga un modelo desde disco."""
    if filepath is None and latest:
        if not os.path.exists(MODELS_DIR):
            raise FileNotFoundError(f"No existe el directorio {MODELS_DIR}")

        files = [f for f in os.listdir(MODELS_DIR)
                 if f.endswith(".pkl") and league in f]

        if not files:
            raise FileNotFoundError(f"No hay modelos guardados para {league}")

        files.sort(key=lambda x: os.path.getmtime(os.path.join(MODELS_DIR, x)), reverse=True)
        filepath = os.path.join(MODELS_DIR, files[0])

    with open(filepath, "rb") as f:
        model = pickle.load(f)

    metadata = model.get("_metadata", {})
    print(f"Modelo cargado: {filepath}")
    print(f"  Guardado: {metadata.get('saved_at', 'N/A')}")
    print(f"  Equipos: {len(model['teams'])}")
    if "gammas" in model:
        g = model["gammas"]
        print(f"  Gamma medio: {model['gamma']:.4f} | Min: {g.min():.4f} | Max: {g.max():.4f}")
    else:
        print(f"  Gamma: {model['gamma']:.4f} (modelo legacy)")
    print(f"  Rho: {model['rho']:.4f}")

    return model


def list_models(league=None):
    """Lista todos los modelos guardados."""
    if not os.path.exists(MODELS_DIR):
        print("No hay modelos guardados")
        return []

    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    if league:
        files = [f for f in files if league in f]

    if not files:
        print("No hay modelos guardados")
        return []

    print(f"Modelos en {MODELS_DIR}/:")
    for f in sorted(files, reverse=True):
        path = os.path.join(MODELS_DIR, f)
        size = os.path.getsize(path) / 1024
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        print(f"  {f} ({size:.1f} KB) - {mtime.strftime('%Y-%m-%d %H:%M')}")

    return files
