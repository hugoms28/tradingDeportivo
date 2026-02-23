# -*- coding: utf-8 -*-
"""
research/optimize_filters.py — Filter Optimizer Study
======================================================
Grid search edge_min x odds_cap para mercados 1X2, O/U 2.5 y AH.

Uso:
    python research/optimize_filters.py           # genera oportunidades + grid search
    python research/optimize_filters.py --skip-generate  # carga cache y solo hace grid search
"""

import os, sys, json, time, math, io, urllib.request, argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["TRADING_CACHE_DIR"]  = os.path.join(ROOT, "backend", "cache")
os.environ["TRADING_MODELS_DIR"] = os.path.join(ROOT, "backend", "models")
os.environ["TRADING_CACHE_TTL"]  = "999999"
sys.path.insert(0, ROOT)

from trading_deportivo.data  import shots_to_df, build_matches_with_dates
from trading_deportivo.model import fit_dixon_coles_xg, predict_match
from trading_deportivo.config import CACHE_DIR

# =============================================================================
# CONFIGURACION
# =============================================================================
LEAGUES       = ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"]
SEASON        = "2024_2025"
TRAIN_FRAC    = 0.70
KELLY_FRAC    = 0.25
MAX_STAKE_PCT = 0.05
BANKROLL0     = 500.0        # por liga
MIN_N         = 50           # minimo apuestas para relevancia estadistica

EDGE_STEPS    = [i / 100.0 for i in range(0, 26)]           # 0% a 25% (26 valores)
ODDS_CAPS     = [round(2.00 + 0.10 * i, 2) for i in range(11)]  # 2.00 a 3.00 (paso 0.10)

OPP_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opportunities.json")
GRID_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_results.json")
REPORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "report_v2_gamma_nll_Feb2026.md")

FD_URLS = {
    "EPL":        ["https://www.football-data.co.uk/mmz4281/2425/E0.csv",
                   "https://www.football-data.co.uk/mmz4281/2526/E0.csv"],
    "La_Liga":    ["https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
                   "https://www.football-data.co.uk/mmz4281/2526/SP1.csv"],
    "Bundesliga": ["https://www.football-data.co.uk/mmz4281/2425/D1.csv",
                   "https://www.football-data.co.uk/mmz4281/2526/D1.csv"],
    "Serie_A":    ["https://www.football-data.co.uk/mmz4281/2425/I1.csv",
                   "https://www.football-data.co.uk/mmz4281/2526/I1.csv"],
    "Ligue_1":    ["https://www.football-data.co.uk/mmz4281/2425/F1.csv",
                   "https://www.football-data.co.uk/mmz4281/2526/F1.csv"],
}

TEAM_MAP = {
    # EPL
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "Bournemouth": "Bournemouth",
    "Brentford": "Brentford", "Brighton": "Brighton", "Burnley": "Burnley",
    "Chelsea": "Chelsea", "Crystal Palace": "Crystal Palace", "Everton": "Everton",
    "Fulham": "Fulham", "Ipswich": "Ipswich", "Leeds": "Leeds",
    "Leicester": "Leicester", "Liverpool": "Liverpool", "Man City": "Manchester City",
    "Man United": "Manchester United", "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest", "Southampton": "Southampton",
    "Sunderland": "Sunderland", "Tottenham": "Tottenham", "West Ham": "West Ham",
    "Wolves": "Wolverhampton Wanderers",
    # La Liga
    "Alaves": "Alaves", "Ath Bilbao": "Athletic Club", "Ath Madrid": "Atletico Madrid",
    "Barcelona": "Barcelona", "Betis": "Real Betis", "Celta": "Celta Vigo",
    "Elche": "Elche", "Espanol": "Espanyol", "Getafe": "Getafe", "Girona": "Girona",
    "Las Palmas": "Las Palmas", "Leganes": "Leganes", "Levante": "Levante",
    "Mallorca": "Mallorca", "Osasuna": "Osasuna", "Oviedo": "Real Oviedo",
    "Real Madrid": "Real Madrid", "Sevilla": "Sevilla", "Sociedad": "Real Sociedad",
    "Valencia": "Valencia", "Valladolid": "Real Valladolid", "Vallecano": "Rayo Vallecano",
    "Villarreal": "Villarreal",
    # Bundesliga
    "Augsburg": "Augsburg", "Bayern Munich": "Bayern Munich", "Bochum": "Bochum",
    "Dortmund": "Borussia Dortmund", "Ein Frankfurt": "Eintracht Frankfurt",
    "FC Koln": "FC Cologne", "Freiburg": "Freiburg", "Hamburg": "Hamburger SV",
    "Heidenheim": "FC Heidenheim", "Hoffenheim": "Hoffenheim",
    "Holstein Kiel": "Holstein Kiel", "Leverkusen": "Bayer Leverkusen",
    "M'gladbach": "Borussia M.Gladbach", "Mainz": "Mainz 05",
    "RB Leipzig": "RasenBallsport Leipzig", "St Pauli": "St. Pauli",
    "Stuttgart": "VfB Stuttgart", "Union Berlin": "Union Berlin",
    "Werder Bremen": "Werder Bremen", "Wolfsburg": "Wolfsburg",
    # Serie A
    "Atalanta": "Atalanta", "Bologna": "Bologna", "Cagliari": "Cagliari",
    "Como": "Como", "Cremonese": "Cremonese", "Empoli": "Empoli",
    "Fiorentina": "Fiorentina", "Genoa": "Genoa", "Inter": "Inter",
    "Juventus": "Juventus", "Lazio": "Lazio", "Lecce": "Lecce",
    "Milan": "AC Milan", "Monza": "Monza", "Napoli": "Napoli",
    "Parma": "Parma", "Pisa": "Pisa", "Roma": "Roma", "Sassuolo": "Sassuolo",
    "Torino": "Torino", "Udinese": "Udinese", "Venezia": "Venezia", "Verona": "Verona",
    # Ligue 1
    "Angers": "Angers", "Auxerre": "Auxerre", "Brest": "Brest",
    "Le Havre": "Le Havre", "Lens": "Lens", "Lille": "Lille",
    "Lorient": "Lorient", "Lyon": "Lyon", "Marseille": "Marseille",
    "Metz": "Metz", "Monaco": "Monaco", "Montpellier": "Montpellier",
    "Nantes": "Nantes", "Nice": "Nice", "Paris FC": "Paris FC",
    "Paris SG": "Paris Saint Germain", "Reims": "Reims", "Rennes": "Rennes",
    "St Etienne": "Saint-Etienne", "Strasbourg": "Strasbourg", "Toulouse": "Toulouse",
}

MARKET_GROUPS = {
    "1X2":  ["1X2_H", "1X2_D", "1X2_A"],
    "OU25": ["OU25_O", "OU25_U"],
    "AH":   ["AH_H",  "AH_A"],
}

# =============================================================================
# UTILIDADES
# =============================================================================

def load_cache(league):
    m_path = os.path.join(CACHE_DIR, f"{league}_{SEASON}_matches.json")
    s_path = os.path.join(CACHE_DIR, f"{league}_{SEASON}_shots.json")
    with open(m_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_matches = data.get("matches", data) if isinstance(data, dict) else data
    with open(s_path, "r", encoding="utf-8") as f:
        raw_shots = json.load(f)
    return raw_matches, raw_shots


def prepare_df(raw_matches, raw_shots):
    df_shots = shots_to_df(raw_shots)
    df_shots["venue"] = df_shots["h_a"].map({"h": "home", "a": "away"})
    df_shots["xg"]    = df_shots["xg_understat"]
    df_xg = build_matches_with_dates(raw_matches, df_shots)
    goals_map = {
        str(m.get("id")): (
            int(m.get("goals", {}).get("h", 0)),
            int(m.get("goals", {}).get("a", 0)),
        )
        for m in raw_matches if m.get("isResult")
    }
    df_xg["hg"] = df_xg["match_id"].map(lambda x: goals_map.get(str(x), (None, None))[0])
    df_xg["ag"] = df_xg["match_id"].map(lambda x: goals_map.get(str(x), (None, None))[1])
    df_xg = df_xg.dropna(subset=["hg", "ag"])
    df_xg[["hg", "ag"]] = df_xg[["hg", "ag"]].astype(int)
    return df_xg.sort_values("datetime").reset_index(drop=True)


def fetch_fd_extended(league):
    """Descarga CSVs FD y extrae odds Pinnacle closing para 1X2, O/U 2.5 y AH."""
    frames = []
    for url in FD_URLS[league]:
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                content = r.read().decode("latin-1")
            df = pd.read_csv(io.StringIO(content))
            df = df.dropna(subset=["HomeTeam", "Date", "FTHG", "FTAG"])
            frames.append(df)
            print(f"    {url.split('/')[-2]}/{url.split('/')[-1]}: {len(df)} filas")
        except Exception as e:
            print(f"    WARN: {url}: {e}")
    if not frames:
        return pd.DataFrame()

    fd = pd.concat(frames, ignore_index=True)
    fd["date"]    = pd.to_datetime(fd["Date"], dayfirst=True).dt.date
    fd["home_us"] = fd["HomeTeam"].map(TEAM_MAP)
    fd["away_us"] = fd["AwayTeam"].map(TEAM_MAP)

    unmapped = (set(fd.loc[fd["home_us"].isna(), "HomeTeam"]) |
                set(fd.loc[fd["away_us"].isna(), "AwayTeam"]))
    if unmapped:
        print(f"    WARN equipos sin mapear: {sorted(unmapped)}")
    fd = fd.dropna(subset=["home_us", "away_us"])

    def get_col(df, *names):
        for n in names:
            if n in df.columns:
                return n
        return None

    # 1X2 closing (preferir PSCH/PSCD/PSCA, fallback PSH/PSD/PSA)
    c_ph = get_col(fd, "PSCH", "PSH")
    c_pd = get_col(fd, "PSCD", "PSD")
    c_pa = get_col(fd, "PSCA", "PSA")
    # O/U 2.5
    c_o25 = get_col(fd, "PC>2.5", "P>2.5")
    c_u25 = get_col(fd, "PC<2.5", "P<2.5")
    # AH
    c_ahh = get_col(fd, "PCAHH", "PAHH")
    c_aha = get_col(fd, "PCAHA", "PAHA")
    c_ahc = get_col(fd, "AHCh", "AHh")

    keep   = ["date", "home_us", "away_us", "FTHG", "FTAG"]
    rename = {}
    for c, alias in [(c_ph, "ps_h"), (c_pd, "ps_d"), (c_pa, "ps_a"),
                     (c_o25, "ps_o25"), (c_u25, "ps_u25"),
                     (c_ahh, "ps_ahh"), (c_aha, "ps_aha"), (c_ahc, "ah_ch")]:
        if c:
            keep.append(c)
            rename[c] = alias

    fd = fd[keep].rename(columns=rename)
    fd = fd.drop_duplicates(subset=["date", "home_us", "away_us"])
    return fd


def _resolve_ah_simple(hg, ag, ah_ch):
    """Resultado AH para linea entera o media (sin quarter lines). +1/0/-1."""
    eff = (hg - ag) + ah_ch
    if eff > 0:   return  1.0
    elif eff < 0: return -1.0
    else:         return  0.0   # push (solo en lineas enteras)


def resolve_ah(hg, ag, ah_ch):
    """
    Resultado para apuesta HOME AH en la linea ah_ch.
    +1.0=win, -1.0=loss, 0.0=push, +0.5=half_win, -0.5=half_loss.
    """
    rest = round(abs(ah_ch) % 0.5, 2)
    if abs(rest - 0.25) < 0.01:
        # Quarter line: split en dos lineas adyacentes
        L1 = math.floor(ah_ch * 2) / 2.0   # e.g. -1.25 -> -1.5
        L2 = L1 + 0.5                        # e.g. -1.0
        r1 = _resolve_ah_simple(hg, ag, L1)
        r2 = _resolve_ah_simple(hg, ag, L2)
        return (r1 + r2) / 2.0
    else:
        return _resolve_ah_simple(hg, ag, ah_ch)


def kelly_stake(prob, odds, bankroll):
    b = odds - 1.0
    if b <= 0:
        return 0.0
    k = (b * prob - (1.0 - prob)) / b
    if k <= 0:
        return 0.0
    return round(min(k * KELLY_FRAC * bankroll, MAX_STAKE_PCT * bankroll), 2)


# =============================================================================
# GENERAR OPORTUNIDADES (walk-forward)
# =============================================================================

def generate_opportunities():
    """Reentrena modelos (70/30), genera predicciones y las cruza con FD CSV."""
    all_opps = []

    for league in LEAGUES:
        print(f"\n{'='*62}")
        print(f"  LIGA: {league}")
        print(f"{'='*62}")
        t0 = time.time()

        # 1. Cache + split
        raw_matches, raw_shots = load_cache(league)
        df = prepare_df(raw_matches, raw_shots)
        n_train = max(int(len(df) * TRAIN_FRAC), 100)
        train_df = df.iloc[:n_train].copy()
        test_df  = df.iloc[n_train:].copy()
        print(f"  Train: {n_train}  Test: {len(test_df)}")

        # 2. Entrenar modelo
        train_ids = set(train_df["match_id"].astype(str))
        train_raw = [m for m in raw_matches
                     if m.get("isResult") and str(m.get("id")) in train_ids]
        model = fit_dixon_coles_xg(
            train_df[["match_id", "home_team", "away_team", "home_xg", "away_xg", "datetime"]],
            raw_matches=train_raw, reg=0.001, use_decay=True, half_life=60,
        )
        print(f"  Convergencia: {'OK' if model['converged'] else 'WARN'}")

        # 3. Predicciones test set (todos los mercados)
        preds = []
        for _, row in test_df.iterrows():
            try:
                pred = predict_match(row["home_team"], row["away_team"], model)
                preds.append({
                    "date":     row["datetime"].date(),
                    "home":     row["home_team"],
                    "away":     row["away_team"],
                    "p1":       pred["p_home"],
                    "px":       pred["p_draw"],
                    "p2":       pred["p_away"],
                    "p_o25":    pred["p_over_25"],
                    "p_u25":    pred["p_under_25"],
                    "ah_probs": pred["ah_probs"],   # dict {line: {home, away}}
                    "hg":       row["hg"],
                    "ag":       row["ag"],
                })
            except ValueError:
                pass
        df_preds = pd.DataFrame(preds)
        print(f"  Predicciones: {len(df_preds)}")

        # 4. Descargar FD CSV extendido
        print(f"  Descargando FD CSV extendido...")
        fd = fetch_fd_extended(league)
        if fd.empty:
            print(f"  ERROR: no se pudo descargar FD CSV")
            continue

        # 5. Merge
        df_m = df_preds.merge(
            fd,
            left_on=["date", "home", "away"],
            right_on=["date", "home_us", "away_us"],
            how="inner",
        )
        print(f"  Matched: {len(df_m)} / {len(df_preds)}")

        if len(df_m) < 5:
            print(f"  SKIP: muy pocos matches")
            continue

        # 6. Construir filas de oportunidad
        n_before = len(all_opps)
        for _, row in df_m.iterrows():
            hg = int(row["hg"])
            ag = int(row["ag"])
            dt = row["date"]
            actual = "H" if hg > ag else ("D" if hg == ag else "A")

            # --- 1X2 ---
            for mkt, prob, col, true_out in [
                ("1X2_H", row["p1"], "ps_h", "H"),
                ("1X2_D", row["px"], "ps_d", "D"),
                ("1X2_A", row["p2"], "ps_a", "A"),
            ]:
                odds = row.get(col)
                if pd.isna(odds) or float(odds) <= 1.0:
                    continue
                all_opps.append({
                    "date": str(dt), "league": league, "market": mkt,
                    "prob": float(prob), "odds": float(odds),
                    "edge": float(prob) - 1.0 / float(odds),
                    "result_mult": 1.0 if actual == true_out else -1.0,
                })

            # --- O/U 2.5 ---
            total = hg + ag
            for mkt, prob, col, wins in [
                ("OU25_O", row["p_o25"], "ps_o25", total > 2.5),
                ("OU25_U", row["p_u25"], "ps_u25", total < 2.5),
            ]:
                odds = row.get(col)
                if pd.isna(odds) or float(odds) <= 1.0:
                    continue
                all_opps.append({
                    "date": str(dt), "league": league, "market": mkt,
                    "prob": float(prob), "odds": float(odds),
                    "edge": float(prob) - 1.0 / float(odds),
                    "result_mult": 1.0 if wins else -1.0,
                })

            # --- AH ---
            ah_ch_raw = row.get("ah_ch")
            ps_ahh    = row.get("ps_ahh")
            ps_aha    = row.get("ps_aha")
            if (ah_ch_raw is None or ps_ahh is None or ps_aha is None or
                    pd.isna(ah_ch_raw) or pd.isna(ps_ahh) or pd.isna(ps_aha)):
                continue

            ah_ch  = float(ah_ch_raw)
            ps_ahh = float(ps_ahh)
            ps_aha = float(ps_aha)
            if ps_ahh <= 1.0 or ps_aha <= 1.0:
                continue

            ah_probs_dict = row["ah_probs"]
            closest = min(ah_probs_dict.keys(), key=lambda x: abs(x - ah_ch))
            if abs(closest - ah_ch) > 0.5:
                continue   # linea demasiado lejos del rango del modelo

            p_ah_h = float(ah_probs_dict[closest]["home"])
            p_ah_a = float(ah_probs_dict[closest]["away"])
            r_h    = resolve_ah(hg, ag, ah_ch)
            r_a    = -r_h   # away es opuesto (push permanece 0)

            for mkt, prob, odds, result_mult in [
                ("AH_H", p_ah_h, ps_ahh, r_h),
                ("AH_A", p_ah_a, ps_aha, r_a),
            ]:
                all_opps.append({
                    "date": str(dt), "league": league, "market": mkt,
                    "prob": prob, "odds": odds,
                    "edge": prob - 1.0 / odds,
                    "result_mult": result_mult,
                })

        print(f"  Oportunidades anadidas: {len(all_opps) - n_before}  |  "
              f"Tiempo: {time.time()-t0:.1f}s")

    return pd.DataFrame(all_opps)


# =============================================================================
# SIMULACION KELLY P&L
# =============================================================================

def simulate_pnl(df_sub):
    """
    Simula P&L Kelly con bankroll variable (BANKROLL0 por liga).
    Retorna dict con metricas agregadas.
    """
    if len(df_sub) == 0:
        return {"n": 0, "pnl": 0.0, "max_dd": 0.0, "wins": 0, "staked": 0.0}

    total_pnl    = 0.0
    total_n      = 0
    total_wins   = 0
    total_staked = 0.0
    global_max_dd = 0.0

    for league in LEAGUES:
        lg = df_sub[df_sub["league"] == league].sort_values("date")
        if len(lg) == 0:
            continue

        bankroll = BANKROLL0
        peak     = BANKROLL0

        for row in lg.itertuples(index=False):
            stake = kelly_stake(row.prob, row.odds, bankroll)
            if stake <= 0:
                continue

            mult = row.result_mult
            if   mult ==  1.0: pnl_b =  stake * (row.odds - 1)
            elif mult == -1.0: pnl_b = -stake
            elif mult ==  0.0: pnl_b =  0.0
            elif mult ==  0.5: pnl_b =  0.5 * stake * (row.odds - 1)
            elif mult == -0.5: pnl_b = -0.5 * stake
            else:
                pnl_b = mult * stake * (row.odds - 1) if mult > 0 else abs(mult) * (-stake)

            bankroll      += pnl_b
            peak           = max(peak, bankroll)
            global_max_dd  = max(global_max_dd, peak - bankroll)
            total_n       += 1
            total_staked  += stake
            total_pnl     += pnl_b
            if mult > 0:
                total_wins += 1

    return {
        "n":      total_n,
        "pnl":    round(total_pnl, 2),
        "max_dd": round(global_max_dd, 2),
        "wins":   total_wins,
        "staked": round(total_staked, 2),
    }


# =============================================================================
# GRID SEARCH
# =============================================================================

def run_grid_search(df_opps):
    """Grid search edge_min x odds_cap por mercado."""
    all_results = {}
    total_combos = len(EDGE_STEPS) * len(ODDS_CAPS)

    for mkt_name, mkt_codes in MARKET_GROUPS.items():
        print(f"\n  Mercado {mkt_name}: {total_combos} combinaciones...", end="", flush=True)
        df_mkt = df_opps[df_opps["market"].isin(mkt_codes)].copy()
        t0 = time.time()

        rows = []
        for edge_min in EDGE_STEPS:
            for odds_cap in ODDS_CAPS:
                subset = df_mkt[
                    (df_mkt["edge"] >= edge_min) &
                    (df_mkt["odds"] <= odds_cap)
                ]
                m = simulate_pnl(subset)
                roi_pct   = round(m["pnl"] / (BANKROLL0 * len(LEAGUES)) * 100, 2)
                yield_pct = round(m["pnl"] / m["staked"] * 100, 2) if m["staked"] > 0 else 0.0
                win_rate  = round(m["wins"] / m["n"] * 100, 1) if m["n"] > 0 else 0.0
                rows.append({
                    "edge_min_pct": round(edge_min * 100, 0),
                    "odds_cap":     odds_cap,
                    "n":            m["n"],
                    "pnl":          m["pnl"],
                    "max_dd":       m["max_dd"],
                    "win_rate":     win_rate,
                    "roi_pct":      roi_pct,
                    "yield_pct":    yield_pct,
                    "staked":       m["staked"],
                })

        all_results[mkt_name] = pd.DataFrame(rows)
        print(f" {time.time()-t0:.1f}s")

    return all_results


# =============================================================================
# IMPRIMIR RESULTADOS
# =============================================================================

def print_results(grid_results, df_opps):
    print(f"\n\n{'='*72}")
    print("  FILTER OPTIMIZER STUDY — Resultados")
    print(f"{'='*72}")
    print(f"  Oportunidades totales: {len(df_opps)}")
    print(f"  Ligas: {', '.join(LEAGUES)}")
    mkt_counts = df_opps.groupby("market").size()
    for mkt, n in mkt_counts.items():
        pct_pos = (df_opps.loc[df_opps["market"]==mkt, "result_mult"] > 0).mean() * 100
        print(f"    {mkt:<8}: {n:>4} oportunidades | Win% real: {pct_pos:.1f}%")

    for mkt_name, df_grid in grid_results.items():
        print(f"\n{'-'*72}")
        print(f"  MERCADO: {mkt_name}")
        print(f"{'-'*72}")

        valid = df_grid[df_grid["n"] >= MIN_N].copy()
        print(f"  Combinaciones totales: {len(df_grid)} | Con N >= {MIN_N}: {len(valid)}")

        if len(valid) == 0:
            print("  Sin combinaciones estadisticamente relevantes.")
            continue

        # Top 10 por P&L
        top10 = valid.nlargest(10, "pnl")
        print(f"\n  TOP 10 por P&L (N >= {MIN_N}):")
        hdr = f"  {'Edge>=':>6} {'Odds<=':>7} {'N':>5} {'Win%':>6} {'P&L':>9} {'MaxDD':>8} {'ROI':>7} {'Yield':>7}"
        print(hdr)
        print(f"  {'-'*60}")
        for _, r in top10.iterrows():
            print(f"  {r['edge_min_pct']:>5.0f}%  {r['odds_cap']:>7.2f}"
                  f"  {r['n']:>5.0f}  {r['win_rate']:>5.1f}%"
                  f"  €{r['pnl']:>+7.0f}  €{r['max_dd']:>6.0f}"
                  f"  {r['roi_pct']:>+6.1f}%  {r['yield_pct']:>+6.1f}%")

        # Sweet spot
        best = valid.loc[valid["pnl"].idxmax()]
        print(f"\n  >>> SWEET SPOT: edge >= {best['edge_min_pct']:.0f}%, "
              f"odds <= {best['odds_cap']:.2f} "
              f"=> P&L EUR{best['pnl']:+.0f}, N={best['n']:.0f}, "
              f"ROI {best['roi_pct']:+.1f}%, Yield {best['yield_pct']:+.1f}%")

        # Matriz P&L condensada
        print(f"\n  Matriz P&L EUR  (N<{MIN_N}=---) - filas=edge%, cols=odds_cap:")
        edge_sub = [0, 3, 5, 7, 10, 12, 15, 20, 25]
        odds_sub = [1.60, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00]

        pivot_pnl = df_grid.pivot_table(
            values="pnl", index="edge_min_pct", columns="odds_cap", aggfunc="first"
        )
        pivot_n = df_grid.pivot_table(
            values="n", index="edge_min_pct", columns="odds_cap", aggfunc="first"
        )

        e_rows = [e for e in edge_sub if e in pivot_pnl.index]
        o_cols = [c for c in odds_sub if c in pivot_pnl.columns]

        if e_rows and o_cols:
            hdr_c = "  " + " ".join(f"{c:>7.1f}" for c in o_cols)
            print(hdr_c)
            for e in e_rows:
                vals = []
                for c in o_cols:
                    pv = pivot_pnl.at[e, c] if c in pivot_pnl.columns else None
                    nv = pivot_n.at[e, c]   if c in pivot_n.columns   else None
                    if pv is None or nv is None or nv < MIN_N:
                        vals.append("    ---")
                    else:
                        vals.append(f" {pv:>+6.0f}")
                print(f"  {e:>3.0f}%  |{''.join(vals)}")


# =============================================================================
# AÑADIR SECCION 6 AL INFORME
# =============================================================================

def append_section_6(grid_results, df_opps):
    sweet_spots = {}
    for mkt_name, df_grid in grid_results.items():
        valid = df_grid[df_grid["n"] >= MIN_N]
        if len(valid) > 0:
            sweet_spots[mkt_name] = valid.loc[valid["pnl"].idxmax()]

    lines = [
        "",
        "---",
        "",
        "## 6. Optimizacion de Filtros de Ejecucion",
        "",
        f"**Fecha**: 2026-02-23  ",
        "**Objetivo**: Encontrar el par (edge_min, odds_cap) optimo por mercado que  ",
        "maximiza el P&L real vs Pinnacle closing con muestra >= 50 apuestas.  ",
        "",
        f"**Grid**: edge_min 0–25% (paso 1%) × odds_cap 1.20–5.00 (paso 0.20)  ",
        "**Mercados**: 1X2, O/U 2.5, Asian Handicap  ",
        "**Datos**: Pinnacle closing (football-data.co.uk), test set oct 2025 – feb 2026  ",
        f"**Universo total**: {len(df_opps)} oportunidades ({', '.join(LEAGUES)})",
        "",
    ]

    mkt_counts = df_opps.groupby("market").size().to_dict()
    lines.append("| Mercado | N opps | Win% real |")
    lines.append("|---------|-------:|----------:|")
    for mkt, n in sorted(mkt_counts.items()):
        pct_pos = (df_opps.loc[df_opps["market"] == mkt, "result_mult"] > 0).mean() * 100
        lines.append(f"| {mkt} | {n} | {pct_pos:.1f}% |")
    lines.append("")
    lines.append("---")
    lines.append("")

    for mkt_name, df_grid in grid_results.items():
        valid = df_grid[df_grid["n"] >= MIN_N]
        lines.append(f"### Mercado: {mkt_name}")

        if len(valid) == 0:
            lines.append(f"Sin combinaciones estadisticamente relevantes (N < {MIN_N}).")
            lines.append("")
            continue

        best = sweet_spots[mkt_name]
        lines += [
            f"**Sweet spot**: edge ≥ {best['edge_min_pct']:.0f}%, odds ≤ {best['odds_cap']:.2f}",
            "",
            "Top 5 combinaciones por P&L:",
            "",
            "| Edge≥ | Odds≤ | N | Win% | P&L | ROI | Yield |",
            "|------:|------:|--:|-----:|----:|----:|------:|",
        ]
        top5 = valid.nlargest(5, "pnl")
        for _, r in top5.iterrows():
            lines.append(
                f"| {r['edge_min_pct']:.0f}% | {r['odds_cap']:.2f}"
                f" | {r['n']:.0f} | {r['win_rate']:.1f}%"
                f" | €{r['pnl']:+.0f} | {r['roi_pct']:+.1f}%"
                f" | {r['yield_pct']:+.1f}% |"
            )
        lines.append("")

    # Tabla de parametros recomendados
    lines += [
        "---",
        "",
        "### Parametros recomendados para produccion",
        "",
        "> Basados en sweet spot P&L-maximo con N >= 50, test Pinnacle oct-feb 2026.",
        "",
        "| Mercado | Edge min | Odds max | N | P&L | Yield |",
        "|---------|----------:|---------:|--:|----:|------:|",
    ]
    for mkt, best in sweet_spots.items():
        lines.append(
            f"| {mkt} | {best['edge_min_pct']:.0f}% | {best['odds_cap']:.2f}"
            f" | {best['n']:.0f} | €{best['pnl']:+.0f} | {best['yield_pct']:+.1f}% |"
        )

    lines += [
        "",
        "> **Limitacion**: muestra ~150 partidos por liga en test set. Continuar",
        "> validando en produccion matchweek a matchweek antes de ajustar definitivamente.",
        "",
        f"*Script: `research/optimize_filters.py`. Grid completo: `research/grid_results.json`.*",
    ]

    with open(REPORT_FILE, "a", encoding="utf-8") as fh:
        fh.write("\n" + "\n".join(lines) + "\n")

    print(f"\nSeccion 6 anadida a {REPORT_FILE}")
    return sweet_spots


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter Optimizer Study")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Carga oportunidades desde cache (omite walk-forward)")
    args = parser.parse_args()

    # --- Step 1: Generar o cargar oportunidades ---
    if args.skip_generate and os.path.exists(OPP_FILE):
        print(f"Cargando oportunidades desde {OPP_FILE}...")
        with open(OPP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        df_opps = pd.DataFrame(data)
        print(f"  {len(df_opps)} oportunidades cargadas.")
    else:
        print("Generando oportunidades (walk-forward + FD CSVs)...")
        df_opps = generate_opportunities()

        # Guardar cache
        def _conv(o):
            import datetime
            if isinstance(o, (np.floating,)):   return float(o)
            if isinstance(o, (np.integer,)):    return int(o)
            if isinstance(o, datetime.date):    return str(o)
            raise TypeError(type(o))

        with open(OPP_FILE, "w", encoding="utf-8") as f:
            json.dump(df_opps.to_dict("records"), f, indent=2, default=_conv)
        print(f"\nOportunidades guardadas: {OPP_FILE} ({len(df_opps)} filas)")

    if len(df_opps) == 0:
        print("ERROR: no hay oportunidades. Revisa los CSVs y el cache.")
        sys.exit(1)

    df_opps["date"]       = pd.to_datetime(df_opps["date"]).dt.date
    df_opps["edge"]       = df_opps["edge"].astype(float)
    df_opps["odds"]       = df_opps["odds"].astype(float)
    df_opps["result_mult"]= df_opps["result_mult"].astype(float)

    # --- Step 2: Grid search ---
    print(f"\nGrid search: {len(EDGE_STEPS)} edges x {len(ODDS_CAPS)} odds_caps x "
          f"{len(MARKET_GROUPS)} mercados = "
          f"{len(EDGE_STEPS)*len(ODDS_CAPS)*len(MARKET_GROUPS)} combinaciones totales")
    t_gs = time.time()
    grid_results = run_grid_search(df_opps)
    print(f"Grid search completado en {time.time()-t_gs:.1f}s")

    # Guardar grid JSON
    def _conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)):  return int(o)
        raise TypeError(type(o))

    grid_json = {mkt: df.to_dict("records") for mkt, df in grid_results.items()}
    with open(GRID_FILE, "w", encoding="utf-8") as f:
        json.dump(grid_json, f, indent=2, default=_conv)
    print(f"Grid guardado: {GRID_FILE}")

    # --- Step 3: Imprimir resultados ---
    print_results(grid_results, df_opps)

    # --- Step 4: Añadir seccion 6 al informe ---
    sweet_spots = append_section_6(grid_results, df_opps)

    # --- Resumen final ---
    print(f"\n{'='*60}")
    print("  PARAMETROS GANADORES RECOMENDADOS")
    print(f"{'='*60}")
    for mkt, best in sweet_spots.items():
        print(f"  {mkt:<6}: edge >= {best['edge_min_pct']:.0f}%,  "
              f"odds <= {best['odds_cap']:.2f}  "
              f"[N={best['n']:.0f}, P&L €{best['pnl']:+.0f}, "
              f"Yield {best['yield_pct']:+.1f}%]")
    print()
