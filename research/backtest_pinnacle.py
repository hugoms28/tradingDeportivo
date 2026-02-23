# -*- coding: utf-8 -*-
"""
research/backtest_pinnacle.py — Backtest vs odds de cierre de Pinnacle (football-data.co.uk)
==============================================================================================
Metodologia:
  - Misma logica que walk_forward_2526.py: split 70% train / 30% test.
  - Para el test set, descarga los CSVs de football-data.co.uk (temporadas
    24-25 y 25-26) que contienen PSCH, PSCD, PSCA (Pinnacle closing odds).
  - Hace join por (fecha, home_team, away_team) con mapeo de nombres.
  - Calcula P&L real con Kelly 0.25 SOLO cuando el modelo tiene edge positivo
    vs la cuota de cierre de Pinnacle.
  - Guarda resultados en research/report_pinnacle_vs_gamma_Feb2026.md

Ejecutar desde la raiz del proyecto:
    python research/backtest_pinnacle.py
"""

import os
import sys
import json
import time
import io
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["TRADING_CACHE_DIR"]  = os.path.join(ROOT, "backend", "cache")
os.environ["TRADING_MODELS_DIR"] = os.path.join(ROOT, "backend", "models")
os.environ["TRADING_CACHE_TTL"]  = "999999"
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from trading_deportivo.data  import shots_to_df, build_matches_with_dates
from trading_deportivo.model import fit_dixon_coles_xg, predict_match
from trading_deportivo.config import CACHE_DIR

# =============================================================================
# CONFIGURACION
# =============================================================================
LEAGUES    = ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"]
SEASON     = "2024_2025"
TRAIN_FRAC = 0.70
MIN_TRAIN  = 100
KELLY_FRAC = 0.25
MAX_STAKE  = 0.05
BANKROLL0  = 500.0
MIN_EDGE   = 0.0   # solo edge > 0 vs Pinnacle

# URLs football-data.co.uk
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

# Mapeo football-data.co.uk -> nombres Understat (exactos en cache)
TEAM_MAP = {
    # EPL
    "Arsenal":        "Arsenal",
    "Aston Villa":    "Aston Villa",
    "Bournemouth":    "Bournemouth",
    "Brentford":      "Brentford",
    "Brighton":       "Brighton",
    "Burnley":        "Burnley",
    "Chelsea":        "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton":        "Everton",
    "Fulham":         "Fulham",
    "Ipswich":        "Ipswich",
    "Leeds":          "Leeds",
    "Leicester":      "Leicester",
    "Liverpool":      "Liverpool",
    "Man City":       "Manchester City",
    "Man United":     "Manchester United",
    "Newcastle":      "Newcastle United",
    "Nott'm Forest":  "Nottingham Forest",
    "Southampton":    "Southampton",
    "Sunderland":     "Sunderland",
    "Tottenham":      "Tottenham",
    "West Ham":       "West Ham",
    "Wolves":         "Wolverhampton Wanderers",
    # La Liga
    "Alaves":         "Alaves",
    "Ath Bilbao":     "Athletic Club",
    "Ath Madrid":     "Atletico Madrid",
    "Barcelona":      "Barcelona",
    "Betis":          "Real Betis",
    "Celta":          "Celta Vigo",
    "Elche":          "Elche",
    "Espanol":        "Espanyol",
    "Getafe":         "Getafe",
    "Girona":         "Girona",
    "Las Palmas":     "Las Palmas",
    "Leganes":        "Leganes",
    "Levante":        "Levante",
    "Mallorca":       "Mallorca",
    "Osasuna":        "Osasuna",
    "Oviedo":         "Real Oviedo",
    "Real Madrid":    "Real Madrid",
    "Sevilla":        "Sevilla",
    "Sociedad":       "Real Sociedad",
    "Valencia":       "Valencia",
    "Valladolid":     "Real Valladolid",
    "Vallecano":      "Rayo Vallecano",
    "Villarreal":     "Villarreal",
    # Bundesliga
    "Augsburg":       "Augsburg",
    "Bayern Munich":  "Bayern Munich",
    "Bochum":         "Bochum",
    "Dortmund":       "Borussia Dortmund",
    "Ein Frankfurt":  "Eintracht Frankfurt",
    "FC Koln":        "FC Cologne",
    "Freiburg":       "Freiburg",
    "Hamburg":        "Hamburger SV",
    "Heidenheim":     "FC Heidenheim",
    "Hoffenheim":     "Hoffenheim",
    "Holstein Kiel":  "Holstein Kiel",
    "Leverkusen":     "Bayer Leverkusen",
    "M'gladbach":     "Borussia M.Gladbach",
    "Mainz":          "Mainz 05",
    "RB Leipzig":     "RasenBallsport Leipzig",
    "St Pauli":       "St. Pauli",
    "Stuttgart":      "VfB Stuttgart",
    "Union Berlin":   "Union Berlin",
    "Werder Bremen":  "Werder Bremen",
    "Wolfsburg":      "Wolfsburg",
    # Serie A
    "Atalanta":       "Atalanta",
    "Bologna":        "Bologna",
    "Cagliari":       "Cagliari",
    "Como":           "Como",
    "Cremonese":      "Cremonese",
    "Empoli":         "Empoli",
    "Fiorentina":     "Fiorentina",
    "Genoa":          "Genoa",
    "Inter":          "Inter",
    "Juventus":       "Juventus",
    "Lazio":          "Lazio",
    "Lecce":          "Lecce",
    "Milan":          "AC Milan",
    "Monza":          "Monza",
    "Napoli":         "Napoli",
    "Parma":          "Parma",
    "Pisa":           "Pisa",
    "Roma":           "Roma",
    "Sassuolo":       "Sassuolo",
    "Torino":         "Torino",
    "Udinese":        "Udinese",
    "Venezia":        "Venezia",
    "Verona":         "Verona",
    # Ligue 1
    "Angers":         "Angers",
    "Auxerre":        "Auxerre",
    "Brest":          "Brest",
    "Le Havre":       "Le Havre",
    "Lens":           "Lens",
    "Lille":          "Lille",
    "Lorient":        "Lorient",
    "Lyon":           "Lyon",
    "Marseille":      "Marseille",
    "Metz":           "Metz",
    "Monaco":         "Monaco",
    "Montpellier":    "Montpellier",
    "Nantes":         "Nantes",
    "Nice":           "Nice",
    "Paris FC":       "Paris FC",
    "Paris SG":       "Paris Saint Germain",
    "Reims":          "Reims",
    "Rennes":         "Rennes",
    "St Etienne":     "Saint-Etienne",
    "Strasbourg":     "Strasbourg",
    "Toulouse":       "Toulouse",
}


# =============================================================================
# UTILIDADES
# =============================================================================

def load_cache(league: str):
    m_path = os.path.join(CACHE_DIR, f"{league}_{SEASON}_matches.json")
    s_path = os.path.join(CACHE_DIR, f"{league}_{SEASON}_shots.json")
    with open(m_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_matches = data.get("matches", data) if isinstance(data, dict) else data
    with open(s_path, "r", encoding="utf-8") as f:
        raw_shots = json.load(f)
    return raw_matches, raw_shots


def prepare_df(raw_matches, raw_shots) -> pd.DataFrame:
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
    df_xg["hg"] = df_xg["hg"].astype(int)
    df_xg["ag"] = df_xg["ag"].astype(int)
    return df_xg.sort_values("datetime").reset_index(drop=True)


def fetch_fd_csv(league: str) -> pd.DataFrame:
    """Descarga y concatena CSVs 24-25 y 25-26 de football-data.co.uk."""
    frames = []
    for url in FD_URLS[league]:
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                content = r.read().decode("latin-1")
            df = pd.read_csv(io.StringIO(content))
            df = df.dropna(subset=["HomeTeam", "Date"])
            # Eliminar filas sin resultados (partidos futuros)
            df = df.dropna(subset=["FTHG", "FTAG"])
            frames.append(df)
            print(f"    {url.split('/')[-2]}/{url.split('/')[-1]}: {len(df)} partidos")
        except Exception as exc:
            print(f"    WARN: no se pudo descargar {url}: {exc}")
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    # Normalizar fecha -> date object
    combined["date"] = pd.to_datetime(combined["Date"], dayfirst=True).dt.date
    # Normalizar nombres de equipo -> Understat
    combined["home_us"] = combined["HomeTeam"].map(TEAM_MAP)
    combined["away_us"] = combined["AwayTeam"].map(TEAM_MAP)
    unmapped = set(combined["HomeTeam"][combined["home_us"].isna()].unique()) | \
               set(combined["AwayTeam"][combined["away_us"].isna()].unique())
    if unmapped:
        print(f"    WARN: equipos sin mapear: {sorted(unmapped)}")
    combined = combined.dropna(subset=["home_us", "away_us"])
    # Quedarse con columnas utiles
    ps_cols = [c for c in ["PSCH", "PSCD", "PSCA"] if c in combined.columns]
    if not ps_cols:
        ps_cols = [c for c in ["PSH", "PSD", "PSA"] if c in combined.columns]
        print(f"    INFO: usando odds de apertura (PSH/PSD/PSA); closing no disponibles")
    keep = ["date", "home_us", "away_us", "FTHG", "FTAG"] + ps_cols
    combined = combined[keep].rename(columns={
        ps_cols[0]: "odds_h",
        ps_cols[1]: "odds_d",
        ps_cols[2]: "odds_a",
    })
    combined = combined.dropna(subset=["odds_h", "odds_d", "odds_a"])
    combined = combined.drop_duplicates(subset=["date", "home_us", "away_us"])
    return combined


def kelly_stake(prob: float, odds: float, bankroll: float) -> float:
    b = odds - 1
    if b <= 0:
        return 0.0
    k = (b * prob - (1 - prob)) / b
    if k <= 0:
        return 0.0
    return round(min(k * KELLY_FRAC * bankroll, MAX_STAKE * bankroll), 2)


# =============================================================================
# BACKTEST POR LIGA
# =============================================================================

def run_league(league: str) -> dict:
    print(f"\n{'='*62}")
    print(f"  LIGA: {league}")
    print(f"{'='*62}")
    t0 = time.time()

    # 1. Modelo: cargar cache, split, entrenar
    raw_matches, raw_shots = load_cache(league)
    df = prepare_df(raw_matches, raw_shots)
    n = len(df)
    n_train = max(int(n * TRAIN_FRAC), MIN_TRAIN)

    train_df = df.iloc[:n_train].copy()
    test_df  = df.iloc[n_train:].copy()

    print(f"  Train: {n_train} partidos  Test: {len(test_df)} partidos")
    print(f"  Entrenando modelo gamma_i...")

    train_ids = set(train_df["match_id"].astype(str))
    train_raw = [m for m in raw_matches
                 if m.get("isResult") and str(m.get("id")) in train_ids]
    model = fit_dixon_coles_xg(
        train_df[["match_id", "home_team", "away_team", "home_xg", "away_xg", "datetime"]],
        raw_matches=train_raw, reg=0.001, use_decay=True, half_life=60,
    )
    print(f"  Convergencia: {'OK' if model['converged'] else 'NO'}")

    # 2. Generar predicciones para test set
    preds = []
    for _, row in test_df.iterrows():
        try:
            pred = predict_match(row["home_team"], row["away_team"], model)
            preds.append({
                "date":      row["datetime"].date(),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "p1":        pred["p_home"],
                "px":        pred["p_draw"],
                "p2":        pred["p_away"],
                "hg":        row["hg"],
                "ag":        row["ag"],
            })
        except ValueError:
            pass
    df_preds = pd.DataFrame(preds)
    print(f"  Predicciones generadas: {len(df_preds)}")

    # 3. Descargar odds Pinnacle de football-data.co.uk
    print(f"  Descargando CSVs football-data.co.uk...")
    df_fd = fetch_fd_csv(league)
    if df_fd.empty:
        return {"error": "No se pudieron descargar los CSVs de football-data"}
    print(f"  Partidos en football-data: {len(df_fd)}")

    # 4. Join por (fecha, home_team, away_team)
    df_merged = df_preds.merge(
        df_fd,
        left_on=["date", "home_team", "away_team"],
        right_on=["date", "home_us", "away_us"],
        how="inner",
    )
    match_rate = len(df_merged) / len(df_preds) * 100 if len(df_preds) > 0 else 0
    print(f"  Partidos matched: {len(df_merged)} / {len(df_preds)}  ({match_rate:.1f}%)")

    if len(df_merged) < 5:
        return {"error": f"Muy pocos partidos matched ({len(df_merged)})"}

    # 5. Calcular P&L Kelly vs Pinnacle closing
    bankroll = BANKROLL0
    peak     = BANKROLL0
    max_dd   = 0.0
    bets     = []
    n_value  = 0

    for _, row in df_merged.iterrows():
        p1, px, p2  = row["p1"], row["px"], row["p2"]
        oh, od, oa  = row["odds_h"], row["odds_d"], row["odds_a"]
        hg, ag      = int(row["hg"]), int(row["ag"])

        result = "H" if hg > ag else ("D" if hg == ag else "A")

        for outcome, prob, odds in [("H", p1, oh), ("D", px, od), ("A", p2, oa)]:
            implied = 1.0 / odds
            edge    = prob - implied
            if edge <= MIN_EDGE:
                continue
            stake = kelly_stake(prob, odds, bankroll)
            if stake <= 0:
                continue
            n_value += 1
            win = (result == outcome)
            pnl = stake * (odds - 1) if win else -stake
            bankroll += pnl
            peak   = max(peak, bankroll)
            max_dd = max(max_dd, peak - bankroll)
            bets.append({
                "home": row["home_team"], "away": row["away_team"],
                "date": str(row["date"]),
                "outcome": outcome, "prob": round(prob, 4), "odds": round(odds, 3),
                "edge": round(edge, 4), "stake": stake, "win": win, "pnl": round(pnl, 2),
            })

    pnl_total = round(bankroll - BANKROLL0, 2)
    roi       = round(pnl_total / BANKROLL0 * 100, 2) if n_value > 0 else 0.0
    elapsed   = time.time() - t0

    # Estadisticas de valor
    df_bets = pd.DataFrame(bets) if bets else pd.DataFrame(
        columns=["outcome", "prob", "odds", "edge", "stake", "win", "pnl"])
    win_rate = df_bets["win"].mean() * 100 if len(df_bets) > 0 else 0
    avg_edge = df_bets["edge"].mean() * 100 if len(df_bets) > 0 else 0
    avg_odds = df_bets["odds"].mean() if len(df_bets) > 0 else 0

    print(f"\n  -- RESULTADOS vs PINNACLE CLOSING -------------------------")
    print(f"  Partidos evaluados     : {len(df_merged)}")
    print(f"  Apuestas con edge > 0  : {len(bets)}")
    print(f"  Win rate               : {win_rate:.1f}%")
    print(f"  Edge medio             : {avg_edge:+.2f}%")
    print(f"  Odds medias            : {avg_odds:.2f}")
    print(f"  P&L real               : EUR {pnl_total:+.2f}")
    print(f"  Max Drawdown           : EUR {max_dd:.2f}")
    print(f"  ROI                    : {roi:+.1f}%")
    print(f"  Tiempo                 : {elapsed:.1f}s")

    return {
        "league":       league,
        "n_test":       len(df_preds),
        "n_matched":    len(df_merged),
        "match_rate":   round(match_rate, 1),
        "n_bets":       len(bets),
        "win_rate":     round(win_rate, 1),
        "avg_edge":     round(avg_edge, 2),
        "avg_odds":     round(avg_odds, 3),
        "pnl":          pnl_total,
        "bankroll":     round(bankroll, 2),
        "max_dd":       round(max_dd, 2),
        "roi":          roi,
        "converged":    model["converged"],
        "elapsed_s":    round(elapsed, 1),
        "bets":         bets,
    }


# =============================================================================
# MAIN
# =============================================================================

results = {}
for league in LEAGUES:
    try:
        results[league] = run_league(league)
    except Exception as exc:
        import traceback
        print(f"\nERROR en {league}: {exc}")
        traceback.print_exc()
        results[league] = {"error": str(exc)}

# Guardar JSON con detalle de apuestas
def _conv(o):
    if isinstance(o, (np.floating,)):  return float(o)
    if isinstance(o, (np.integer,)):   return int(o)
    if isinstance(o, np.ndarray):      return o.tolist()
    raise TypeError(type(o))

out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pinnacle_bets.json")
with open(out_json, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2, default=_conv)

# =============================================================================
# TABLA RESUMEN
# =============================================================================
valid = [r for r in results.values() if "error" not in r]

print(f"\n\n{'='*80}")
print("  BACKTEST vs PINNACLE CLOSING — Temporada 25-26 (Kelly 0.25)")
print(f"{'='*80}")
hdr = (f"{'Liga':<14} {'Matched':>7} {'Match%':>6} {'Bets':>5} "
       f"{'WinRate':>8} {'AvgEdge':>8} {'AvgOdds':>8} {'P&L':>9} {'MaxDD':>8} {'ROI':>7}")
print(hdr)
print("-" * 80)
for league, r in results.items():
    if "error" in r:
        print(f"{league:<14}  ERROR: {r['error']}")
    else:
        print(
            f"{r['league']:<14} {r['n_matched']:>7} {r['match_rate']:>5.1f}% "
            f"{r['n_bets']:>5} {r['win_rate']:>7.1f}% "
            f"{r['avg_edge']:>+7.2f}% {r['avg_odds']:>8.3f} "
            f"EUR{r['pnl']:>+8.2f} EUR{r['max_dd']:>7.2f} {r['roi']:>+6.1f}%"
        )

if valid:
    tot_matched  = sum(r["n_matched"] for r in valid)
    tot_bets     = sum(r["n_bets"]    for r in valid)
    tot_pnl      = round(sum(r["pnl"] for r in valid), 2)
    all_bets     = [b for r in valid for b in r["bets"]]
    df_all       = pd.DataFrame(all_bets) if all_bets else pd.DataFrame()
    overall_wr   = df_all["win"].mean() * 100 if len(df_all) > 0 else 0
    overall_edge = df_all["edge"].mean() * 100 if len(df_all) > 0 else 0
    overall_odds = df_all["odds"].mean()       if len(df_all) > 0 else 0
    max_dd       = max(r["max_dd"] for r in valid)
    print("-" * 80)
    print(
        f"{'TOTAL':<14} {tot_matched:>7} {'':>6} "
        f"{tot_bets:>5} {overall_wr:>7.1f}% "
        f"{overall_edge:>+7.2f}% {overall_odds:>8.3f} "
        f"EUR{tot_pnl:>+8.2f} EUR{max_dd:>7.2f}"
    )

# =============================================================================
# GENERAR INFORME MARKDOWN
# =============================================================================
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "report_pinnacle_vs_gamma_Feb2026.md")

lines = [
    "# Backtest Dixon-Coles gamma_i vs Pinnacle Closing Odds",
    "## Temporada 2025-26 — Febrero 2026",
    "",
    f"**Fecha**: {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
    "**Modelo**: Dixon-Coles + xG, gamma_i por equipo, time decay 60 dias  ",
    "**Metodologia**: Split 70/30 temporal. Test set: oct 2025 - feb 2026.  ",
    "**Odds**: Pinnacle closing (PSCH/PSCD/PSCA) de football-data.co.uk  ",
    "**Kelly**: 0.25, max stake 5% bankroll, solo edge > 0 vs Pinnacle  ",
    "**Bankroll inicial por liga**: EUR 500",
    "",
    "---",
    "",
    "## Resultados por liga",
    "",
    "| Liga | Matched | Match% | Apuestas | Win% | Edge medio | Odds medias | P&L | Max DD | ROI |",
    "|------|--------:|-------:|---------:|-----:|-----------:|------------:|----:|-------:|----:|",
]

for league, r in results.items():
    if "error" in r:
        lines.append(f"| {league} | — | — | ERROR | — | — | — | — | — | {r['error']} |")
    else:
        lines.append(
            f"| {r['league']} | {r['n_matched']} | {r['match_rate']:.1f}% "
            f"| {r['n_bets']} | {r['win_rate']:.1f}% "
            f"| {r['avg_edge']:+.2f}% | {r['avg_odds']:.2f} "
            f"| EUR {r['pnl']:+.2f} | EUR {r['max_dd']:.2f} | {r['roi']:+.1f}% |"
        )

if valid:
    lines.append(
        f"| **TOTAL** | {tot_matched} | — "
        f"| {tot_bets} | {overall_wr:.1f}% "
        f"| {overall_edge:+.2f}% | {overall_odds:.2f} "
        f"| **EUR {tot_pnl:+.2f}** | EUR {max_dd:.2f} | — |"
    )

lines += [
    "",
    "---",
    "",
    "## Distribucion de apuestas por outcome",
    "",
]

if all_bets:
    df_all["result"] = df_all["win"].map({True: "win", False: "loss"})
    by_outcome = df_all.groupby("outcome").agg(
        n=("pnl", "count"),
        win_rate=("win", lambda x: x.mean() * 100),
        avg_edge=("edge", lambda x: x.mean() * 100),
        pnl=("pnl", "sum"),
    ).round(2)
    lines.append("| Outcome | N apuestas | Win% | Edge medio | P&L total |")
    lines.append("|---------|----------:|-----:|-----------:|----------:|")
    for out, row in by_outcome.iterrows():
        lines.append(
            f"| {out} | {int(row['n'])} | {row['win_rate']:.1f}% "
            f"| {row['avg_edge']:+.2f}% | EUR {row['pnl']:+.2f} |"
        )

lines += [
    "",
    "---",
    "",
    "## Notas metodologicas",
    "",
    "- **Odds usadas**: PSCH/PSCD/PSCA (Pinnacle closing). Si no disponibles, "
    "PSH/PSD/PSA (opening).",
    "- **Match rate**: join por (fecha exacta, home_team, away_team). "
    "Partidos sin match se excluyen.",
    "- **Kelly vs naive**: el P&L aqui es REAL (vs Pinnacle), no vs mercado naive.",
    "  Un edge positivo vs Pinnacle es un edge real.",
    "- **Limitacion**: la simulacion no descuenta comisiones ni considera "
    "liquidez de mercado.",
    "",
    f"*Detalle de apuestas en `research/pinnacle_bets.json`.*",
]

with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))

print(f"\nInforme guardado en: {REPORT}")
print(f"Detalle JSON en: {out_json}")
