# -*- coding: utf-8 -*-
"""
trading-deportivo: Football prediction model using Dixon-Coles with xG data.
"""
from .config import SUPPORTED_LEAGUES

# Data
from .data import (
    fetch_all_shots,
    fetch_match_shots,
    get_league_match_ids,
    shots_to_df,
    build_match_xg_matrix,
    build_matches_with_dates,
    clear_cache,
)

# Model
from .model import (
    fit_dixon_coles_xg,
    predict_match,
    predict_matchday,
    export_predictions,
    kelly_fraction,
    save_model,
    load_model,
    list_models,
)

# Odds
from .odds import (
    fetch_ps3838_odds,
    send_telegram_alert,
)

# Betting
from .betting import (
    log_bet,
    log_bets_from_predictions,
    update_result,
    show_roi_stats,
    show_pending_bets,
)

# Backtest
from .backtest import (
    evaluate_model,
    temporal_validation,
    backtest_vs_market,
)

__version__ = "0.1.0"
