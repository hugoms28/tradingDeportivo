# -*- coding: utf-8 -*-
"""
Configuracion central del paquete trading-deportivo.
Constantes, rutas y variables de entorno.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# LIGAS SOPORTADAS
# =============================================================================
SUPPORTED_LEAGUES = ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"]

# =============================================================================
# CACHE
# =============================================================================
CACHE_DIR = os.getenv("TRADING_CACHE_DIR", "cache")
CACHE_TTL_HOURS = int(os.getenv("TRADING_CACHE_TTL", "48"))

# =============================================================================
# MODELOS
# =============================================================================
MODELS_DIR = os.getenv("TRADING_MODELS_DIR", "models")

# =============================================================================
# APUESTAS
# =============================================================================
BETS_FILE = os.getenv("TRADING_BETS_FILE", "bets_history.csv")

# =============================================================================
# MODELO DIXON-COLES (defaults)
# =============================================================================
DEFAULT_REG = 0.001
DEFAULT_HALF_LIFE = 60
DEFAULT_MAX_GOALS = 10
DEFAULT_MIN_EDGE = 0.03
DEFAULT_KELLY_FRACTION = 0.25

# =============================================================================
# PS3838 (PINNACLE)
# =============================================================================
PS3838_USERNAME = os.getenv("PS3838_USERNAME")
PS3838_PASSWORD = os.getenv("PS3838_PASSWORD")
PS3838_BASE_URL = "https://api.ps3838.com"

# =============================================================================
# TELEGRAM
# =============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# =============================================================================
# THE-ODDS-API (legacy, mantenido por compatibilidad)
# =============================================================================
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
