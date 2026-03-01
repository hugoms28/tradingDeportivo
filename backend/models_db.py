# -*- coding: utf-8 -*-
"""SQLAlchemy models for the trading-deportivo database."""
from datetime import datetime

from sqlalchemy import (
    Column, Integer, Text, Float, Boolean, DateTime, ForeignKey,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    league = Column(Text, nullable=False, index=True)
    run_id = Column(Text, nullable=False, index=True)
    home_team = Column(Text, nullable=False)
    away_team = Column(Text, nullable=False)
    p_home = Column(Float)
    p_draw = Column(Float)
    p_away = Column(Float)
    lambda_home = Column(Float)
    mu_away = Column(Float)
    best_ou_line = Column(Float)
    best_ou_prob = Column(Float)
    best_ah_line = Column(Float)
    best_ah_prob = Column(Float)
    odds_home = Column(Float)
    odds_draw = Column(Float)
    odds_away = Column(Float)
    edge_home = Column(Float)
    edge_draw = Column(Float)
    edge_away = Column(Float)
    recommended_bet = Column(Text)
    kelly_stake = Column(Float)
    raw_data = Column(Text)  # Full DataFrame row as JSON (all markets)
    created_at = Column(DateTime, default=datetime.utcnow)


class Bet(Base):
    __tablename__ = "bets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)
    event = Column(Text, nullable=False)
    league = Column(Text, index=True)
    source = Column(Text, nullable=False)  # Modelo / Tipster / Propia
    tipster_name = Column(Text, default="")
    market = Column(Text, nullable=False)
    pick = Column(Text, nullable=False)
    odds = Column(Float, nullable=False)
    model_prob = Column(Float)
    stake = Column(Float, nullable=False)
    edge = Column(Float)
    sport = Column(Text, nullable=True)           # Futbol / Baloncesto / etc.
    result = Column(Text)  # win / half_win / loss / half_loss / void / null
    pnl = Column(Float, default=0.0)
    closing_odds = Column(Float, nullable=True)   # Odd de cierre PS3838
    clv = Column(Float, nullable=True)            # CLV% = (entry/closing - 1)*100
    match_starts_at = Column(Text, nullable=True)  # ISO datetime from PS3838
    bookmaker = Column(Text, nullable=True)         # Casa de apuestas
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Text, primary_key=True)  # uuid
    league = Column(Text, nullable=False, index=True)
    status = Column(Text, nullable=False, default="running")
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    n_matches = Column(Integer)
    mse = Column(Float)
    converged = Column(Boolean)
    error = Column(Text)


class Setting(Base):
    __tablename__ = "settings"

    key = Column(Text, primary_key=True)
    value = Column(Text, nullable=False)  # JSON serialized
