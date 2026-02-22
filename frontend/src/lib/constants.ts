import type { DisciplineSettings, BetSource } from "./types";

export const DEFAULT_INITIAL_BANKROLL = 500;

export const DEFAULT_SETTINGS: DisciplineSettings = {
  dailyStopLoss: -60,
  weeklyStopLoss: -100,
  maxDrawdownPct: 0.20,
  maxDailyBets: 5,
  cooldownLosses: 2,
  cooldownHours: 2,
  kellyFraction: 0.25,
  maxStakePct: 0.05,
};

export const SOURCES: BetSource[] = ["Modelo", "Tipster", "Propia"];

export const MARKETS = ["1X2", "Over/Under", "Asian Handicap", "BTTS", "Double Chance", "Otro"] as const;

export const LEAGUES = [
  "EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1",
] as const;

export const STORAGE_KEY = "trading-deportivo-state";
export const SETTINGS_KEY = "trading-deportivo-settings";
