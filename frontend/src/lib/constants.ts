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

export const SPORTS = ["Fútbol", "Baloncesto", "Tenis", "Esports", "Dardos", "Hockey", "Combi"] as const;

export const LEAGUES = [
  "EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1",
] as const;

export const BOOKMAKERS = ["PS3838", "Bet365", "Winamax"] as const;
export type Bookmaker = typeof BOOKMAKERS[number];

export const BOOKMAKER_COLORS: Record<string, { bg: string; text: string; activeBg: string }> = {
  "PS3838":  { bg: "bg-violet-900", text: "text-violet-300", activeBg: "bg-violet-700" },
  "Bet365":  { bg: "bg-green-900",  text: "text-green-300",  activeBg: "bg-green-700" },
  "Winamax": { bg: "bg-rose-900",   text: "text-rose-300",   activeBg: "bg-rose-700" },
};

export const STORAGE_KEY = "trading-deportivo-state";
export const SETTINGS_KEY = "trading-deportivo-settings";
