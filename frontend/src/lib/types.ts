export interface Bet {
  id: number;
  event: string;
  source: BetSource;
  tipsterName: string;
  market: string;
  pick: string;
  odds: number;
  modelProb: number | null;
  stake: number;
  edge: number | null;
  result: "win" | "half_win" | "loss" | "half_loss" | "void" | null;
  pnl: number;
  closingOdds: number | null;
  clv: number | null;
  matchStartsAt: string | null;
  timestamp: string;
  resolvedAt: string | null;
  league?: string;
}

export type BetSource = "Modelo" | "Tipster" | "Propia";

export interface DisciplineSettings {
  dailyStopLoss: number;
  weeklyStopLoss: number;
  maxDrawdownPct: number;
  maxDailyBets: number;
  cooldownLosses: number;
  cooldownHours: number;
  kellyFraction: number;
  maxStakePct: number;
}

export interface BettingState {
  bankroll: number;
  initialBankroll: number;
  peakBankroll: number;
  bets: Bet[];
  lockedUntil: string | null;
  lockReason: string | null;
}

export interface LockStatus {
  locked: boolean;
  reason?: string;
  until?: string;
}

export interface BetFormData {
  event: string;
  source: BetSource;
  tipsterName: string;
  modelProb: string;
  odds: string;
  stake: string;
  market: string;
  pick: string;
  league: string;
}

export type TrafficLight = "green" | "orange" | "red";

export interface Stats {
  totalPnL: number;
  wins: number;
  losses: number;
  winRate: number;
  roi: number;
  todayPnL: number;
  weekPnL: number;
  todayBetsCount: number;
  drawdownFromPeak: number;
  consecutiveLosses: number;
  bySource: Record<string, { count: number; pnl: number; winRate: number }>;
  resolvedCount: number;
  pendingCount: number;
  avgClv: number | null;
  clvCount: number;
}
