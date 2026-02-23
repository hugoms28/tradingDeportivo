/**
 * API client for the trading-deportivo backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

// ─── Types ─────────────────────────────────────────────────────────────────

export interface ValueBet {
  market: string;
  label: string;
  edge: number;
  prob: number | null;
  odds: number | null;
  type: "principal" | "consejo";
}

export interface Prediction {
  id: number;
  league: string;
  run_id: string;
  home_team: string;
  away_team: string;
  p_home: number | null;
  p_draw: number | null;
  p_away: number | null;
  lambda_home: number | null;
  mu_away: number | null;
  odds_home: number | null;
  odds_draw: number | null;
  odds_away: number | null;
  edge_home: number | null;
  edge_draw: number | null;
  edge_away: number | null;
  recommended_bet: string | null;
  kelly_stake: number | null;
  value_bets: ValueBet[];
  raw: Record<string, unknown>;
  starts_at: string | null;
  created_at: string | null;
}

export interface ApiBet {
  id: number;
  predictionId: number | null;
  event: string;
  league: string;
  source: string;
  tipsterName: string;
  sport: string | null;
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
}

export interface BetStats {
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
  bankroll: number;
  initialBankroll: number;
  peakBankroll: number;
  avgClv: number | null;
  clvCount: number;
}

export interface ModelInfo {
  filename: string;
  league?: string;
  saved_at?: string;
  n_teams?: number;
  gamma?: number;
  rho?: number;
  mse?: number;
  converged?: boolean;
  teams?: string[];
}

export interface ModelRunInfo {
  id: string;
  league: string;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  n_matches: number | null;
  mse: number | null;
  converged: boolean | null;
  error: string | null;
}

export interface SchedulerStatus {
  enabled: boolean;
  config: { day_of_week: string; hour: number; minute: number };
  next_run: string | null;
  scheduler_running: boolean;
}

export interface ApiSettings {
  bankroll: string;
  initial_bankroll: string;
  peak_bankroll: string;
  discipline_settings: Record<string, number>;
}

// ─── Predictions ───────────────────────────────────────────────────────────

export async function startPredictions(leagues: string[]) {
  return fetchJSON<{ run_id: string; leagues: string[]; status: string }>(
    "/predict",
    { method: "POST", body: JSON.stringify({ leagues }) },
  );
}

export async function getPredictionStatus(runId: string) {
  return fetchJSON<Record<string, unknown>>(`/predict/${runId}/status`);
}

export async function getPredictions(league?: string, runId?: string) {
  const params = new URLSearchParams();
  if (league) params.set("league", league);
  if (runId) params.set("run_id", runId);
  const qs = params.toString();
  return fetchJSON<Prediction[]>(`/predictions${qs ? `?${qs}` : ""}`);
}

export async function getLatestPredictions() {
  return fetchJSON<Record<string, Prediction[]>>("/predictions/latest");
}

// ─── Bets ──────────────────────────────────────────────────────────────────

export async function getBets(filters?: {
  league?: string;
  status?: string;
}) {
  const params = new URLSearchParams();
  if (filters?.league) params.set("league", filters.league);
  if (filters?.status) params.set("status", filters.status);
  const qs = params.toString();
  return fetchJSON<ApiBet[]>(`/bets${qs ? `?${qs}` : ""}`);
}

export async function createBet(data: {
  event: string;
  league: string;
  source: string;
  tipster_name?: string;
  sport?: string | null;
  market: string;
  pick: string;
  odds: number;
  model_prob?: number | null;
  stake: number;
  edge?: number | null;
  prediction_id?: number | null;
  match_starts_at?: string | null;
}) {
  return fetchJSON<ApiBet>("/bets", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function triggerAutoResolve() {
  return fetchJSON<{ resolved: number; skipped: number; errors: number }>(
    "/bets/auto-resolve",
    { method: "POST" },
  );
}

export async function resolveBet(id: number, result: "win" | "half_win" | "loss" | "half_loss" | "void") {
  return fetchJSON<ApiBet>(`/bets/${id}/resolve`, {
    method: "PATCH",
    body: JSON.stringify({ result }),
  });
}

export async function getBetStats() {
  return fetchJSON<BetStats>("/bets/stats");
}

// ─── Models ────────────────────────────────────────────────────────────────

export async function listModels(league?: string) {
  const qs = league ? `?league=${league}` : "";
  return fetchJSON<ModelInfo[]>(`/models${qs}`);
}

export async function getLatestModel(league: string) {
  return fetchJSON<ModelInfo>(`/models/${league}/latest`);
}

export interface TrainingLeagueProgress {
  league: string;
  step: string;
  step_label: string;
  pct: number;
  status: string;
  error: string | null;
  started_at: string;
}

export interface TrainingProgress {
  train_id: string;
  status: string;
  leagues: Record<string, TrainingLeagueProgress>;
}

export async function startTraining(leagues: string[]) {
  return fetchJSON<{ status: string; train_id: string; leagues: string[] }>("/train", {
    method: "POST",
    body: JSON.stringify({ leagues }),
  });
}

export async function getTrainingProgress(trainId: string) {
  return fetchJSON<TrainingProgress>(`/train/${trainId}/progress`);
}

export async function getModelRuns(league?: string) {
  const qs = league ? `?league=${league}` : "";
  return fetchJSON<ModelRunInfo[]>(`/model-runs${qs}`);
}

// ─── Settings ──────────────────────────────────────────────────────────────

export async function getSettings() {
  return fetchJSON<ApiSettings>("/settings");
}

export async function updateSettings(data: {
  bankroll?: number;
  initial_bankroll?: number;
  peak_bankroll?: number;
  discipline_settings?: Record<string, number>;
}) {
  return fetchJSON<{ status: string; updated: string[] }>("/settings", {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

// ─── Scheduler ─────────────────────────────────────────────────────────────

export async function getSchedulerStatus() {
  return fetchJSON<SchedulerStatus>("/scheduler/status");
}

export async function toggleScheduler() {
  return fetchJSON<{ enabled: boolean; message: string }>("/scheduler/toggle", {
    method: "POST",
  });
}

export async function updateSchedulerConfig(config: {
  day_of_week?: string;
  hour?: number;
  minute?: number;
}) {
  return fetchJSON<{ config: Record<string, unknown>; enabled: boolean }>(
    "/scheduler/config",
    { method: "PUT", body: JSON.stringify(config) },
  );
}

// ─── Health ────────────────────────────────────────────────────────────────

export async function checkHealth() {
  return fetchJSON<{ status: string }>("/health");
}
