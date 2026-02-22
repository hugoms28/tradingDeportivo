import type { Bet, BettingState, DisciplineSettings, LockStatus, Stats, TrafficLight } from "./types";
import { SOURCES } from "./constants";

export function calcKellyStake(
  bankroll: number,
  modelProb: number,
  odds: number,
  kellyFraction: number,
  maxStakePct: number,
): number {
  const q = 1 - modelProb;
  const b = odds - 1;
  const kelly = (modelProb * b - q) / b;
  if (kelly <= 0) return 0;
  const fractionalKelly = kelly * kellyFraction;
  const maxStake = bankroll * maxStakePct;
  return Math.min(fractionalKelly * bankroll, maxStake);
}

export function calcEdge(modelProb: number, odds: number): number {
  return (modelProb * odds - 1) * 100;
}

export function computeStats(state: BettingState): Stats {
  const resolved = state.bets.filter((b) => b.result !== null);
  const today = new Date().toDateString();
  const todayBets = state.bets.filter(
    (b) => new Date(b.timestamp).toDateString() === today,
  );
  const todayResolved = todayBets.filter((b) => b.result !== null);
  const todayPnL = todayResolved.reduce((s, b) => s + b.pnl, 0);

  const weekStart = new Date();
  weekStart.setDate(weekStart.getDate() - weekStart.getDay() + 1);
  weekStart.setHours(0, 0, 0, 0);
  const weekBets = state.bets.filter(
    (b) => new Date(b.timestamp) >= weekStart && b.result !== null,
  );
  const weekPnL = weekBets.reduce((s, b) => s + b.pnl, 0);

  const totalPnL = resolved.reduce((s, b) => s + b.pnl, 0);
  const wins = resolved.filter((b) => b.result === "win" || b.result === "half_win").length;
  const losses = resolved.filter((b) => b.result === "loss" || b.result === "half_loss").length;
  const totalStaked = resolved.reduce((s, b) => s + b.stake, 0);
  const winRate = resolved.length > 0 ? (wins / resolved.length) * 100 : 0;
  const roi = totalStaked > 0 ? (totalPnL / totalStaked) * 100 : 0;

  const drawdownFromPeak =
    state.peakBankroll > 0
      ? (state.peakBankroll - state.bankroll) / state.peakBankroll
      : 0;

  let consecutiveLosses = 0;
  for (let i = resolved.length - 1; i >= 0; i--) {
    if (resolved[i].result === "loss" || resolved[i].result === "half_loss") consecutiveLosses++;
    else break;
  }

  const bySource: Stats["bySource"] = {};
  SOURCES.forEach((s) => {
    const sb = resolved.filter((b) => b.source === s);
    bySource[s] = {
      count: sb.length,
      pnl: sb.reduce((acc, b) => acc + b.pnl, 0),
      winRate:
        sb.length > 0
          ? (sb.filter((b) => b.result === "win" || b.result === "half_win").length / sb.length) * 100
          : 0,
    };
  });

  const clvBets = resolved.filter((b) => b.clv !== null && b.clv !== undefined);
  const avgClv = clvBets.length > 0
    ? clvBets.reduce((s, b) => s + (b.clv ?? 0), 0) / clvBets.length
    : null;

  return {
    totalPnL,
    wins,
    losses,
    winRate,
    roi,
    todayPnL,
    weekPnL,
    todayBetsCount: todayBets.length,
    drawdownFromPeak,
    consecutiveLosses,
    bySource,
    resolvedCount: resolved.length,
    pendingCount: state.bets.filter((b) => b.result === null).length,
    avgClv,
    clvCount: clvBets.length,
  };
}

export function computeLockStatus(
  state: BettingState,
  stats: Stats,
  settings: DisciplineSettings,
): LockStatus {
  const now = new Date();

  if (state.lockedUntil && new Date(state.lockedUntil) > now) {
    return {
      locked: true,
      reason: state.lockReason || "Bloqueado manualmente",
      until: state.lockedUntil,
    };
  }

  if (stats.todayPnL <= settings.dailyStopLoss) {
    return {
      locked: true,
      reason: `Stop-loss diario alcanzado (${stats.todayPnL.toFixed(2)}\u20AC)`,
      until: "fin del d\u00EDa",
    };
  }

  if (stats.weekPnL <= settings.weeklyStopLoss) {
    return {
      locked: true,
      reason: `Stop-loss semanal alcanzado (${stats.weekPnL.toFixed(2)}\u20AC)`,
      until: "siguiente semana",
    };
  }

  if (stats.drawdownFromPeak >= settings.maxDrawdownPct) {
    return {
      locked: true,
      reason: `Drawdown m\u00E1ximo (${(stats.drawdownFromPeak * 100).toFixed(1)}%) desde pico de ${state.peakBankroll.toFixed(2)}\u20AC`,
      until: "revisi\u00F3n manual",
    };
  }

  if (stats.todayBetsCount >= settings.maxDailyBets) {
    return {
      locked: true,
      reason: `M\u00E1ximo de apuestas diarias (${settings.maxDailyBets})`,
      until: "ma\u00F1ana",
    };
  }

  if (stats.consecutiveLosses >= settings.cooldownLosses) {
    const lastLoss = state.bets.filter((b) => b.result === "loss" || b.result === "half_loss").slice(-1)[0];
    if (lastLoss?.resolvedAt) {
      const cooldownEnd = new Date(
        new Date(lastLoss.resolvedAt).getTime() +
          settings.cooldownHours * 3600000,
      );
      if (now < cooldownEnd) {
        const mins = Math.ceil((cooldownEnd.getTime() - now.getTime()) / 60000);
        return {
          locked: true,
          reason: `Cooldown anti-tilt (${stats.consecutiveLosses} p\u00E9rdidas seguidas)`,
          until: `${mins} minutos`,
        };
      }
    }
  }

  return { locked: false };
}

export function computeTrafficLight(
  lockStatus: LockStatus,
  stats: Stats,
  settings: DisciplineSettings,
): TrafficLight {
  if (lockStatus.locked) return "red";
  if (stats.drawdownFromPeak >= settings.maxDrawdownPct * 0.7) return "orange";
  if (stats.todayPnL <= settings.dailyStopLoss * 0.7) return "orange";
  if (stats.weekPnL <= settings.weeklyStopLoss * 0.7) return "orange";
  if (stats.consecutiveLosses >= settings.cooldownLosses - 1) return "orange";
  if (stats.todayBetsCount >= settings.maxDailyBets - 1) return "orange";
  return "green";
}

export function equityCurve(bets: Bet[], initialBankroll: number): number[] {
  const curve = [initialBankroll];
  let running = initialBankroll;
  bets.forEach((b) => {
    if (b.result !== null) {
      running += b.pnl;
      curve.push(running);
    }
  });
  return curve;
}
