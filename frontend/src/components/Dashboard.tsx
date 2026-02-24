"use client";

import { useMemo } from "react";
import type { Bet, BettingState, DisciplineSettings, LockStatus, Stats } from "@/lib/types";
import { formatCurrency } from "@/lib/format";
import { StatCard } from "./StatCard";
import { EquityChart } from "./EquityChart";

interface Props {
  state: BettingState;
  stats: Stats;
  settings: DisciplineSettings;
  lockStatus: LockStatus;
  onResolveBet: (id: number, result: "win" | "half_win" | "loss" | "half_loss" | "void") => void;
  onUnlock: () => void;
}

function computeGroupStats(bets: Bet[]) {
  const resolved = bets.filter((b) => b.result !== null);
  const pnl = resolved.reduce((s, b) => s + b.pnl, 0);
  const staked = resolved.reduce((s, b) => s + b.stake, 0);
  const wins = resolved.filter((b) => b.result === "win" || b.result === "half_win").length;
  const roi = staked > 0 ? (pnl / staked) * 100 : 0;
  const winRate = resolved.length > 0 ? (wins / resolved.length) * 100 : 0;
  const clvBets = resolved.filter((b) => b.clv != null);
  const avgClv = clvBets.length > 0
    ? clvBets.reduce((s, b) => s + (b.clv ?? 0), 0) / clvBets.length
    : null;
  return {
    pnl,
    roi,
    winRate,
    resolved: resolved.length,
    total: bets.length,
    pending: bets.filter((b) => b.result === null).length,
    avgClv,
    clvCount: clvBets.length,
  };
}

export function Dashboard({ state, stats, settings, lockStatus, onResolveBet, onUnlock }: Props) {
  const pendingBets = state.bets.filter((b) => b.result === null);

  const realStats = useMemo(
    () => computeGroupStats(state.bets.filter((b) => b.source === "Tipster" || b.source === "Propia")),
    [state.bets],
  );
  const modelStats = useMemo(
    () => computeGroupStats(state.bets.filter((b) => b.source === "Modelo")),
    [state.bets],
  );

  return (
    <div className="flex flex-col gap-5">
      {/* Lock banner */}
      {lockStatus.locked && (
        <div className="bg-gradient-to-r from-red-950 to-red-900 rounded-xl p-5 border border-red-700 flex justify-between items-center flex-wrap gap-3">
          <div>
            <div className="text-[15px] font-bold mb-1">SISTEMA BLOQUEADO</div>
            <div className="text-xs text-red-300">{lockStatus.reason}</div>
            <div className="text-[11px] text-red-400 mt-1">Hasta: {lockStatus.until}</div>
          </div>
          <button
            onClick={onUnlock}
            className="border border-red-700 text-red-300 rounded-lg px-4 py-2 text-xs hover:bg-red-900/50 transition"
          >
            Desbloquear ⚠️
          </button>
        </div>
      )}

      {/* Stats grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        <StatCard
          label="Bankroll"
          value={`${state.bankroll.toFixed(2)}€`}
          color={state.bankroll >= state.initialBankroll ? "#10b981" : "#ef4444"}
        />
        <StatCard
          label="P&L Total"
          value={formatCurrency(stats.totalPnL)}
          color={stats.totalPnL >= 0 ? "#10b981" : "#ef4444"}
        />
        <StatCard
          label="ROI"
          value={`${stats.roi.toFixed(1)}%`}
          color={stats.roi >= 0 ? "#10b981" : "#ef4444"}
        />
        <StatCard
          label="Win Rate"
          value={`${stats.winRate.toFixed(0)}%`}
          color={stats.winRate >= 50 ? "#10b981" : "#f59e0b"}
        />
        <StatCard
          label="Hoy"
          value={formatCurrency(stats.todayPnL)}
          color={stats.todayPnL >= 0 ? "#10b981" : "#ef4444"}
        />
        <StatCard
          label="Drawdown"
          value={`${(stats.drawdownFromPeak * 100).toFixed(1)}%`}
          color={stats.drawdownFromPeak > 0.1 ? "#ef4444" : stats.drawdownFromPeak > 0.05 ? "#f59e0b" : "#10b981"}
        />
        <StatCard
          label="CLV Medio"
          value={
            stats.avgClv !== null
              ? `${stats.avgClv >= 0 ? "+" : ""}${stats.avgClv.toFixed(1)}%`
              : "—"
          }
          subtitle={stats.clvCount > 0 ? `${stats.clvCount} apuestas` : "sin datos"}
          color={
            stats.avgClv === null
              ? "#64748b"
              : stats.avgClv >= 1
              ? "#10b981"
              : stats.avgClv >= 0
              ? "#f59e0b"
              : "#ef4444"
          }
        />
        <StatCard
          label="Semana"
          value={formatCurrency(stats.weekPnL)}
          color={stats.weekPnL >= 0 ? "#10b981" : "#ef4444"}
        />
      </div>

      {/* Equity chart */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5">
        <div className="text-[11px] text-slate-500 uppercase tracking-widest mb-4">Equity Curve</div>
        <EquityChart bets={state.bets} initialBankroll={state.initialBankroll} />
      </div>

      {/* Stats por grupo */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

        {/* Dinero Real */}
        <div className="bg-[#111827] border border-emerald-900/50 rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <div className="text-[11px] text-emerald-500 uppercase tracking-widest font-semibold">
              Dinero Real — Tipster / Propia
            </div>
            <span className="text-[10px] bg-emerald-950 text-emerald-400 px-2 py-0.5 rounded-full font-semibold">
              {realStats.total} apuestas
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-[#0f172a] rounded-lg p-3 text-center">
              <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">P&L</div>
              <div className="text-lg font-bold font-[family-name:var(--font-display)]"
                style={{ color: realStats.pnl >= 0 ? "#10b981" : "#ef4444" }}>
                {formatCurrency(realStats.pnl)}
              </div>
            </div>
            <div className="bg-[#0f172a] rounded-lg p-3 text-center">
              <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">ROI</div>
              <div className="text-lg font-bold font-[family-name:var(--font-display)]"
                style={{ color: realStats.roi >= 0 ? "#10b981" : "#ef4444" }}>
                {realStats.roi.toFixed(1)}%
              </div>
            </div>
            <div className="bg-[#0f172a] rounded-lg p-3 text-center">
              <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">Win Rate</div>
              <div className="text-lg font-bold font-[family-name:var(--font-display)]"
                style={{ color: realStats.winRate >= 50 ? "#10b981" : "#f59e0b" }}>
                {realStats.winRate.toFixed(0)}%
              </div>
            </div>
            <div className="bg-[#0f172a] rounded-lg p-3 text-center">
              <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">Resueltas</div>
              <div className="text-lg font-bold text-slate-300 font-[family-name:var(--font-display)]">
                {realStats.resolved}
                <span className="text-slate-600 text-xs font-normal">/{realStats.total}</span>
              </div>
            </div>
          </div>
          {realStats.avgClv !== null && (
            <div className="mt-3 bg-[#0f172a] rounded-lg px-4 py-2.5 flex justify-between items-center">
              <span className="text-[10px] text-slate-500 uppercase tracking-widest">CLV Medio</span>
              <span className="text-sm font-bold font-mono"
                style={{ color: realStats.avgClv >= 0 ? "#10b981" : "#ef4444" }}>
                {realStats.avgClv >= 0 ? "+" : ""}{realStats.avgClv.toFixed(1)}%
                <span className="text-[10px] text-slate-600 ml-1 font-normal">({realStats.clvCount})</span>
              </span>
            </div>
          )}
        </div>

        {/* Modelo */}
        <div className="bg-[#111827] border border-indigo-900/50 rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <div className="text-[11px] text-indigo-400 uppercase tracking-widest font-semibold">
              Modelo
            </div>
            <span className="text-[10px] bg-indigo-950 text-indigo-400 px-2 py-0.5 rounded-full font-semibold">
              {modelStats.total} apuestas
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-[#0f172a] rounded-lg p-3 text-center">
              <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">P&L</div>
              <div className="text-lg font-bold font-[family-name:var(--font-display)]"
                style={{ color: modelStats.pnl >= 0 ? "#10b981" : "#ef4444" }}>
                {formatCurrency(modelStats.pnl)}
              </div>
            </div>
            <div className="bg-[#0f172a] rounded-lg p-3 text-center">
              <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">ROI</div>
              <div className="text-lg font-bold font-[family-name:var(--font-display)]"
                style={{ color: modelStats.roi >= 0 ? "#10b981" : "#ef4444" }}>
                {modelStats.roi.toFixed(1)}%
              </div>
            </div>
            <div className="bg-[#0f172a] rounded-lg p-3 text-center">
              <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">Win Rate</div>
              <div className="text-lg font-bold font-[family-name:var(--font-display)]"
                style={{ color: modelStats.winRate >= 50 ? "#10b981" : "#f59e0b" }}>
                {modelStats.winRate.toFixed(0)}%
              </div>
            </div>
            <div className="bg-[#0f172a] rounded-lg p-3 text-center">
              <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">Resueltas</div>
              <div className="text-lg font-bold text-slate-300 font-[family-name:var(--font-display)]">
                {modelStats.resolved}
                <span className="text-slate-600 text-xs font-normal">/{modelStats.total}</span>
              </div>
            </div>
          </div>
          {modelStats.avgClv !== null && (
            <div className="mt-3 bg-[#0f172a] rounded-lg px-4 py-2.5 flex justify-between items-center">
              <span className="text-[10px] text-slate-500 uppercase tracking-widest">CLV Medio</span>
              <span className="text-sm font-bold font-mono"
                style={{ color: modelStats.avgClv >= 0 ? "#10b981" : "#ef4444" }}>
                {modelStats.avgClv >= 0 ? "+" : ""}{modelStats.avgClv.toFixed(1)}%
                <span className="text-[10px] text-slate-600 ml-1 font-normal">({modelStats.clvCount})</span>
              </span>
            </div>
          )}
        </div>

      </div>

      {/* Pending bets */}
      {pendingBets.length > 0 && (
        <div className="bg-[#111827] border border-slate-800 rounded-xl p-5">
          <div className="text-[11px] text-slate-500 uppercase tracking-widest mb-4">
            Apuestas Pendientes ({pendingBets.length})
          </div>
          <div className="flex flex-col gap-2">
            {pendingBets.map((bet) => (
              <div
                key={bet.id}
                className="bg-[#0f172a] rounded-lg px-4 py-3 flex justify-between items-center flex-wrap gap-2"
              >
                <div>
                  <div className="text-sm font-semibold">{bet.event}</div>
                  <div className="text-xs text-slate-500">
                    {bet.pick} @ {bet.odds} &middot; {bet.stake.toFixed(2)}€ &middot; {bet.source}
                    {bet.tipsterName ? ` (${bet.tipsterName})` : ""} &middot; {bet.market}
                    {bet.league ? ` &middot; ${bet.league}` : ""}
                  </div>
                </div>
                <div className="flex gap-1">
                  <button
                    onClick={() => onResolveBet(bet.id, "win")}
                    className="border border-emerald-600 text-emerald-400 rounded px-2.5 py-1 text-[10px] hover:bg-emerald-950 transition"
                    title="Ganada"
                  >
                    G
                  </button>
                  <button
                    onClick={() => onResolveBet(bet.id, "half_win")}
                    className="border border-emerald-800 text-emerald-600 rounded px-2 py-1 text-[10px] hover:bg-emerald-950 transition"
                    title="½ Ganada"
                  >
                    ½G
                  </button>
                  <button
                    onClick={() => onResolveBet(bet.id, "half_loss")}
                    className="border border-red-800 text-red-600 rounded px-2 py-1 text-[10px] hover:bg-red-950 transition"
                    title="½ Perdida"
                  >
                    ½P
                  </button>
                  <button
                    onClick={() => onResolveBet(bet.id, "loss")}
                    className="border border-red-600 text-red-400 rounded px-2.5 py-1 text-[10px] hover:bg-red-950 transition"
                    title="Perdida"
                  >
                    P
                  </button>
                  <button
                    onClick={() => onResolveBet(bet.id, "void")}
                    className="border border-slate-700 text-slate-500 rounded px-2 py-1 text-[10px] hover:bg-slate-800 transition"
                    title="Nula"
                  >
                    N
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Quick status */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-4 grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="text-xs text-slate-500">
          Apuestas hoy:{" "}
          <span className={`font-semibold ${stats.todayBetsCount >= settings.maxDailyBets ? "text-red-400" : "text-slate-200"}`}>
            {stats.todayBetsCount}/{settings.maxDailyBets}
          </span>
        </div>
        <div className="text-xs text-slate-500">
          Rachas negativas:{" "}
          <span className={`font-semibold ${stats.consecutiveLosses >= settings.cooldownLosses ? "text-red-400" : "text-slate-200"}`}>
            {stats.consecutiveLosses}
          </span>
        </div>
        <div className="text-xs text-slate-500">
          P&L semana:{" "}
          <span className={`font-semibold ${stats.weekPnL >= 0 ? "text-emerald-400" : "text-red-400"}`}>
            {formatCurrency(stats.weekPnL)}
          </span>
        </div>
        <div className="text-xs text-slate-500">
          Pico: <span className="font-semibold text-slate-200">{state.peakBankroll.toFixed(2)}€</span>
        </div>
      </div>
    </div>
  );
}
