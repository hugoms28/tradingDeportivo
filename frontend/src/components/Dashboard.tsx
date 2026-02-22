"use client";

import type { BettingState, DisciplineSettings, LockStatus, Stats } from "@/lib/types";
import { formatCurrency } from "@/lib/format";
import { StatCard } from "./StatCard";
import { EquityChart } from "./EquityChart";
import { SOURCES } from "@/lib/constants";

interface Props {
  state: BettingState;
  stats: Stats;
  settings: DisciplineSettings;
  lockStatus: LockStatus;
  onResolveBet: (id: number, result: "win" | "half_win" | "loss" | "half_loss" | "void") => void;
  onUnlock: () => void;
}

export function Dashboard({ state, stats, settings, lockStatus, onResolveBet, onUnlock }: Props) {
  const pendingBets = state.bets.filter((b) => b.result === null);

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

      {/* P&L by source */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5">
        <div className="text-[11px] text-slate-500 uppercase tracking-widest mb-4">P&L por Fuente</div>
        <div className="grid grid-cols-3 gap-3">
          {SOURCES.map((s) => (
            <div key={s} className="bg-[#0f172a] rounded-lg p-4 text-center">
              <div className="text-xs text-slate-500 mb-2">{s}</div>
              <div
                className="text-lg font-bold font-[family-name:var(--font-display)]"
                style={{ color: stats.bySource[s]?.pnl >= 0 ? "#10b981" : "#ef4444" }}
              >
                {formatCurrency(stats.bySource[s]?.pnl ?? 0)}
              </div>
              <div className="text-[10px] text-slate-500 mt-1">
                {stats.bySource[s]?.count ?? 0} apuestas &middot; {(stats.bySource[s]?.winRate ?? 0).toFixed(0)}% WR
              </div>
            </div>
          ))}
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
