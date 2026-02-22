"use client";

import { useState, useMemo } from "react";
import type { Bet } from "@/lib/types";
import { formatCurrency, formatDate } from "@/lib/format";

const LEAGUE_LABELS: Record<string, string> = {
  EPL: "Premier League",
  La_Liga: "La Liga",
  Bundesliga: "Bundesliga",
  Serie_A: "Serie A",
  Ligue_1: "Ligue 1",
};

interface Props {
  bets: Bet[];
  onResolveBet: (id: number, result: "win" | "half_win" | "loss" | "half_loss" | "void") => void;
}

export function History({ bets, onResolveBet }: Props) {
  const [selectedLeague, setSelectedLeague] = useState<string>("Todas");

  const leagues = useMemo(() => {
    const seen = new Set<string>();
    for (const b of bets) {
      if (b.league) seen.add(b.league);
    }
    return Array.from(seen).sort();
  }, [bets]);

  const sorted = useMemo(() => {
    const filtered =
      selectedLeague === "Todas" ? bets : bets.filter((b) => b.league === selectedLeague);
    return [...filtered].sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }, [bets, selectedLeague]);

  return (
    <div className="bg-[#111827] border border-slate-800 rounded-xl p-5">
      <div className="flex items-center justify-between flex-wrap gap-3 mb-4">
        <div className="text-[15px] font-bold font-[family-name:var(--font-display)]">
          Historial ({sorted.length}{selectedLeague !== "Todas" ? `/${bets.length}` : ""} apuestas)
        </div>
        {leagues.length > 0 && (
          <div className="flex items-center gap-1 flex-wrap">
            {["Todas", ...leagues].map((league) => (
              <button
                key={league}
                onClick={() => setSelectedLeague(league)}
                className={`text-[11px] font-semibold px-2.5 py-1 rounded-full transition ${
                  selectedLeague === league
                    ? "bg-indigo-600 text-white"
                    : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200"
                }`}
              >
                {league === "Todas" ? "Todas" : (LEAGUE_LABELS[league] ?? league)}
              </button>
            ))}
          </div>
        )}
      </div>

      {sorted.length === 0 ? (
        <div className="text-center py-10 text-slate-500 text-sm">Sin apuestas registradas</div>
      ) : (
        <div className="flex flex-col gap-1.5">
          {sorted.map((bet) => {
            const borderColor =
              bet.result === "win"
                ? "border-l-emerald-500"
                : bet.result === "half_win"
                  ? "border-l-emerald-700"
                  : bet.result === "loss"
                    ? "border-l-red-500"
                    : bet.result === "half_loss"
                      ? "border-l-red-800"
                      : bet.result === "void"
                        ? "border-l-slate-600"
                        : "border-l-amber-500";

            return (
              <div
                key={bet.id}
                className={`bg-[#0f172a] rounded-lg px-4 py-3 border-l-[3px] ${borderColor} flex justify-between items-center flex-wrap gap-2`}
              >
                <div>
                  <div className="text-sm font-semibold">
                    {bet.event}
                    {bet.matchStartsAt && (
                      <span className="ml-2 text-xs font-normal text-slate-400">
                        {formatDate(bet.matchStartsAt)}
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-slate-500">
                    {bet.pick} @ {bet.odds} · {bet.stake.toFixed(2)}€ · {bet.source}
                    {bet.tipsterName ? ` (${bet.tipsterName})` : ""} · {bet.market}
                    {bet.league ? ` · ${bet.league}` : ""}
                  </div>
                  <div className="text-[10px] text-slate-700 mt-0.5">{formatDate(bet.timestamp)}</div>
                </div>
                <div className="text-right">
                  {bet.result ? (
                    <>
                      <div
                        className="text-sm font-bold"
                        style={{ color: bet.pnl >= 0 ? "#10b981" : "#ef4444" }}
                      >
                        {formatCurrency(bet.pnl)}
                      </div>
                      <div className="flex items-center justify-end gap-2 mt-0.5">
                        <div
                          className={`text-[10px] font-semibold uppercase ${
                            bet.result === "win"
                              ? "text-emerald-500"
                              : bet.result === "half_win"
                                ? "text-emerald-700"
                                : bet.result === "loss"
                                  ? "text-red-500"
                                  : bet.result === "half_loss"
                                    ? "text-red-700"
                                    : "text-slate-500"
                          }`}
                        >
                          {bet.result === "win"
                            ? "Ganada"
                            : bet.result === "half_win"
                              ? "½ Ganada"
                              : bet.result === "loss"
                                ? "Perdida"
                                : bet.result === "half_loss"
                                  ? "½ Perdida"
                                  : "Nula"}
                        </div>
                        {bet.clv !== null && bet.clv !== undefined && (
                          <span
                            className={`text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded ${
                              bet.clv >= 0
                                ? "bg-emerald-950 text-emerald-400"
                                : "bg-red-950 text-red-400"
                            }`}
                            title={`CLV: odd entrada ${bet.odds} / cierre ${bet.closingOdds?.toFixed(2)}`}
                          >
                            CLV {bet.clv >= 0 ? "+" : ""}{bet.clv.toFixed(1)}%
                          </span>
                        )}
                      </div>
                    </>
                  ) : (
                    <div className="flex items-center gap-2 flex-wrap justify-end">
                      <span className="text-[10px] font-semibold text-amber-500 uppercase">
                        Pendiente
                      </span>
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
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
