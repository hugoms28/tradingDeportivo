"use client";

import { useState, useMemo } from "react";
import type { Bet } from "@/lib/types";
import { formatCurrency, formatDate } from "@/lib/format";
import { SOURCES, BOOKMAKERS, BOOKMAKER_COLORS } from "@/lib/constants";

const LEAGUE_LABELS: Record<string, string> = {
  EPL: "Premier League",
  La_Liga: "La Liga",
  Bundesliga: "Bundesliga",
  Serie_A: "Serie A",
  Ligue_1: "Ligue 1",
};

const MONTH_NAMES = [
  "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
  "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
];

function yearMonthKey(date: Date): string {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`;
}

// Devuelve las semanas Lun-Dom que tienen días en el mes dado
function getMonthWeeks(year: number, month: number): Array<{ start: Date; end: Date }> {
  const firstDay = new Date(year, month, 1);
  const lastDay = new Date(year, month + 1, 0);
  lastDay.setHours(23, 59, 59, 999);

  // Lunes anterior o igual al primer día del mes
  const dow = firstDay.getDay(); // 0=Dom, 1=Lun, ..., 6=Sab
  const offsetToMonday = dow === 0 ? -6 : 1 - dow;
  const firstMonday = new Date(firstDay);
  firstMonday.setDate(firstDay.getDate() + offsetToMonday);

  const weeks: Array<{ start: Date; end: Date }> = [];
  let cur = new Date(firstMonday);

  while (cur <= lastDay) {
    const weekEnd = new Date(cur);
    weekEnd.setDate(cur.getDate() + 6);
    weekEnd.setHours(23, 59, 59, 999);
    weeks.push({ start: new Date(cur), end: new Date(weekEnd) });
    cur = new Date(cur);
    cur.setDate(cur.getDate() + 7);
  }

  return weeks;
}

// Devuelve el índice (1-based) de la semana a la que pertenece la fecha, o -1
function getWeekIndex(date: Date, weeks: Array<{ start: Date; end: Date }>): number {
  for (let i = 0; i < weeks.length; i++) {
    if (date >= weeks[i].start && date <= weeks[i].end) return i + 1;
  }
  return -1;
}

interface Props {
  bets: Bet[];
}

export function History({ bets }: Props) {
  const now = new Date();
  const currentKey = yearMonthKey(now);

  const [selectedMonth, setSelectedMonth] = useState<string>(currentKey);
  const [selectedWeek, setSelectedWeek] = useState<number>(0); // 0 = mes completo
  const [selectedLeague, setSelectedLeague] = useState<string>("Todas");
  const [selectedSource, setSelectedSource] = useState<string>("Todas");
  const [selectedBookmaker, setSelectedBookmaker] = useState<string>("Todas");

  // Meses disponibles: los que tienen apuestas + el mes actual
  const availableMonths = useMemo(() => {
    const seen = new Set<string>([currentKey]);
    for (const b of bets) seen.add(yearMonthKey(new Date(b.timestamp)));
    return Array.from(seen).sort().reverse();
  }, [bets, currentKey]);

  const { year, month } = useMemo(() => {
    const [y, m] = selectedMonth.split("-").map(Number);
    return { year: y, month: m - 1 };
  }, [selectedMonth]);

  const weeks = useMemo(() => getMonthWeeks(year, month), [year, month]);
  const totalWeeks = weeks.length;

  const handleMonthChange = (key: string) => {
    setSelectedMonth(key);
    setSelectedWeek(0);
  };

  // Apuestas filtradas por todos los criterios
  const filtered = useMemo(() => {
    return bets.filter((b) => {
      const d = new Date(b.timestamp);
      if (yearMonthKey(d) !== selectedMonth) return false;
      if (selectedWeek !== 0 && getWeekIndex(d, weeks) !== selectedWeek) return false;
      if (selectedLeague !== "Todas" && b.league !== selectedLeague) return false;
      if (selectedSource !== "Todas" && b.source !== selectedSource) return false;
      if (selectedBookmaker !== "Todas" && b.bookmaker !== selectedBookmaker) return false;
      return true;
    });
  }, [bets, selectedMonth, selectedWeek, selectedLeague, selectedSource, selectedBookmaker, weeks]);

  const MODEL_LEAGUES = Object.keys(LEAGUE_LABELS);

  const sorted = useMemo(
    () => [...filtered].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()),
    [filtered]
  );

  // Estadísticas del periodo
  const periodStats = useMemo(() => {
    const resolved = filtered.filter((b) => b.result !== null);
    const pnl = resolved.reduce((s, b) => s + b.pnl, 0);
    const staked = resolved.reduce((s, b) => s + b.stake, 0);
    const wins = resolved.filter((b) => b.result === "win" || b.result === "half_win").length;
    const roi = staked > 0 ? (pnl / staked) * 100 : 0;
    const winRate = resolved.length > 0 ? (wins / resolved.length) * 100 : 0;
    return { pnl, roi, winRate, resolved: resolved.length, total: filtered.length };
  }, [filtered]);

  const monthLabel = `${MONTH_NAMES[month]} ${year}`;
  const periodLabel = selectedWeek === 0 ? monthLabel : `Semana ${selectedWeek} · ${monthLabel}`;

  return (
    <div className="bg-[#111827] border border-slate-800 rounded-xl p-5">
      <div className="flex flex-col gap-3 mb-4">

        {/* Título + selector de mes */}
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="text-[15px] font-bold font-[family-name:var(--font-display)]">
            Historial
          </div>
          <select
            value={selectedMonth}
            onChange={(e) => handleMonthChange(e.target.value)}
            className="bg-[#0f172a] border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-200 outline-none focus:border-emerald-600 transition cursor-pointer"
          >
            {availableMonths.map((key) => {
              const [y, m] = key.split("-").map(Number);
              return (
                <option key={key} value={key}>
                  {MONTH_NAMES[m - 1]} {y}
                </option>
              );
            })}
          </select>
        </div>

        {/* Selector de semana */}
        <div className="flex items-center gap-1 flex-wrap">
          {[0, ...Array.from({ length: totalWeeks }, (_, i) => i + 1)].map((w) => (
            <button
              key={w}
              onClick={() => setSelectedWeek(w)}
              className={`text-[11px] font-semibold px-2.5 py-1 rounded-full transition ${
                selectedWeek === w
                  ? "bg-violet-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200"
              }`}
            >
              {w === 0 ? "Mes completo" : `Semana ${w}`}
            </button>
          ))}
        </div>

        {/* Barra de stats del periodo */}
        <div className="grid grid-cols-4 gap-2 bg-[#0f172a] rounded-lg px-4 py-3 border border-slate-800">
          <div className="text-center">
            <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-0.5">P&L</div>
            <div
              className="text-sm font-bold font-[family-name:var(--font-display)]"
              style={{ color: periodStats.pnl >= 0 ? "#10b981" : "#ef4444" }}
            >
              {formatCurrency(periodStats.pnl)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-0.5">ROI</div>
            <div
              className="text-sm font-bold font-[family-name:var(--font-display)]"
              style={{ color: periodStats.roi >= 0 ? "#10b981" : "#ef4444" }}
            >
              {periodStats.roi.toFixed(1)}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-0.5">Win Rate</div>
            <div
              className="text-sm font-bold font-[family-name:var(--font-display)]"
              style={{ color: periodStats.winRate >= 50 ? "#10b981" : "#f59e0b" }}
            >
              {periodStats.winRate.toFixed(0)}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-0.5">Apuestas</div>
            <div className="text-sm font-bold text-slate-300 font-[family-name:var(--font-display)]">
              {periodStats.resolved}
              <span className="text-slate-600 text-[10px] font-normal">/{periodStats.total}</span>
            </div>
          </div>
        </div>

        {/* Filtro por liga */}
        <div className="flex items-center gap-1 flex-wrap">
          {["Todas", ...MODEL_LEAGUES].map((league) => (
            <button
              key={league}
              onClick={() => setSelectedLeague(league)}
              className={`text-[11px] font-semibold px-2.5 py-1 rounded-full transition ${
                selectedLeague === league
                  ? "bg-indigo-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200"
              }`}
            >
              {league === "Todas" ? "Todas" : LEAGUE_LABELS[league]}
            </button>
          ))}
        </div>

        {/* Filtro por fuente */}
        <div className="flex items-center gap-1 flex-wrap">
          {["Todas", ...SOURCES].map((source) => (
            <button
              key={source}
              onClick={() => setSelectedSource(source)}
              className={`text-[11px] font-semibold px-2.5 py-1 rounded-full transition ${
                selectedSource === source
                  ? "bg-emerald-700 text-white"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200"
              }`}
            >
              {source}
            </button>
          ))}
        </div>

        {/* Filtro por casa de apuestas */}
        <div className="flex items-center gap-1 flex-wrap">
          {["Todas", ...BOOKMAKERS].map((bm) => {
            const colors = BOOKMAKER_COLORS[bm];
            const isActive = selectedBookmaker === bm;
            return (
              <button
                key={bm}
                onClick={() => setSelectedBookmaker(bm)}
                className={`text-[11px] font-semibold px-2.5 py-1 rounded-full transition ${
                  isActive
                    ? colors
                      ? `${colors.bg} ${colors.text}`
                      : "bg-slate-600 text-white"
                    : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200"
                }`}
              >
                {bm}
              </button>
            );
          })}
        </div>
      </div>

      {/* Lista de apuestas */}
      {sorted.length === 0 ? (
        <div className="text-center py-10 text-slate-500 text-sm">
          Sin apuestas en {periodLabel}
        </div>
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
                      <div className="flex items-center justify-end gap-2 mt-0.5 flex-wrap">
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
                        {bet.bookmaker && (() => {
                          const c = BOOKMAKER_COLORS[bet.bookmaker];
                          return c ? (
                            <span className={`text-[10px] font-semibold px-2 py-0.5 rounded ${c.bg} ${c.text}`}>
                              {bet.bookmaker}
                            </span>
                          ) : (
                            <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                              {bet.bookmaker}
                            </span>
                          );
                        })()}
                      </div>
                    </>
                  ) : (
                    <div className="flex items-center gap-2 flex-wrap justify-end">
                      <span className="text-[10px] font-semibold text-amber-500 uppercase">
                        Pendiente
                      </span>
                      {bet.bookmaker && (() => {
                        const c = BOOKMAKER_COLORS[bet.bookmaker];
                        return c ? (
                          <span className={`text-[10px] font-semibold px-2 py-0.5 rounded ${c.bg} ${c.text}`}>
                            {bet.bookmaker}
                          </span>
                        ) : (
                          <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                            {bet.bookmaker}
                          </span>
                        );
                      })()}
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
