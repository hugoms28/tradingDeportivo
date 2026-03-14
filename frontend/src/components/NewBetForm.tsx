"use client";

import { useState, useMemo } from "react";
import type { BetFormData, DisciplineSettings, LockStatus } from "@/lib/types";
import { calcKellyStake, calcEdge } from "@/lib/calculations";
import { SOURCES, MARKETS, LEAGUES, SPORTS, BOOKMAKERS, BOOKMAKER_COLORS } from "@/lib/constants";
import { searchPs3838Events, getPs3838Lines, type Ps3838Lines, type Ps3838EventResult } from "@/lib/api";

interface Props {
  bankroll: number;
  settings: DisciplineSettings;
  lockStatus: LockStatus;
  onPlaceBet: (form: BetFormData) => Promise<string | null> | string | null;
  onNotify: (msg: string, type: "success" | "error" | "info") => void;
  onNavigate: (view: string) => void;
}

const inputCls =
  "bg-[#0f172a] border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 w-full outline-none focus:border-emerald-600 transition";

export function NewBetForm({ bankroll, settings, lockStatus, onPlaceBet, onNotify, onNavigate }: Props) {
  const [form, setForm] = useState<BetFormData>({
    event: "",
    source: "Modelo",
    tipsterName: "",
    sport: "",
    modelProb: "",
    odds: "",
    stake: "",
    market: "1X2",
    pick: "",
    league: "",
    bookmaker: "",
    ps3838EventId: null,
  });

  const [ps3838Query, setPs3838Query] = useState("");
  const [ps3838Results, setPs3838Results] = useState<Ps3838EventResult[]>([]);
  const [ps3838Searching, setPs3838Searching] = useState(false);
  const [ps3838SelectedEvent, setPs3838SelectedEvent] = useState<Ps3838EventResult | null>(null);
  const [ps3838Lines, setPs3838Lines] = useState<Ps3838Lines | null>(null);
  const [ps3838LoadingLines, setPs3838LoadingLines] = useState(false);

  const kelly = useMemo(() => {
    const prob = parseFloat(form.modelProb) / 100;
    const odds = parseFloat(form.odds);
    if (!prob || !odds || prob <= 0 || prob >= 1 || odds <= 1) return null;
    const stake = calcKellyStake(bankroll, prob, odds, settings.kellyFraction, settings.maxStakePct);
    const edge = calcEdge(prob, odds);
    return { stake: Math.max(stake, 0), edge };
  }, [form.modelProb, form.odds, bankroll, settings]);

  const handlePs3838Search = async () => {
    const q = ps3838Query;
    if (q.length < 2) return;
    setPs3838Searching(true);
    try {
      const result = await searchPs3838Events(q);
      if ("error" in result) {
        setPs3838Results([]);
      } else {
        setPs3838Results(result);
      }
    } catch {
      setPs3838Results([]);
    } finally {
      setPs3838Searching(false);
    }
  };

  const handlePs3838SelectEvent = async (ev: Ps3838EventResult) => {
    setPs3838SelectedEvent(ev);
    setPs3838Results([]);
    setPs3838Query(`${ev.home} vs ${ev.away}`);
    setPs3838Lines(null);
    // Auto-fill event name
    set({ event: `${ev.home} vs ${ev.away}`, ps3838EventId: ev.event_id, ps3838SportId: ev.sport_id, ps3838LeagueId: ev.league_id });
    // Fetch lines
    setPs3838LoadingLines(true);
    try {
      const lines = await getPs3838Lines(ev.event_id, ev.sport_id);
      if ("error" in lines) {
        onNotify(`Error obteniendo líneas: ${lines.error}`, "error");
      } else {
        setPs3838Lines(lines);
      }
    } catch (e) {
      onNotify(`Error obteniendo líneas: ${e}`, "error");
    } finally {
      setPs3838LoadingLines(false);
    }
  };

  const handleSelectPs3838Line = (market: string, pick: string, odds: number) => {
    set({ market, pick, odds: String(odds) });
  };

  const handleSubmit = async () => {
    const error = await onPlaceBet(form);
    if (error) {
      onNotify(error, "error");
      return;
    }
    const odds = parseFloat(form.odds);
    const stake = form.source === "Modelo" && kelly ? kelly.stake : parseFloat(form.stake);
    const placed = form.bookmaker === "PS3838" && form.ps3838EventId
      ? `Colocada en PS3838: ${stake.toFixed(2)}€ @ ${odds}`
      : `Apuesta registrada: ${stake.toFixed(2)}€ @ ${odds}`;
    onNotify(placed, "success");
    setPs3838Lines(null);
    setPs3838Results([]);
    setPs3838Query("");
    setPs3838SelectedEvent(null);
    setForm({ event: "", source: "Modelo", tipsterName: "", sport: "", modelProb: "", odds: "", stake: "", market: "1X2", pick: "", league: "", bookmaker: "", ps3838EventId: null });
    onNavigate("dashboard");
  };

  const set = (patch: Partial<BetFormData>) => setForm((p) => ({ ...p, ...patch }));

  return (
    <div className="bg-[#111827] border border-slate-800 rounded-xl p-6 max-w-[500px]">
      <div className="text-[15px] font-bold mb-6 font-[family-name:var(--font-display)]">
        Nueva Apuesta
      </div>

      {lockStatus.locked && (
        <div className="bg-red-950 rounded-lg px-4 py-3 mb-5 text-xs text-red-300">
          ⛔ {lockStatus.reason} — No puedes apostar ahora.
        </div>
      )}

      <div className="flex flex-col gap-4">
        {/* Event */}
        <div>
          <label className="text-[11px] text-slate-500 uppercase tracking-widest mb-1.5 block font-semibold">
            Evento *
          </label>
          <input
            className={inputCls}
            placeholder="Real Madrid vs Barcelona"
            value={form.event}
            onChange={(e) => set({ event: e.target.value })}
          />
        </div>

        {/* Source + Market */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-[11px] text-slate-500 uppercase tracking-widest mb-1.5 block font-semibold">Fuente</label>
            <select className={inputCls} value={form.source} onChange={(e) => set({ source: e.target.value as BetFormData["source"] })}>
              {SOURCES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div>
            <label className="text-[11px] text-slate-500 uppercase tracking-widest mb-1.5 block font-semibold">Mercado</label>
            <select className={inputCls} value={form.market} onChange={(e) => set({ market: e.target.value })}>
              {MARKETS.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
        </div>

        {/* League */}
        <div>
          <label className="text-[11px] text-slate-500 uppercase tracking-widest mb-1.5 block font-semibold">Liga</label>
          <select className={inputCls} value={form.league} onChange={(e) => set({ league: e.target.value })}>
            <option value="">-- Seleccionar --</option>
            {LEAGUES.map((l) => <option key={l} value={l}>{l}</option>)}
          </select>
        </div>

        {/* Tipster name */}
        {form.source === "Tipster" && (
          <div>
            <label className="text-[11px] text-slate-500 uppercase tracking-widest mb-1.5 block font-semibold">
              Nombre del Tipster
            </label>
            <input className={inputCls} placeholder="Nombre" value={form.tipsterName} onChange={(e) => set({ tipsterName: e.target.value })} />
          </div>
        )}

        {/* Sport selector (Tipster / Propia) */}
        {(form.source === "Tipster" || form.source === "Propia") && (
          <div>
            <label className="text-[11px] text-slate-500 uppercase tracking-widest mb-1.5 block font-semibold">
              Deporte
            </label>
            <select className={inputCls} value={form.sport} onChange={(e) => set({ sport: e.target.value })}>
              <option value="">-- Seleccionar --</option>
              {SPORTS.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        )}

        {/* Pick */}
        <div>
          <label className="text-[11px] text-slate-500 uppercase tracking-widest mb-1.5 block font-semibold">Pick *</label>
          <input
            className={inputCls}
            placeholder="Over 2.5 / Home / AH -1.5..."
            value={form.pick}
            onChange={(e) => set({ pick: e.target.value })}
          />
        </div>

        {/* Odds + Model Prob */}
        <div className={`grid gap-3 ${form.source === "Modelo" ? "grid-cols-2" : "grid-cols-1"}`}>
          <div>
            <label className="text-[11px] text-slate-500 uppercase tracking-widest mb-1.5 block font-semibold">Odds *</label>
            <input
              className={inputCls}
              type="number"
              step="0.01"
              placeholder="2.10"
              value={form.odds}
              onChange={(e) => set({ odds: e.target.value })}
            />
          </div>
          {form.source === "Modelo" && (
            <div>
              <label className="text-[11px] text-slate-500 uppercase tracking-widest mb-1.5 block font-semibold">
                Prob. Modelo (%)
              </label>
              <input
                className={inputCls}
                type="number"
                step="0.1"
                placeholder="55.0"
                value={form.modelProb}
                onChange={(e) => set({ modelProb: e.target.value })}
              />
            </div>
          )}
        </div>

        {/* Kelly output */}
        {form.source === "Modelo" && kelly && (
          <div className={`rounded-lg px-4 py-3.5 ${kelly.edge > 0 ? "bg-emerald-950" : "bg-red-950"}`}>
            <div className={`text-[10px] uppercase tracking-widest mb-2 ${kelly.edge > 0 ? "text-emerald-300" : "text-red-300"}`}>
              Kelly Criterion (¼ Kelly)
            </div>
            <div className="flex justify-between items-center">
              <div>
                <span
                  className="text-[22px] font-bold font-[family-name:var(--font-display)]"
                  style={{ color: kelly.edge > 0 ? "#10b981" : "#ef4444" }}
                >
                  {kelly.stake.toFixed(2)}€
                </span>
                <span className="text-xs text-slate-500 ml-2">
                  ({(kelly.stake / bankroll * 100).toFixed(1)}% del bankroll)
                </span>
              </div>
              <div className="text-right">
                <div className="text-xs text-slate-500">Edge</div>
                <div
                  className="text-base font-semibold"
                  style={{ color: kelly.edge > 0 ? "#10b981" : "#ef4444" }}
                >
                  {kelly.edge.toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Manual stake for non-Model */}
        {form.source !== "Modelo" && (
          <div>
            <label className="text-[11px] text-slate-500 uppercase tracking-widest mb-1.5 block font-semibold">Stake (€) *</label>
            <input
              className={inputCls}
              type="number"
              step="0.01"
              placeholder="20.00"
              value={form.stake}
              onChange={(e) => set({ stake: e.target.value })}
            />
            <div className="text-[10px] text-slate-500 mt-1">
              Máximo: {(bankroll * settings.maxStakePct).toFixed(2)}€ ({(settings.maxStakePct * 100).toFixed(0)}% del bankroll)
            </div>
          </div>
        )}

        {/* Bookmaker */}
        <div>
          <label className="text-[11px] text-slate-500 uppercase tracking-widest mb-1.5 block font-semibold">
            Casa de Apuestas
          </label>
          <div className="flex gap-2 flex-wrap">
            {BOOKMAKERS.map((bm) => {
              const c = BOOKMAKER_COLORS[bm];
              const isActive = form.bookmaker === bm;
              return (
                <button
                  key={bm}
                  type="button"
                  onClick={() => set({ bookmaker: isActive ? "" : bm })}
                  className={`text-[12px] font-semibold px-3 py-1.5 rounded-lg transition border ${
                    isActive
                      ? `${c.activeBg} ${c.text} border-transparent`
                      : `bg-transparent ${c.text} border-current opacity-60 hover:opacity-100`
                  }`}
                >
                  {bm}
                </button>
              );
            })}
          </div>
        </div>

        {/* PS3838: búsqueda libre de evento + selector de línea */}
        {form.bookmaker === "PS3838" && (
          <div className="border border-violet-800/50 rounded-xl p-4 space-y-3 bg-violet-950/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-[11px] text-violet-300 font-semibold uppercase tracking-widest">
                  Colocar en PS3838
                </span>
                {ps3838Lines?.is_live && (
                  <span className="text-[9px] font-bold text-red-400 bg-red-950 px-1.5 py-0.5 rounded uppercase tracking-wider">LIVE</span>
                )}
              </div>
              {form.ps3838EventId && (
                <span className="text-[10px] text-emerald-400">✓ Vinculado · ID {form.ps3838EventId}</span>
              )}
            </div>

            {/* Campo de búsqueda libre */}
            <div className="relative">
              <label className="text-[10px] text-slate-500 uppercase tracking-widest mb-1 block">
                Buscar equipo (mínimo 2 letras)
              </label>
              <div className="flex gap-2">
                <input
                  className={inputCls}
                  placeholder="Ej: Zhan, Arsenal, Lakers... (Enter para buscar)"
                  value={ps3838Query}
                  onChange={(e) => { setPs3838Query(e.target.value); setPs3838Results([]); }}
                  onKeyDown={(e) => e.key === "Enter" && handlePs3838Search()}
                />
                <button
                  type="button"
                  onClick={handlePs3838Search}
                  disabled={ps3838Searching || ps3838Query.length < 2}
                  className="px-3 py-2 rounded-lg bg-violet-700 hover:bg-violet-600 disabled:opacity-40 text-white text-xs font-semibold transition whitespace-nowrap"
                >
                  {ps3838Searching ? "..." : "Buscar"}
                </button>
              </div>

              {/* Resultados de búsqueda */}
              {ps3838Results.length > 0 && (
                <div className="mt-1 border border-slate-700 rounded-lg overflow-hidden bg-slate-900 max-h-48 overflow-y-auto">
                  {ps3838Results.map((ev) => (
                    <button
                      key={ev.event_id}
                      type="button"
                      onClick={() => handlePs3838SelectEvent(ev)}
                      className="w-full text-left px-3 py-2 hover:bg-violet-900/40 transition border-b border-slate-800 last:border-0"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold text-slate-200">
                          {ev.home} vs {ev.away}
                        </span>
                        {ev.status === "I" && (
                          <span className="text-[9px] font-bold text-red-400 bg-red-950 px-1.5 py-0.5 rounded uppercase tracking-wider">LIVE</span>
                        )}
                      </div>
                      <div className="text-[10px] text-slate-500">
                        {ev.sport_name ? `${ev.sport_name} · ` : ""}{ev.league} · {ev.starts ? new Date(ev.starts).toLocaleString("es-ES", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }) : ""}
                      </div>
                    </button>
                  ))}
                </div>
              )}

              {!ps3838Searching && ps3838Results.length === 0 && ps3838SelectedEvent === null && ps3838Query.length >= 2 && (
                <div className="mt-1 text-[11px] text-slate-600 px-1">Sin resultados — pulsa Buscar o Enter</div>
              )}
            </div>

            {/* Stake */}
            <div>
              <label className="text-[10px] text-slate-500 uppercase tracking-widest mb-1 block">
                Stake (€) *
              </label>
              <input
                className={inputCls}
                type="number"
                min="0.01"
                step="0.5"
                placeholder={
                  form.source === "Modelo" && kelly
                    ? `Kelly: ${kelly.stake.toFixed(2)}`
                    : "Importe a apostar"
                }
                value={form.stake}
                onChange={(e) => set({ stake: e.target.value })}
              />
              {form.stake && parseFloat(form.stake) < 20 && (
                <div className="text-[10px] text-amber-400 mt-1">
                  Mínimo por API ~20€ — stakes inferiores pueden ser rechazados
                </div>
              )}
            </div>

            {/* Líneas del evento seleccionado */}
            {ps3838LoadingLines && (
              <div className="text-xs text-slate-500">Cargando líneas...</div>
            )}

            {ps3838Lines && !ps3838LoadingLines && !ps3838Lines.line_id && (
              <div className="text-[11px] text-amber-400 bg-amber-950/40 rounded px-3 py-2">
                Sin líneas disponibles para este evento. Rellena manualmente mercado, pick y odds arriba.
              </div>
            )}

            {ps3838Lines && !ps3838LoadingLines && ps3838Lines.line_id && (
              <div className="space-y-2">
                <div className="text-[10px] text-slate-500 uppercase tracking-widest">
                  Selecciona línea → auto-rellena mercado, pick y odds:
                </div>

                {/* 1X2 / Moneyline */}
                {ps3838Lines.moneyline && (
                  <div className="space-y-1">
                    <div className="text-[10px] text-slate-400 font-semibold">1X2 / Moneyline</div>
                    <div className="flex gap-1.5">
                      {[
                        { label: "1 (Local)", odds: ps3838Lines.moneyline.home },
                        ...(ps3838Lines.moneyline.draw ? [{ label: "X (Empate)", odds: ps3838Lines.moneyline.draw }] : []),
                        { label: "2 (Visitante)", odds: ps3838Lines.moneyline.away },
                      ].map(({ label, odds }) => (
                        <button
                          key={label}
                          type="button"
                          onClick={() => handleSelectPs3838Line("1X2", label, odds)}
                          className={`flex-1 py-1.5 rounded text-[10px] font-semibold transition border ${
                            form.pick === label
                              ? "bg-violet-700 text-white border-transparent"
                              : "bg-slate-800 text-slate-300 border-slate-700 hover:border-violet-600"
                          }`}
                        >
                          {label}<br />{odds.toFixed(2)}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Asian Handicap / Spread */}
                {ps3838Lines.spreads.length > 0 && (
                  <div className="space-y-1">
                    <div className="text-[10px] text-slate-400 font-semibold">Handicap / Spread</div>
                    <div className="flex flex-col gap-1 max-h-36 overflow-y-auto">
                      {ps3838Lines.spreads.map((s, i) => (
                        <div key={s.altLineId ?? `spread-${i}`} className="flex gap-1.5">
                          <button
                            type="button"
                            onClick={() => handleSelectPs3838Line("Asian Handicap", `AH Home ${s.hdp}`, s.home)}
                            className={`flex-1 py-1 rounded text-[10px] font-semibold transition border ${
                              form.pick === `AH Home ${s.hdp}`
                                ? "bg-violet-700 text-white border-transparent"
                                : "bg-slate-800 text-slate-300 border-slate-700 hover:border-violet-600"
                            }`}
                          >
                            Home {s.hdp > 0 ? `+${s.hdp}` : s.hdp} · {s.home.toFixed(2)}
                          </button>
                          <button
                            type="button"
                            onClick={() => handleSelectPs3838Line("Asian Handicap", `AH Away ${-s.hdp}`, s.away)}
                            className={`flex-1 py-1 rounded text-[10px] font-semibold transition border ${
                              form.pick === `AH Away ${-s.hdp}`
                                ? "bg-violet-700 text-white border-transparent"
                                : "bg-slate-800 text-slate-300 border-slate-700 hover:border-violet-600"
                            }`}
                          >
                            Away {-s.hdp > 0 ? `+${-s.hdp}` : -s.hdp} · {s.away.toFixed(2)}
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Over/Under / Totals */}
                {ps3838Lines.totals.length > 0 && (
                  <div className="space-y-1">
                    <div className="text-[10px] text-slate-400 font-semibold">Over / Under · Totales</div>
                    <div className="flex flex-col gap-1 max-h-36 overflow-y-auto">
                      {ps3838Lines.totals.map((t, i) => (
                        <div key={t.altLineId ?? `total-${i}`} className="flex gap-1.5">
                          <button
                            type="button"
                            onClick={() => handleSelectPs3838Line("Over/Under", `Over ${t.points}`, t.over)}
                            className={`flex-1 py-1 rounded text-[10px] font-semibold transition border ${
                              form.pick === `Over ${t.points}`
                                ? "bg-violet-700 text-white border-transparent"
                                : "bg-slate-800 text-slate-300 border-slate-700 hover:border-violet-600"
                            }`}
                          >
                            Over {t.points} · {t.over.toFixed(2)}
                          </button>
                          <button
                            type="button"
                            onClick={() => handleSelectPs3838Line("Over/Under", `Under ${t.points}`, t.under)}
                            className={`flex-1 py-1 rounded text-[10px] font-semibold transition border ${
                              form.pick === `Under ${t.points}`
                                ? "bg-violet-700 text-white border-transparent"
                                : "bg-slate-800 text-slate-300 border-slate-700 hover:border-violet-600"
                            }`}
                          >
                            Under {t.points} · {t.under.toFixed(2)}
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Submit */}
        <button
          onClick={handleSubmit}
          disabled={lockStatus.locked}
          className={`rounded-lg px-6 py-3 text-sm font-bold mt-2 transition ${
            lockStatus.locked
              ? "bg-slate-800 text-slate-600 cursor-not-allowed"
              : "bg-emerald-500 text-slate-900 hover:bg-emerald-400 cursor-pointer"
          }`}
        >
          {lockStatus.locked
            ? "⛔ BLOQUEADO"
            : form.bookmaker === "PS3838" && form.ps3838EventId
              ? "COLOCAR EN PS3838"
              : "REGISTRAR APUESTA"}
        </button>
      </div>
    </div>
  );
}
