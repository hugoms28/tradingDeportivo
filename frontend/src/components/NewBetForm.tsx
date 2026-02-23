"use client";

import { useState, useMemo } from "react";
import type { BetFormData, DisciplineSettings, LockStatus } from "@/lib/types";
import { calcKellyStake, calcEdge } from "@/lib/calculations";
import { SOURCES, MARKETS, LEAGUES, SPORTS } from "@/lib/constants";

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
  });

  const kelly = useMemo(() => {
    const prob = parseFloat(form.modelProb) / 100;
    const odds = parseFloat(form.odds);
    if (!prob || !odds || prob <= 0 || prob >= 1 || odds <= 1) return null;
    const stake = calcKellyStake(bankroll, prob, odds, settings.kellyFraction, settings.maxStakePct);
    const edge = calcEdge(prob, odds);
    return { stake: Math.max(stake, 0), edge };
  }, [form.modelProb, form.odds, bankroll, settings]);

  const handleSubmit = async () => {
    const error = await onPlaceBet(form);
    if (error) {
      onNotify(error, "error");
      return;
    }
    const odds = parseFloat(form.odds);
    const stake = form.source === "Modelo" && kelly ? kelly.stake : parseFloat(form.stake);
    onNotify(`Apuesta registrada: ${stake.toFixed(2)}€ @ ${odds}`, "success");
    setForm({ event: "", source: "Modelo", tipsterName: "", sport: "", modelProb: "", odds: "", stake: "", market: "1X2", pick: "", league: "" });
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
          {lockStatus.locked ? "⛔ BLOQUEADO" : "REGISTRAR APUESTA"}
        </button>
      </div>
    </div>
  );
}
