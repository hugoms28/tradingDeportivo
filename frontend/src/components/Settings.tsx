"use client";

import type { DisciplineSettings, LockStatus } from "@/lib/types";

interface Props {
  settings: DisciplineSettings;
  lockStatus: LockStatus;
  onUpdate: (patch: Partial<DisciplineSettings>) => void;
  onReset: () => void;
  onUnlock: () => void;
}

const inputCls =
  "bg-[#0f172a] border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 w-full outline-none focus:border-emerald-600 transition";

const fields: {
  key: keyof DisciplineSettings;
  label: string;
  step: number;
  mult?: number;
}[] = [
  { key: "dailyStopLoss", label: "Stop-Loss Diario (€)", step: 10 },
  { key: "weeklyStopLoss", label: "Stop-Loss Semanal (€)", step: 10 },
  { key: "maxDrawdownPct", label: "Max Drawdown (%)", step: 5, mult: 100 },
  { key: "maxDailyBets", label: "Max Apuestas/Día", step: 1 },
  { key: "cooldownLosses", label: "Cooldown tras X pérdidas", step: 1 },
  { key: "cooldownHours", label: "Horas de Cooldown", step: 0.5 },
  { key: "kellyFraction", label: "Fracción Kelly", step: 0.05 },
  { key: "maxStakePct", label: "Max Stake (% bankroll)", step: 1, mult: 100 },
];

export function Settings({ settings, lockStatus, onUpdate, onReset, onUnlock }: Props) {
  return (
    <div className="flex flex-col gap-5">
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-6">
        <div className="text-[15px] font-bold mb-5 font-[family-name:var(--font-display)]">
          Parámetros de Control
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {fields.map(({ key, label, step, mult }) => (
            <div key={key}>
              <label className="text-[11px] text-slate-500 uppercase tracking-widest mb-1.5 block font-semibold">
                {label}
              </label>
              <input
                className={inputCls}
                type="number"
                step={step}
                value={mult ? (settings[key] * mult).toFixed(0) : settings[key]}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (!isNaN(v)) onUpdate({ [key]: mult ? v / mult : v });
                }}
              />
            </div>
          ))}
        </div>
      </div>

      <div className="bg-[#111827] border border-slate-800 rounded-xl p-6">
        <div className="text-[15px] font-bold mb-4 font-[family-name:var(--font-display)]">Acciones</div>
        <div className="flex gap-3 flex-wrap">
          <button
            onClick={() => {
              if (window.confirm("¿Resetear todo el sistema? Se perderán todos los datos.")) {
                onReset();
              }
            }}
            className="border border-red-700 text-red-400 rounded-lg px-4 py-2.5 text-xs hover:bg-red-950 transition"
          >
            Resetear Sistema
          </button>
          {lockStatus.locked && (
            <button
              onClick={onUnlock}
              className="border border-amber-600 text-amber-400 rounded-lg px-4 py-2.5 text-xs hover:bg-amber-950 transition"
            >
              Desbloquear Manual
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
