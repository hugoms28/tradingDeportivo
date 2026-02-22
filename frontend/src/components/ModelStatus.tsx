"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  type ModelInfo,
  type ModelRunInfo,
  type TrainingProgress,
  type TrainingLeagueProgress,
  type SchedulerStatus,
  getLatestModel,
  getModelRuns,
  startTraining,
  getTrainingProgress,
  getSchedulerStatus,
  toggleScheduler,
} from "@/lib/api";

const LEAGUES = ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"];

interface Props {
  onNotify: (msg: string, type: "success" | "error" | "info") => void;
}

export function ModelStatus({ onNotify }: Props) {
  const [models, setModels] = useState<Record<string, ModelInfo>>({});
  const [runs, setRuns] = useState<ModelRunInfo[]>([]);
  const [scheduler, setScheduler] = useState<SchedulerStatus | null>(null);
  const [loading, setLoading] = useState(true);

  // Training state
  const [trainId, setTrainId] = useState<string | null>(null);
  const [progress, setProgress] = useState<TrainingProgress | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [runsData, schedulerData] = await Promise.all([
        getModelRuns(),
        getSchedulerStatus(),
      ]);
      setRuns(runsData);
      setScheduler(schedulerData);

      const modelPromises = LEAGUES.map(async (l) => {
        try {
          const m = await getLatestModel(l);
          return [l, m] as [string, ModelInfo];
        } catch {
          return [l, null] as [string, null];
        }
      });
      const results = await Promise.all(modelPromises);
      const modelsMap: Record<string, ModelInfo> = {};
      for (const [league, model] of results) {
        if (model && !("error" in model)) {
          modelsMap[league] = model;
        }
      }
      setModels(modelsMap);
    } catch (e) {
      onNotify(`Error cargando datos: ${e}`, "error");
    } finally {
      setLoading(false);
    }
  }, [onNotify]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Cleanup poll on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const handleTrain = async (leagues: string[]) => {
    try {
      const res = await startTraining(leagues);
      setTrainId(res.train_id);
      onNotify(`Entrenamiento iniciado: ${leagues.join(", ")}`, "info");

      // Start polling
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const p = await getTrainingProgress(res.train_id);
          setProgress(p);
          if (p.status === "completed" || p.status === "failed") {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            onNotify(
              p.status === "completed"
                ? "Entrenamiento completado"
                : "Entrenamiento con errores",
              p.status === "completed" ? "success" : "error",
            );
            fetchData();
          }
        } catch {
          // keep polling
        }
      }, 1500);

      // Timeout 15 min
      setTimeout(() => {
        if (pollRef.current) {
          clearInterval(pollRef.current);
          pollRef.current = null;
          setTrainId(null);
          setProgress(null);
        }
      }, 900000);
    } catch (e) {
      onNotify(`Error: ${e}`, "error");
    }
  };

  const handleToggleScheduler = async () => {
    try {
      const res = await toggleScheduler();
      setScheduler((prev) =>
        prev ? { ...prev, enabled: res.enabled } : null,
      );
      onNotify(res.message, "info");
    } catch (e) {
      onNotify(`Error: ${e}`, "error");
    }
  };

  const isTraining = trainId !== null && progress !== null && progress.status === "running";

  if (loading) {
    return <div className="text-center text-slate-500 text-sm py-8">Cargando...</div>;
  }

  return (
    <div className="space-y-6">
      <h2 className="text-base font-bold">Entrenamiento</h2>

      {/* ── Training progress ── */}
      {progress && trainId && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold">
              {isTraining ? "Entrenando..." : progress.status === "completed" ? "Completado" : "Error"}
            </span>
            {!isTraining && (
              <button
                onClick={() => { setTrainId(null); setProgress(null); }}
                className="text-[10px] text-slate-500 hover:text-slate-300"
              >
                Cerrar
              </button>
            )}
          </div>

          {Object.values(progress.leagues).map((lp) => (
            <TrainingProgressBar key={lp.league} progress={lp} />
          ))}
        </div>
      )}

      {/* ── Model cards per league ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {LEAGUES.map((league) => {
          const model = models[league];
          return (
            <div
              key={league}
              className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 space-y-2"
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-sm">{league.replace("_", " ")}</span>
                <button
                  onClick={() => handleTrain([league])}
                  disabled={isTraining}
                  className="text-[10px] px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-50 transition"
                >
                  Reentrenar
                </button>
              </div>
              {model ? (
                <div className="text-xs text-slate-400 space-y-1">
                  <div>Equipos: {model.n_teams}</div>
                  <div>
                    Gamma: {model.gamma?.toFixed(4)} &middot; Rho:{" "}
                    {model.rho?.toFixed(4)}
                  </div>
                  {model.mse != null && <div>MSE: {model.mse.toFixed(6)}</div>}
                  <div>
                    Convergencia:{" "}
                    <span className={model.converged ? "text-emerald-400" : "text-red-400"}>
                      {model.converged ? "OK" : "No"}
                    </span>
                  </div>
                  {model.saved_at && (
                    <div className="text-slate-600">
                      {new Date(model.saved_at).toLocaleDateString()}
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-xs text-slate-600">Sin modelo — entrena primero</div>
              )}
            </div>
          );
        })}
      </div>

      {/* Train all */}
      <button
        onClick={() => handleTrain(LEAGUES)}
        disabled={isTraining}
        className="px-4 py-2 text-xs font-semibold rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 transition"
      >
        {isTraining ? "Entrenando..." : "Reentrenar todas las ligas"}
      </button>

      {/* ── Scheduler ── */}
      {scheduler && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-bold text-sm">Cron Automático</span>
            <button
              onClick={handleToggleScheduler}
              className={`text-[10px] px-3 py-1 rounded font-semibold transition ${
                scheduler.enabled
                  ? "bg-emerald-800 text-emerald-200 hover:bg-emerald-700"
                  : "bg-slate-700 text-slate-300 hover:bg-slate-600"
              }`}
            >
              {scheduler.enabled ? "Activado" : "Desactivado"}
            </button>
          </div>
          <div className="text-xs text-slate-400">
            <div>
              {scheduler.config.day_of_week} a las{" "}
              {String(scheduler.config.hour).padStart(2, "0")}:
              {String(scheduler.config.minute).padStart(2, "0")}
            </div>
            {scheduler.next_run && (
              <div>
                Próxima: {new Date(scheduler.next_run).toLocaleString()}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Run history ── */}
      {runs.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-slate-400">Historial</h3>
          <div className="space-y-1">
            {runs.slice(0, 20).map((run) => (
              <div
                key={run.id}
                className="flex items-center gap-3 text-xs py-1.5 px-2 rounded bg-slate-900/30"
              >
                <span
                  className={`w-2 h-2 rounded-full flex-shrink-0 ${
                    run.status === "completed"
                      ? "bg-emerald-500"
                      : run.status === "failed"
                        ? "bg-red-500"
                        : "bg-amber-500 animate-pulse"
                  }`}
                />
                <span className="font-medium w-20">{run.league}</span>
                <span className="text-slate-500">
                  {run.started_at
                    ? new Date(run.started_at).toLocaleString()
                    : "—"}
                </span>
                {run.n_matches != null && (
                  <span className="text-slate-600">{run.n_matches} partidos</span>
                )}
                {run.error && (
                  <span className="text-red-400 truncate max-w-xs">{run.error}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Progress bar component ────────────────────────────────────────────────

function TrainingProgressBar({ progress: p }: { progress: TrainingLeagueProgress }) {
  const isRunning = p.status === "running";
  const isDone = p.status === "completed";
  const isFailed = p.status === "failed";

  const barColor = isFailed
    ? "bg-red-500"
    : isDone
      ? "bg-emerald-500"
      : "bg-emerald-500";

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium">{p.league.replace("_", " ")}</span>
        <span className={`${isFailed ? "text-red-400" : "text-slate-400"}`}>
          {isFailed ? p.error : p.step_label}
        </span>
      </div>
      <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${barColor} ${
            isRunning ? "animate-pulse" : ""
          }`}
          style={{ width: `${p.pct}%` }}
        />
      </div>
      <div className="text-[10px] text-slate-600 text-right">{p.pct}%</div>
    </div>
  );
}
