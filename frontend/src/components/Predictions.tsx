"use client";

import { useState, useEffect, useCallback } from "react";
import {
  type Prediction,
  type ValueBet,
  getLatestPredictions,
  startPredictions,
  getPredictionStatus,
} from "@/lib/api";
import type { Bet } from "@/lib/types";
import { calcKellyStake } from "@/lib/calculations";

const LEAGUES = ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"];

interface Props {
  onNotify: (msg: string, type: "success" | "error" | "info") => void;
  bankroll: number;
  kellyFraction: number;
  maxStakePct: number;
  onPlaceBet: (prediction: Prediction, bet: ValueBet) => Promise<string | null>;
  placedBets: Bet[];
}

export function Predictions({ onNotify, bankroll, kellyFraction, maxStakePct, onPlaceBet, placedBets }: Props) {
  const [league, setLeague] = useState("EPL");
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [runStatus, setRunStatus] = useState<string | null>(null);

  const fetchPredictions = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getLatestPredictions();
      setPredictions(data[league] ?? []);
    } catch (e) {
      onNotify(`Error cargando predicciones: ${e}`, "error");
    } finally {
      setLoading(false);
    }
  }, [league, onNotify]);

  useEffect(() => {
    fetchPredictions();
  }, [fetchPredictions]);

  const handleRunModel = async (leagues: string[]) => {
    setRunning(true);
    setRunStatus(`Ejecutando ${leagues.join(", ")}...`);
    try {
      const res = await startPredictions(leagues);
      onNotify(`Predicciones iniciadas (${res.run_id})`, "info");

      const poll = setInterval(async () => {
        try {
          const status = await getPredictionStatus(res.run_id);
          const s = status as Record<string, unknown>;
          if (s.status === "completed" || s.status === "partial") {
            clearInterval(poll);
            setRunning(false);
            setRunStatus(null);
            onNotify(
              s.status === "completed"
                ? "Predicciones completadas"
                : "Algunas ligas fallaron",
              s.status === "completed" ? "success" : "error",
            );
            fetchPredictions();
          }
        } catch {
          // Keep polling
        }
      }, 3000);

      setTimeout(() => {
        clearInterval(poll);
        setRunning(false);
        setRunStatus(null);
      }, 300000);
    } catch (e) {
      setRunning(false);
      setRunStatus(null);
      onNotify(`Error: ${e}`, "error");
    }
  };

  // Separate matches with value bets vs no value
  const withValue = predictions.filter((p) => p.value_bets && p.value_bets.length > 0);
  const withoutValue = predictions.filter((p) => !p.value_bets || p.value_bets.length === 0);

  return (
    <div className="space-y-4">
      {/* Header + Actions */}
      <div className="flex items-center justify-between">
        <h2 className="text-base font-bold">Predicciones</h2>
        <div className="flex gap-2">
          <button
            onClick={() => handleRunModel([league])}
            disabled={running}
            className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            {running ? runStatus : `Generar ${league}`}
          </button>
          <button
            onClick={() => handleRunModel(LEAGUES)}
            disabled={running}
            className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            Todas las ligas
          </button>
        </div>
      </div>

      {/* League tabs */}
      <div className="flex gap-1 border-b border-slate-800">
        {LEAGUES.map((l) => (
          <button
            key={l}
            onClick={() => setLeague(l)}
            className={`px-3 py-2 text-xs font-semibold transition ${
              league === l
                ? "text-emerald-400 border-b-2 border-emerald-400"
                : "text-slate-500 hover:text-slate-300"
            }`}
          >
            {l.replace("_", " ")}
          </button>
        ))}
      </div>

      {/* Loading */}
      {loading && (
        <div className="text-center text-slate-500 text-sm py-8">Cargando...</div>
      )}

      {/* Empty state */}
      {!loading && predictions.length === 0 && (
        <div className="text-center text-slate-600 text-sm py-8">
          No hay predicciones para {league}. Pulsa &quot;Generar&quot; para ejecutar el modelo.
        </div>
      )}

      {/* ── Value Bets ── */}
      {!loading && withValue.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-emerald-400">
            Value Bets ({withValue.length})
          </h3>
          {withValue.map((p) => (
            <MatchCard
              key={p.id}
              prediction={p}
              bankroll={bankroll}
              kellyFraction={kellyFraction}
              maxStakePct={maxStakePct}
              onPlaceBet={onPlaceBet}
              onNotify={onNotify}
              placedBets={placedBets}
            />
          ))}
        </div>
      )}

      {/* ── Rest of matches ── */}
      {!loading && withoutValue.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-slate-500 mt-6">
            Sin value ({withoutValue.length})
          </h3>
          {withoutValue.map((p) => (
            <MatchCardCompact key={p.id} prediction={p} />
          ))}
        </div>
      )}

      {/* Run info */}
      {predictions.length > 0 && (
        <div className="text-xs text-slate-600 mt-4">
          {predictions.length} partidos &middot; Run: {predictions[0]?.run_id}{" "}
          &middot;{" "}
          {predictions[0]?.created_at
            ? new Date(predictions[0].created_at).toLocaleString()
            : ""}
        </div>
      )}
    </div>
  );
}

// ─── Match Card (with value bets) ──────────────────────────────────────────

interface MatchCardProps {
  prediction: Prediction;
  bankroll: number;
  kellyFraction: number;
  maxStakePct: number;
  onPlaceBet: (prediction: Prediction, bet: ValueBet) => Promise<string | null>;
  onNotify: (msg: string, type: "success" | "error" | "info") => void;
  placedBets: Bet[];
}

function MatchCard({ prediction: p, bankroll, kellyFraction, maxStakePct, onPlaceBet, onNotify, placedBets }: MatchCardProps) {
  const principals = p.value_bets.filter((v) => v.type === "principal");
  const tips = p.value_bets.filter((v) => v.type === "consejo");

  return (
    <div className="rounded-xl border border-emerald-800/50 bg-slate-900/70 p-4 space-y-3">
      {/* Match header */}
      <div className="flex items-center justify-between">
        <div className="font-semibold text-sm">
          {p.home_team} vs {p.away_team}
        </div>
        <div className="text-xs text-slate-500">
          xG: {p.lambda_home?.toFixed(1)} - {p.mu_away?.toFixed(1)}
        </div>
      </div>

      {/* Probabilities row */}
      <div className="flex gap-4 text-xs text-slate-400">
        <span>P(1): {pct(p.p_home)}</span>
        <span>P(X): {pct(p.p_draw)}</span>
        <span>P(2): {pct(p.p_away)}</span>
      </div>

      {/* Principal bets */}
      {principals.length > 0 && (
        <div className="space-y-1.5">
          {principals.map((v, i) => (
            <ValueBetRow
              key={i}
              bet={v}
              prediction={p}
              bankroll={bankroll}
              kellyFraction={kellyFraction}
              maxStakePct={maxStakePct}
              onPlaceBet={onPlaceBet}
              onNotify={onNotify}
              placedBets={placedBets}
            />
          ))}
        </div>
      )}

      {/* Tips */}
      {tips.length > 0 && (
        <div className="border-t border-slate-800 pt-2 space-y-1">
          <div className="text-[10px] text-slate-600 uppercase tracking-widest">
            Consejos
          </div>
          {tips.map((v, i) => (
            <ValueBetRow
              key={i}
              bet={v}
              isTip
              prediction={p}
              bankroll={bankroll}
              kellyFraction={kellyFraction}
              maxStakePct={maxStakePct}
              onPlaceBet={onPlaceBet}
              onNotify={onNotify}
              placedBets={placedBets}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Value bet row ─────────────────────────────────────────────────────────

interface ValueBetRowProps {
  bet: ValueBet;
  isTip?: boolean;
  prediction: Prediction;
  bankroll: number;
  kellyFraction: number;
  maxStakePct: number;
  onPlaceBet: (prediction: Prediction, bet: ValueBet) => Promise<string | null>;
  onNotify: (msg: string, type: "success" | "error" | "info") => void;
  placedBets: Bet[];
}

function ValueBetRow({
  bet,
  isTip = false,
  prediction,
  bankroll,
  kellyFraction,
  maxStakePct,
  onPlaceBet,
  onNotify,
  placedBets,
}: ValueBetRowProps) {
  const [placing, setPlacing] = useState(false);

  // Check against the store's bets so the state persists across tab switches
  const event = `${prediction.home_team} vs ${prediction.away_team}`;
  const alreadyPlaced = placedBets.some(
    (b) => b.event === event && b.market === bet.market && b.pick === bet.label,
  );

  const edgeStr = `${(bet.edge * 100).toFixed(1)}%`;
  const probStr = bet.prob != null ? `${(bet.prob * 100).toFixed(1)}%` : "—";
  const oddsStr = bet.odds != null ? bet.odds.toFixed(2) : "—";

  const stake =
    bet.prob != null && bet.odds != null
      ? Math.round(
          calcKellyStake(bankroll, bet.prob, bet.odds, kellyFraction, maxStakePct) * 100,
        ) / 100
      : 0;

  const handleBet = async () => {
    if (alreadyPlaced) return;
    setPlacing(true);
    const err = await onPlaceBet(prediction, bet);
    setPlacing(false);
    if (err) {
      onNotify(err, "error");
    } else {
      onNotify(
        `Apuesta registrada: ${bet.label} @ ${oddsStr} · ${stake.toFixed(2)}€`,
        "success",
      );
    }
  };

  return (
    <div
      className={`rounded-lg px-3 py-2 text-xs ${
        isTip
          ? "bg-slate-800/50 text-slate-400"
          : "bg-emerald-950/60 text-emerald-200"
      }`}
    >
      {/* Top row: market + label + edge */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
              isTip ? "bg-slate-700 text-slate-300" : "bg-emerald-800 text-emerald-100"
            }`}
          >
            {bet.market}
          </span>
          <span className="font-semibold">{bet.label}</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-slate-500">Prob: {probStr}</span>
          <span className="text-slate-500">Odds: {oddsStr}</span>
          <span className={`font-bold ${isTip ? "text-amber-400" : "text-emerald-400"}`}>
            +{edgeStr}
          </span>
        </div>
      </div>

      {/* Bottom row: stake + bet button */}
      <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-800/60">
        <span className="text-slate-500">
          Stake recomendado:{" "}
          <span className={`font-bold ${isTip ? "text-slate-300" : "text-emerald-300"}`}>
            {stake > 0 ? `${stake.toFixed(2)}€` : "—"}
          </span>
          <span className="text-slate-600 ml-1">
            ({(kellyFraction * 100).toFixed(0)}% Kelly)
          </span>
        </span>
        <button
          onClick={handleBet}
          disabled={placing || stake <= 0 || alreadyPlaced}
          className={`px-3 py-1 rounded-lg text-[11px] font-bold transition cursor-not-allowed ${
            alreadyPlaced
              ? "bg-slate-700 text-slate-400 opacity-60"
              : placing
                ? "bg-slate-600 text-slate-300 opacity-70"
                : isTip
                  ? "bg-slate-700 hover:bg-slate-600 text-slate-200 cursor-pointer"
                  : "bg-emerald-700 hover:bg-emerald-600 text-white cursor-pointer"
          }`}
        >
          {alreadyPlaced ? "✓ Apuesta hecha" : placing ? "Registrando..." : "Hacer apuesta"}
        </button>
      </div>
    </div>
  );
}

// ─── Compact card (no value bets) ──────────────────────────────────────────

function MatchCardCompact({ prediction: p }: { prediction: Prediction }) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-slate-800/50 bg-slate-900/30 px-4 py-2.5 text-xs">
      <span className="text-slate-400">
        {p.home_team} vs {p.away_team}
      </span>
      <div className="flex gap-3 text-slate-600">
        <span>{pct(p.p_home)}</span>
        <span>{pct(p.p_draw)}</span>
        <span>{pct(p.p_away)}</span>
      </div>
    </div>
  );
}

// ─── Helpers ───────────────────────────────────────────────────────────────

function pct(v: number | null) {
  return v != null ? `${(v * 100).toFixed(1)}%` : "—";
}
