"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import type { Bet, BettingState, DisciplineSettings, BetFormData, Stats } from "@/lib/types";
import {
  DEFAULT_INITIAL_BANKROLL,
  DEFAULT_SETTINGS,
  STORAGE_KEY,
  SETTINGS_KEY,
} from "@/lib/constants";
import {
  calcKellyStake,
  calcEdge,
  computeStats,
  computeLockStatus,
  computeTrafficLight,
} from "@/lib/calculations";
import {
  getBets,
  createBet,
  resolveBet as apiResolveBet,
  getBetStats,
  getSettings,
  updateSettings as apiUpdateSettings,
  triggerAutoResolve,
  type ApiBet,
  type Prediction,
  type ValueBet,
} from "@/lib/api";

function getInitialState(): BettingState {
  return {
    bankroll: DEFAULT_INITIAL_BANKROLL,
    initialBankroll: DEFAULT_INITIAL_BANKROLL,
    peakBankroll: DEFAULT_INITIAL_BANKROLL,
    bets: [],
    lockedUntil: null,
    lockReason: null,
  };
}

function apiBetToBet(b: ApiBet): Bet {
  return {
    id: b.id,
    event: b.event,
    source: b.source as Bet["source"],
    tipsterName: b.tipsterName || "",
    market: b.market,
    pick: b.pick,
    odds: b.odds,
    modelProb: b.modelProb,
    stake: b.stake,
    edge: b.edge,
    result: b.result,
    pnl: b.pnl,
    closingOdds: b.closingOdds ?? null,
    clv: b.clv ?? null,
    matchStartsAt: b.matchStartsAt ?? null,
    bookmaker: b.bookmaker ?? null,
    timestamp: b.timestamp,
    resolvedAt: b.resolvedAt,
    league: b.league || undefined,
  };
}

export function useBettingStore() {
  const [state, setState] = useState<BettingState>(getInitialState);
  const [settings, setSettings] = useState<DisciplineSettings>(DEFAULT_SETTINGS);
  const [hydrated, setHydrated] = useState(false);
  const [apiAvailable, setApiAvailable] = useState(false);
  const syncedRef = useRef(false);

  // Try to load from API first, fallback to localStorage
  useEffect(() => {
    async function init() {
      try {
        const [apiBets, apiSettings, apiStats] = await Promise.all([
          getBets(),
          getSettings(),
          getBetStats(),
        ]);

        // API is available — use it as source of truth
        setApiAvailable(true);

        // Auto-resolver apuestas pendientes al iniciar (silencioso)
        try {
          const resolved = await triggerAutoResolve();
          if (resolved.resolved > 0) {
            // Recargar datos si se resolvieron apuestas
            const [freshBets, freshStats] = await Promise.all([getBets(), getBetStats()]);
            apiBets.splice(0, apiBets.length, ...freshBets);
            Object.assign(apiStats, freshStats);
          }
        } catch {
          // Silencioso: Understat puede no estar disponible
        }

        const bets = apiBets.map(apiBetToBet);
        const bankroll = apiStats.bankroll;
        const initialBankroll = apiStats.initialBankroll;
        const peakBankroll = apiStats.peakBankroll;

        setState({
          bankroll,
          initialBankroll,
          peakBankroll,
          bets,
          lockedUntil: null,
          lockReason: null,
        });

        if (apiSettings.discipline_settings) {
          setSettings((prev) => ({
            ...prev,
            ...apiSettings.discipline_settings,
          }));
        }
      } catch {
        // API not available — fallback to localStorage
        console.log("API not available, using localStorage");
        try {
          const saved = localStorage.getItem(STORAGE_KEY);
          if (saved) setState(JSON.parse(saved));
          const savedSettings = localStorage.getItem(SETTINGS_KEY);
          if (savedSettings) setSettings(JSON.parse(savedSettings));
        } catch {
          // ignore parse errors
        }
      }
      setHydrated(true);
    }
    init();
  }, []);

  // Persist to localStorage as backup (always)
  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }, [state, hydrated]);

  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  }, [settings, hydrated]);

  const stats = useMemo(() => computeStats(state), [state]);
  const lockStatus = useMemo(
    () => computeLockStatus(state, stats, settings),
    [state, stats, settings],
  );
  const trafficLight = useMemo(
    () => computeTrafficLight(lockStatus, stats, settings),
    [lockStatus, stats, settings],
  );

  const refreshFromApi = useCallback(async () => {
    if (!apiAvailable) return;
    try {
      const [apiBets, apiStats] = await Promise.all([getBets(), getBetStats()]);
      setState((prev) => ({
        ...prev,
        bets: apiBets.map(apiBetToBet),
        bankroll: apiStats.bankroll,
        initialBankroll: apiStats.initialBankroll,
        peakBankroll: apiStats.peakBankroll,
      }));
    } catch {
      // Silent fail on refresh
    }
  }, [apiAvailable]);

  const placeBet = useCallback(
    async (form: BetFormData): Promise<string | null> => {
      if (lockStatus.locked) return `Sistema BLOQUEADO: ${lockStatus.reason}`;
      if (!form.event || !form.odds || !form.pick) return "Completa todos los campos obligatorios";

      const odds = parseFloat(form.odds);
      let stake: number;

      if (form.source === "Modelo") {
        const prob = parseFloat(form.modelProb) / 100;
        if (!prob || prob <= 0 || prob >= 1) return "Probabilidad inválida";
        const edge = calcEdge(prob, odds);
        if (edge <= 0) return "Edge negativo — Kelly dice NO apostar";
        stake = Math.round(
          calcKellyStake(state.bankroll, prob, odds, settings.kellyFraction, settings.maxStakePct) * 100,
        ) / 100;
      } else {
        stake = parseFloat(form.stake);
      }

      if (!stake || stake <= 0) return "Stake inválido";
      if (stake > state.bankroll) return "Stake superior al bankroll";
      if (stake > state.bankroll * settings.maxStakePct) {
        return `Stake excede máximo (${(settings.maxStakePct * 100).toFixed(0)}% = ${(state.bankroll * settings.maxStakePct).toFixed(2)}€)`;
      }

      const prob = form.source === "Modelo" ? parseFloat(form.modelProb) / 100 : null;
      const edge = prob ? calcEdge(prob, odds) : null;

      if (apiAvailable) {
        // Create bet via API
        try {
          await createBet({
            event: form.event,
            league: form.league || "",
            source: form.source,
            tipster_name: form.tipsterName || "",
            sport: form.sport || null,
            market: form.market,
            pick: form.pick,
            odds,
            model_prob: prob ?? null,
            stake,
            edge,
            bookmaker: form.bookmaker || null,
          });
          await refreshFromApi();
          return null;
        } catch (e) {
          return `Error API: ${e}`;
        }
      } else {
        // Fallback: local state
        const newBet: Bet = {
          id: Date.now(),
          event: form.event,
          source: form.source,
          tipsterName: form.tipsterName,
          sport: form.sport || null,
          market: form.market,
          pick: form.pick,
          odds,
          modelProb: prob ?? null,
          stake,
          edge,
          result: null,
          pnl: 0,
          closingOdds: null,
          clv: null,
          matchStartsAt: null,
          bookmaker: null,
          timestamp: new Date().toISOString(),
          resolvedAt: null,
          league: form.league || undefined,
        };
        setState((prev) => ({ ...prev, bets: [...prev.bets, newBet] }));
        return null;
      }
    },
    [state.bankroll, settings, lockStatus, apiAvailable, refreshFromApi],
  );

  const placeBetFromPrediction = useCallback(
    async (prediction: Prediction, valueBet: ValueBet, bookmaker = ""): Promise<string | null> => {
      if (lockStatus.locked) return `Sistema BLOQUEADO: ${lockStatus.reason}`;
      if (valueBet.prob == null || valueBet.odds == null) return "Odds o probabilidad no disponibles";

      const stake = Math.round(
        calcKellyStake(
          state.bankroll,
          valueBet.prob,
          valueBet.odds,
          settings.kellyFraction,
          settings.maxStakePct,
        ) * 100,
      ) / 100;

      if (stake <= 0) return "Stake calculado = 0 (Kelly negativo o edge insuficiente)";
      if (stake > state.bankroll) return "Stake superior al bankroll disponible";

      if (!apiAvailable) return "API no disponible";

      try {
        await createBet({
          event: `${prediction.home_team} vs ${prediction.away_team}`,
          league: prediction.league,
          source: "Modelo",
          market: valueBet.market,
          pick: valueBet.label,
          odds: valueBet.odds,
          model_prob: valueBet.prob,
          stake,
          edge: valueBet.edge,
          prediction_id: prediction.id,
          match_starts_at: prediction.starts_at ?? null,
          bookmaker: bookmaker || null,
        });
        await refreshFromApi();
        return null;
      } catch (e) {
        return `Error al registrar apuesta: ${e}`;
      }
    },
    [state.bankroll, settings, lockStatus, apiAvailable, refreshFromApi],
  );

  const resolveBet = useCallback(
    async (id: number, result: "win" | "half_win" | "loss" | "half_loss" | "void") => {
      if (apiAvailable) {
        try {
          await apiResolveBet(id, result);
          await refreshFromApi();
        } catch {
          // Fallback to local
          _resolveLocal(id, result, setState);
        }
      } else {
        _resolveLocal(id, result, setState);
      }
    },
    [apiAvailable, refreshFromApi],
  );

  const unlock = useCallback(() => {
    setState((prev) => ({ ...prev, lockedUntil: null, lockReason: null }));
  }, []);

  const reset = useCallback(async () => {
    setState(getInitialState());
    if (apiAvailable) {
      try {
        await apiUpdateSettings({
          bankroll: DEFAULT_INITIAL_BANKROLL,
          initial_bankroll: DEFAULT_INITIAL_BANKROLL,
          peak_bankroll: DEFAULT_INITIAL_BANKROLL,
        });
      } catch {
        // ignore
      }
    }
  }, [apiAvailable]);

  const updateSettings = useCallback(
    async (patch: Partial<DisciplineSettings>) => {
      setSettings((prev) => ({ ...prev, ...patch }));
      if (apiAvailable) {
        try {
          await apiUpdateSettings({ discipline_settings: patch as Record<string, number> });
        } catch {
          // ignore
        }
      }
    },
    [apiAvailable],
  );

  return {
    state,
    settings,
    stats,
    lockStatus,
    trafficLight,
    hydrated,
    apiAvailable,
    placeBet,
    placeBetFromPrediction,
    resolveBet,
    unlock,
    reset,
    updateSettings,
    refreshFromApi,
  };
}

function _resolveLocal(
  id: number,
  result: "win" | "half_win" | "loss" | "half_loss" | "void",
  setState: React.Dispatch<React.SetStateAction<BettingState>>,
) {
  setState((prev) => {
    const bets = prev.bets.map((b) => {
      if (b.id !== id) return b;
      const pnl =
        result === "win"
          ? b.stake * (b.odds - 1)
          : result === "half_win"
            ? (b.stake / 2) * (b.odds - 1)
            : result === "loss"
              ? -b.stake
              : result === "half_loss"
                ? -(b.stake / 2)
                : 0;
      return { ...b, result, pnl, resolvedAt: new Date().toISOString() };
    });
    const totalPnL = bets
      .filter((b) => b.result !== null)
      .reduce((s, b) => s + b.pnl, 0);
    const newBankroll = prev.initialBankroll + totalPnL;
    const newPeak = Math.max(prev.peakBankroll, newBankroll);
    return { ...prev, bets, bankroll: newBankroll, peakBankroll: newPeak };
  });
}
