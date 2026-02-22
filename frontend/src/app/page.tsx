"use client";

import { useState, useEffect } from "react";
import { useBettingStore } from "@/hooks/useBettingStore";
import { Dashboard } from "@/components/Dashboard";
import { NewBetForm } from "@/components/NewBetForm";
import { History } from "@/components/History";
import { Settings } from "@/components/Settings";
import { Predictions } from "@/components/Predictions";
import { ModelStatus } from "@/components/ModelStatus";

type View = "dashboard" | "training" | "predictions" | "newbet" | "history" | "settings";

const tabs: { id: View; label: string }[] = [
  { id: "dashboard", label: "Panel" },
  { id: "training", label: "Entrenamiento" },
  { id: "predictions", label: "Predicciones" },
  { id: "newbet", label: "+ Apuesta" },
  { id: "history", label: "Historial" },
  { id: "settings", label: "Config" },
];

interface Notification {
  msg: string;
  type: "success" | "error" | "info";
}

export default function Page() {
  const store = useBettingStore();
  const [view, setView] = useState<View>("dashboard");
  const [notification, setNotification] = useState<Notification | null>(null);
  const [pendingAlert, setPendingAlert] = useState(false);

  const notify = (msg: string, type: "success" | "error" | "info") => {
    setNotification({ msg, type });
    setTimeout(() => setNotification(null), 4000);
  };

  // Show pending bets alert once after hydration
  useEffect(() => {
    if (!store.hydrated) return;
    const pending = store.state.bets.filter((b) => b.result === null).length;
    if (pending > 0) setPendingAlert(true);
  }, [store.hydrated]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!store.hydrated) {
    return (
      <div className="min-h-screen bg-[#0a0e17] flex items-center justify-center">
        <div className="text-slate-500 text-sm">Cargando...</div>
      </div>
    );
  }

  const lightColor =
    store.trafficLight === "green"
      ? "#10b981"
      : store.trafficLight === "orange"
        ? "#f59e0b"
        : "#ef4444";

  return (
    <div className="min-h-screen bg-[#0a0e17] text-slate-200">
      {/* Header */}
      <header className="border-b border-slate-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div
            className="w-3 h-3 rounded-full animate-[pulse-dot_2s_infinite]"
            style={{ backgroundColor: lightColor }}
          />
          <h1 className="text-lg font-bold font-[family-name:var(--font-display)]">
            Trading Deportivo
          </h1>
        </div>
        <div className="flex items-center gap-3">
          {store.apiAvailable && (
            <span className="text-[10px] text-emerald-600 font-mono">API</span>
          )}
          <div className="text-xs text-slate-600">
            {store.state.bankroll.toFixed(2)}€
          </div>
        </div>
      </header>

      {/* Nav tabs */}
      <nav className="border-b border-slate-800 px-6 flex gap-1 overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setView(tab.id)}
            className={`px-4 py-3 text-xs font-semibold transition whitespace-nowrap ${
              view === tab.id
                ? "text-emerald-400 border-b-2 border-emerald-400"
                : "text-slate-500 hover:text-slate-300"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {/* Notification */}
      {notification && (
        <div
          className={`mx-6 mt-4 rounded-lg px-4 py-3 text-sm animate-[slide-in_0.2s_ease-out] ${
            notification.type === "success"
              ? "bg-emerald-950 text-emerald-300 border border-emerald-800"
              : notification.type === "error"
                ? "bg-red-950 text-red-300 border border-red-800"
                : "bg-blue-950 text-blue-300 border border-blue-800"
          }`}
        >
          {notification.msg}
        </div>
      )}

      {/* Pending bets alert */}
      {pendingAlert && (
        <div className="mx-6 mt-4 rounded-lg px-4 py-3 flex items-center justify-between bg-amber-950/60 border border-amber-700/50">
          <span className="text-sm text-amber-300">
            ⚠️ Tienes{" "}
            <strong>{store.state.bets.filter((b) => b.result === null).length}</strong>{" "}
            apuesta{store.state.bets.filter((b) => b.result === null).length !== 1 ? "s" : ""}{" "}
            pendiente{store.state.bets.filter((b) => b.result === null).length !== 1 ? "s" : ""} de resolver
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => { setView("history"); setPendingAlert(false); }}
              className="text-xs font-semibold text-amber-300 hover:text-amber-100 underline transition"
            >
              Ir al historial
            </button>
            <button
              onClick={() => setPendingAlert(false)}
              className="text-xs text-amber-600 hover:text-amber-400 transition ml-2"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Content */}
      <main className="p-6">
        {view === "dashboard" && (
          <Dashboard
            state={store.state}
            stats={store.stats}
            settings={store.settings}
            lockStatus={store.lockStatus}
            onResolveBet={store.resolveBet}
            onUnlock={store.unlock}
          />
        )}
        {view === "predictions" && (
          <Predictions
            onNotify={notify}
            bankroll={store.state.bankroll}
            kellyFraction={store.settings.kellyFraction}
            maxStakePct={store.settings.maxStakePct}
            onPlaceBet={store.placeBetFromPrediction}
            placedBets={store.state.bets}
          />
        )}
        {view === "newbet" && (
          <NewBetForm
            bankroll={store.state.bankroll}
            settings={store.settings}
            lockStatus={store.lockStatus}
            onPlaceBet={store.placeBet}
            onNotify={notify}
            onNavigate={(v) => setView(v as View)}
          />
        )}
        {view === "history" && (
          <History bets={store.state.bets} onResolveBet={store.resolveBet} />
        )}
        {view === "training" && <ModelStatus onNotify={notify} />}
        {view === "settings" && (
          <Settings
            settings={store.settings}
            lockStatus={store.lockStatus}
            onUpdate={store.updateSettings}
            onReset={store.reset}
            onUnlock={store.unlock}
          />
        )}
      </main>
    </div>
  );
}
