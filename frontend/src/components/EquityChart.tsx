"use client";

import { useMemo } from "react";
import type { Bet } from "@/lib/types";
import { equityCurve } from "@/lib/calculations";

interface Props {
  bets: Bet[];
  initialBankroll: number;
}

export function EquityChart({ bets, initialBankroll }: Props) {
  const curve = useMemo(() => equityCurve(bets, initialBankroll), [bets, initialBankroll]);

  if (curve.length < 2) {
    return (
      <div className="h-[200px] flex items-center justify-center text-slate-500 text-sm">
        Sin datos suficientes para gr&aacute;fico
      </div>
    );
  }

  const width = 600;
  const height = 200;
  const pad = { top: 20, right: 20, bottom: 30, left: 55 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;

  const min = Math.min(...curve) * 0.98;
  const max = Math.max(...curve) * 1.02;
  const range = max - min || 1;

  const points = curve.map((v, i) => ({
    x: pad.left + (i / (curve.length - 1)) * chartW,
    y: pad.top + chartH - ((v - min) / range) * chartH,
    v,
  }));

  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
  const areaPath =
    linePath +
    ` L ${points[points.length - 1].x} ${pad.top + chartH} L ${points[0].x} ${pad.top + chartH} Z`;

  const current = curve[curve.length - 1];
  const isPositive = current >= initialBankroll;
  const color = isPositive ? "#10b981" : "#ef4444";

  const baselineY = pad.top + chartH - ((initialBankroll - min) / range) * chartH;

  const yTicks = 5;
  const yLabels = Array.from({ length: yTicks }, (_, i) => {
    const val = min + (range * i) / (yTicks - 1);
    return { val, y: pad.top + chartH - ((val - min) / range) * chartH };
  });

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} className="block">
      <defs>
        <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.25" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      {yLabels.map((l, i) => (
        <g key={i}>
          <line x1={pad.left} y1={l.y} x2={width - pad.right} y2={l.y} stroke="#1e293b" strokeWidth="1" />
          <text x={pad.left - 8} y={l.y + 4} textAnchor="end" fill="#64748b" fontSize="10" fontFamily="monospace">
            {l.val.toFixed(0)}&euro;
          </text>
        </g>
      ))}
      <line
        x1={pad.left} y1={baselineY} x2={width - pad.right} y2={baselineY}
        stroke="#f59e0b" strokeWidth="1" strokeDasharray="4 3" opacity="0.5"
      />
      <text x={width - pad.right + 4} y={baselineY + 3} fill="#f59e0b" fontSize="9" fontFamily="monospace" opacity="0.7">
        inicio
      </text>
      <path d={areaPath} fill="url(#areaGrad)" />
      <path d={linePath} fill="none" stroke={color} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
      <circle
        cx={points[points.length - 1].x}
        cy={points[points.length - 1].y}
        r="4" fill={color} stroke="#0f172a" strokeWidth="2"
      />
    </svg>
  );
}
