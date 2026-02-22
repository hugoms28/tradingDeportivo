"use client";

interface Props {
  label: string;
  value: string;
  color?: string;
  subtitle?: string;
}

export function StatCard({ label, value, color = "#e2e8f0", subtitle }: Props) {
  return (
    <div className="bg-[#111827] border border-slate-800 rounded-xl p-5">
      <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-2">{label}</div>
      <div className="text-[22px] font-bold font-[family-name:var(--font-display)]" style={{ color }}>
        {value}
      </div>
      {subtitle && (
        <div className="text-[10px] text-slate-600 mt-1">{subtitle}</div>
      )}
    </div>
  );
}
