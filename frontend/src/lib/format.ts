export function formatCurrency(v: number): string {
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}\u20AC`;
}

export function formatDate(d: string): string {
  return new Date(d).toLocaleDateString("es-ES", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatPct(v: number, decimals = 1): string {
  return `${v.toFixed(decimals)}%`;
}
