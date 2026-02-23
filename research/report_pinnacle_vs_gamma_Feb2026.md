# Backtest Dixon-Coles gamma_i vs Pinnacle Closing Odds
## Temporada 2025-26 — Febrero 2026

**Fecha**: 2026-02-23  
**Modelo**: Dixon-Coles + xG, gamma_i por equipo, time decay 60 dias  
**Metodologia**: Split 70/30 temporal. Test set: oct 2025 - feb 2026.  
**Odds**: Pinnacle closing (PSCH/PSCD/PSCA) de football-data.co.uk  
**Kelly**: 0.25, max stake 5% bankroll, solo edge > 0 vs Pinnacle  
**Bankroll inicial por liga**: EUR 500

---

## Resultados por liga

| Liga | Matched | Match% | Apuestas | Win% | Edge medio | Odds medias | P&L | Max DD | ROI |
|------|--------:|-------:|---------:|-----:|-----------:|------------:|----:|-------:|----:|
| EPL | 142 | 74.0% | 192 | 32.8% | +8.54% | 4.07 | EUR -161.74 | EUR 314.65 | -32.4% |
| La_Liga | 135 | 72.6% | 161 | 25.5% | +10.59% | 4.46 | EUR -228.38 | EUR 349.37 | -45.7% |
| Bundesliga | 103 | 68.2% | 132 | 29.5% | +7.87% | 4.52 | EUR -105.78 | EUR 188.09 | -21.2% |
| Serie_A | 118 | 61.5% | 148 | 31.8% | +8.03% | 4.50 | EUR -69.25 | EUR 246.77 | -13.8% |
| Ligue_1 | 107 | 70.4% | 128 | 31.2% | +10.10% | 4.34 | EUR -84.76 | EUR 413.00 | -16.9% |
| **TOTAL** | 605 | — | 761 | 30.2% | +9.02% | 4.36 | **EUR -649.91** | EUR 413.00 | — |

---

## Distribucion de apuestas por outcome

| Outcome | N apuestas | Win% | Edge medio | P&L total |
|---------|----------:|-----:|-----------:|----------:|
| A | 441 | 27.4% | +11.35% | EUR -990.95 |
| D | 140 | 23.6% | +3.17% | EUR +294.63 |
| H | 180 | 42.2% | +7.87% | EUR +46.42 |

---

## Notas metodologicas

- **Odds usadas**: PSCH/PSCD/PSCA (Pinnacle closing). Si no disponibles, PSH/PSD/PSA (opening).
- **Match rate**: join por (fecha exacta, home_team, away_team). Partidos sin match se excluyen.
- **Kelly vs naive**: el P&L aqui es REAL (vs Pinnacle), no vs mercado naive.
  Un edge positivo vs Pinnacle es un edge real.
- **Limitacion**: la simulacion no descuenta comisiones ni considera liquidez de mercado.

*Detalle de apuestas en `research/pinnacle_bets.json`.*