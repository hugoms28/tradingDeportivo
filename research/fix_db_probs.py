# -*- coding: utf-8 -*-
"""
research/fix_db_probs.py — Corrige model_prob almacenada en escala 0-100 -> 0-1.
Ejecutar UNA SOLA VEZ desde la raiz del proyecto:
    python research/fix_db_probs.py
"""
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "backend", "trading.db")

print(f"DB: {DB_PATH}")
conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

# -- Estado previo ----------------------------------------------------------
cur.execute("SELECT COUNT(*) FROM bets WHERE model_prob IS NOT NULL")
total = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM bets WHERE model_prob > 1.0")
to_fix = cur.fetchone()[0]

print(f"\nApuestas con model_prob no nulo : {total}")
print(f"Apuestas con model_prob > 1.0   : {to_fix}  (a corregir)")

if to_fix > 0:
    cur.execute("SELECT id, event, model_prob FROM bets WHERE model_prob > 1.0 ORDER BY id")
    rows = cur.fetchall()
    print("\nMuestra (max 10):")
    for r in rows[:10]:
        print(f"  id={r[0]:>4}  {r[1]:<40}  model_prob={r[2]:.1f}  -> {r[2]/100:.4f}")

    cur.execute("""
        UPDATE bets
        SET model_prob = model_prob / 100.0
        WHERE model_prob > 1.0
    """)
    conn.commit()
    print(f"\nFilas actualizadas: {cur.rowcount}")

    # -- Verificacion post-fix -----------------------------------------------
    cur.execute("SELECT COUNT(*) FROM bets WHERE model_prob > 1.0")
    remaining = cur.fetchone()[0]
    print(f"Filas con model_prob > 1.0 tras fix: {remaining}")
    if remaining == 0:
        print("OK: todas las probabilidades estan en escala 0-1.")
    else:
        print("ATENCION: todavia hay filas con model_prob > 1.0, revisar manualmente.")
else:
    print("\nNo hay nada que corregir — todas las probabilidades ya estan en 0-1.")

conn.close()
