# Proyecto Trading Deportivo - Arquitectura Completa

## Contexto
Modelo Dixon-Coles con xG de Understat para 5 ligas (EPL, La_Liga, Bundesliga, Serie_A, Ligue_1). Paquete Python `trading-deportivo` ya funcional. Frontend Next.js con sistema de disciplina. Objetivo: web completa para ejecutar modelo, ver predicciones, registrar apuestas y trackear resultados.

## Decisiones
- **DB**: SQLite (simple, un archivo, suficiente para 1 usuario)
- **Backend**: FastAPI (wraps trading-deportivo package)
- **Frontend**: Next.js 16 + TypeScript + Tailwind
- **Ejecución modelo**: Botón manual + cron automático (viernes 08:00)
- **Deploy**: Local ahora, VPS (Hetzner/DO ~5€/mes) después del forward test

## Arquitectura

```
trading-deportivo/
├── trading_deportivo/          # Paquete Python (ya existe, no se toca)
├── backend/                    # FastAPI (NUEVO)
│   ├── main.py                 # App FastAPI + CORS
│   ├── database.py             # SQLite con SQLAlchemy/aiosqlite
│   ├── models_db.py            # Tablas: predictions, bets, model_runs, settings
│   ├── routers/
│   │   ├── predictions.py      # GET /predictions, POST /predict (ejecuta modelo)
│   │   ├── bets.py             # CRUD apuestas + resolve
│   │   ├── models.py           # GET /models, POST /train
│   │   └── settings.py         # GET/PUT settings de disciplina
│   ├── services/
│   │   ├── runner.py           # Ejecuta modelo para 1 o 5 ligas (background task)
│   │   └── scheduler.py        # APScheduler: cron jueves/viernes
│   └── requirements.txt
├── frontend/                   # Next.js (ya existe, se adapta)
│   └── src/
│       ├── lib/api.ts          # Cliente HTTP para el backend (NUEVO)
│       ├── hooks/useBettingStore.ts  # Migrar de localStorage a API calls
│       └── components/
│           ├── Predictions.tsx  # Vista de predicciones por liga (NUEVO)
│           ├── ModelStatus.tsx  # Estado del modelo, botón ejecutar (NUEVO)
│           └── ... (existentes)
└── docker-compose.yml          # Para deploy futuro
```

## Base de datos SQLite

### predictions
| Campo | Tipo |
|-------|------|
| id | INTEGER PK |
| league | TEXT |
| run_id | TEXT (agrupa por ejecución) |
| home_team | TEXT |
| away_team | TEXT |
| p_home | REAL |
| p_draw | REAL |
| p_away | REAL |
| lambda_home | REAL |
| mu_away | REAL |
| best_ou_line | REAL |
| best_ou_prob | REAL |
| best_ah_line | REAL |
| best_ah_prob | REAL |
| odds_home | REAL (nullable, de PS3838) |
| odds_draw | REAL |
| odds_away | REAL |
| edge_home | REAL |
| edge_draw | REAL |
| edge_away | REAL |
| recommended_bet | TEXT (nullable) |
| kelly_stake | REAL (nullable) |
| created_at | DATETIME |

### bets
| Campo | Tipo |
|-------|------|
| id | INTEGER PK |
| prediction_id | INTEGER FK (nullable) |
| event | TEXT |
| league | TEXT |
| source | TEXT (Modelo/Tipster/Propia) |
| tipster_name | TEXT |
| market | TEXT |
| pick | TEXT |
| odds | REAL |
| model_prob | REAL (nullable) |
| stake | REAL |
| edge | REAL (nullable) |
| result | TEXT (nullable: win/loss/void) |
| pnl | REAL default 0 |
| created_at | DATETIME |
| resolved_at | DATETIME (nullable) |

### model_runs
| Campo | Tipo |
|-------|------|
| id | TEXT PK (uuid) |
| league | TEXT |
| status | TEXT (running/completed/failed) |
| started_at | DATETIME |
| completed_at | DATETIME (nullable) |
| n_matches | INTEGER |
| mse | REAL |
| converged | BOOLEAN |
| error | TEXT (nullable) |

### settings
| Campo | Tipo |
|-------|------|
| key | TEXT PK |
| value | TEXT (JSON serialized) |

Almacena: bankroll, initial_bankroll, peak_bankroll, discipline_settings como pares key-value JSON.

## API Endpoints

### Predicciones
- `POST /api/predict` — `{leagues: ["EPL", ...]}` → ejecuta en background, retorna run_id
- `GET /api/predict/{run_id}/status` — estado de la ejecución
- `GET /api/predictions?league=EPL` — últimas predicciones por liga
- `GET /api/predictions/latest` — más recientes de todas las ligas

### Apuestas
- `GET /api/bets` — todas (filtros: league, status, date)
- `POST /api/bets` — registrar nueva apuesta
- `PATCH /api/bets/{id}/resolve` — `{result: "win"|"loss"|"void"}` → calcula PnL
- `GET /api/bets/stats` — ROI, win rate, drawdown, etc.

### Modelos
- `GET /api/models` — listar modelos guardados por liga
- `POST /api/train` — entrenar modelo (background task)
- `GET /api/models/{league}/latest` — info del último modelo

### Settings
- `GET /api/settings` — discipline settings + bankroll
- `PUT /api/settings` — actualizar settings

### Scheduler
- `GET /api/scheduler/status` — estado del cron
- `POST /api/scheduler/toggle` — activar/desactivar

## Orden de implementación

1. **Backend esqueleto** — FastAPI + SQLite + tablas
2. **Router predictions** — POST /predict + GET /predictions + runner.py
3. **Router bets** — CRUD + resolve + stats
4. **Router settings** — GET/PUT
5. **Frontend api.ts** — Cliente HTTP
6. **Frontend Predictions.tsx** — Vista predicciones
7. **Migrar useBettingStore** — De localStorage a API
8. **Scheduler** — APScheduler con cron
9. **Frontend ModelStatus.tsx** — Vista modelo
10. **Docker compose** — Para deploy futuro

## Dependencias backend
```
fastapi
uvicorn[standard]
sqlalchemy[asyncio]
aiosqlite
apscheduler
pydantic
```
