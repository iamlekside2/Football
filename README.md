# Football Value Model

A monorepo for researching whether a Dixon-Coles football model can find
**value bets** against bookmaker closing odds — and an honest dashboard to
prove (or disprove) that edge before any money or scraping infrastructure is
built.

> **Current status:** across 6 EPL seasons the model shows **no 1X2 edge**
> (pooled ROI ≈ **−6.6%**, and the bookmaker's closing line is better
> calibrated on both Brier and log-loss). The dashboard exists to keep this
> honest and to support the model work needed to *find* an edge. Live-betting
> infrastructure is intentionally **not** built yet.

## Layout

```
football-value-model/
├── backend/                 # Python: model + backtest + FastAPI
│   ├── poisson_model.py     # Dixon-Coles model + find_value()
│   ├── backtest.py          # walk-forward backtest (no lookahead) + CLI
│   ├── diagnostics.py       # multi-season robustness + calibration + CLI
│   ├── main.py              # FastAPI API wrapping the above
│   └── requirements.txt
└── frontend/                # React + Vite + TypeScript dashboard
    └── src/
        ├── api.ts           # typed API client
        ├── App.tsx
        └── components/      # BacktestView, CalibrationView (recharts)
```

## Methodology (the guardrails)

- **No lookahead.** For every fixture predicted, the model is trained only on
  matches that kicked off strictly earlier (`Date < match_date`).
- **Honest benchmark.** Edge = `model_prob − (1 / closing_odds)`; profit is
  settled at those same closing odds (Bet365 closing, auto-detected).
- **No threshold tuning on the reported season.** Tuning, if done, must use a
  separate held-out season (`backtest.py --tune-csv/--tune-url`).
- **Calibration test.** The model's probabilities are scored against the
  bookmaker's no-vig closing line (Brier / log-loss). If the line wins, there
  is no edge to extract — full stop.

## Running

### Backend (Python 3.10+)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

CLI (no server needed):

```bash
python backtest.py                       # EPL 2023/24, default settings
python backtest.py --season ...          # see --help
python diagnostics.py                    # 6-season robustness + calibration
```

### Frontend

```bash
cd frontend
npm install
npm run dev                              # http://localhost:5173
```

The Vite dev server proxies `/api/*` to the backend on port 8000.

## API

| Method | Path               | Purpose                                              |
|--------|--------------------|------------------------------------------------------|
| GET    | `/api/health`      | liveness                                             |
| GET    | `/api/seasons`     | available seasons                                    |
| POST   | `/api/backtest`    | walk-forward backtest for one season + baselines     |
| POST   | `/api/calibration` | pooled model-vs-bookmaker calibration                |
| GET    | `/api/predict`     | model 1X2/OU probabilities for a fixture (inspector) |
| GET    | `/api/teams`       | teams in a season                                    |

Data source: [football-data.co.uk](https://www.football-data.co.uk/).
