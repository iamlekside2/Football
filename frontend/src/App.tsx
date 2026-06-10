import { useEffect, useState } from 'react'
import './App.css'
import { api } from './api'
import type { BacktestResult, CalibrationResult, Season } from './api'
import BacktestView from './components/BacktestView'
import CalibrationView from './components/CalibrationView'
import PredictionsView from './components/PredictionsView'
import ModelQualityView from './components/ModelQualityView'
import DailyView from './components/DailyView'
import WorldCupView from './components/WorldCupView'

type Tab = 'worldcup' | 'today' | 'predictions' | 'quality' | 'backtest' | 'calibration'

const NAV: { key: Tab; label: string; group: 'site' | 'research' }[] = [
  { key: 'worldcup', label: '🏆 World Cup', group: 'site' },
  { key: 'today', label: 'Today', group: 'site' },
  { key: 'predictions', label: 'Head-to-Head', group: 'site' },
  { key: 'quality', label: 'Model Quality', group: 'site' },
  { key: 'backtest', label: 'Backtest', group: 'research' },
  { key: 'calibration', label: 'Calibration', group: 'research' },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('worldcup')
  const [seasons, setSeasons] = useState<Season[]>([])

  // Research-tab state
  const [seasonCode, setSeasonCode] = useState('2324')
  const [edge, setEdge] = useState(0.05)
  const [minTrain, setMinTrain] = useState(80)
  const [decay, setDecay] = useState(0)
  const [bt, setBt] = useState<BacktestResult | null>(null)
  const [btLoading, setBtLoading] = useState(false)
  const [cal, setCal] = useState<CalibrationResult | null>(null)
  const [calLoading, setCalLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.seasons().then(setSeasons).catch((e) => setError(String(e)))
  }, [])

  const runBacktest = async () => {
    setError(null); setBtLoading(true)
    try {
      setBt(await api.backtest({
        season_code: seasonCode, edge_threshold: edge,
        min_train_matches: minTrain, decay,
      }))
    } catch (e) { setError(String(e)) } finally { setBtLoading(false) }
  }

  const runCalibration = async () => {
    setError(null); setCalLoading(true)
    try { setCal(await api.calibration({ season_codes: seasons.map((s) => s.code) })) }
    catch (e) { setError(String(e)) } finally { setCalLoading(false) }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>⚽ Football Predictor</h1>
          <p className="subtitle">ML match predictions (XGBoost) — with the model’s real accuracy shown openly</p>
        </div>
        <nav className="tabs">
          {NAV.map((n) => (
            <button key={n.key}
              className={`${tab === n.key ? 'active' : ''} ${n.group === 'research' ? 'research' : ''}`}
              onClick={() => setTab(n.key)}>{n.label}</button>
          ))}
        </nav>
      </header>

      {error && <div className="card pad errorbox">{error}</div>}

      {tab === 'worldcup' && <WorldCupView />}
      {tab === 'today' && <DailyView />}
      {tab === 'predictions' && <PredictionsView />}
      {tab === 'quality' && <ModelQualityView />}

      {tab === 'backtest' && (
        <>
          <div className="card pad research-note muted small">
            Research tool — tests whether the model could beat bookmaker odds for betting.
            (It can’t, reliably; that’s why this is a predictions site, not a tipster.)
          </div>
          <div className="card pad controls">
            <label>Season
              <select value={seasonCode} onChange={(e) => setSeasonCode(e.target.value)}>
                {seasons.map((s) => <option key={s.code} value={s.code}>{s.label}</option>)}
              </select>
            </label>
            <label>Edge threshold: <b>{edge.toFixed(2)}</b>
              <input type="range" min={0} max={0.2} step={0.01} value={edge}
                onChange={(e) => setEdge(Number(e.target.value))} /></label>
            <label>Min train matches: <b>{minTrain}</b>
              <input type="range" min={20} max={150} step={10} value={minTrain}
                onChange={(e) => setMinTrain(Number(e.target.value))} /></label>
            <label>Time decay: <b>{decay.toFixed(4)}</b>
              <input type="range" min={0} max={0.005} step={0.0002} value={decay}
                onChange={(e) => setDecay(Number(e.target.value))} /></label>
            <button className="primary" onClick={runBacktest} disabled={btLoading}>
              {btLoading ? 'Running…' : 'Run backtest'}
            </button>
          </div>
          <BacktestView result={bt} loading={btLoading} />
        </>
      )}

      {tab === 'calibration' && (
        <>
          <div className="card pad controls">
            <span className="muted">
              Pools every fixture across all seasons and scores the Dixon-Coles model against the
              bookmaker’s no-vig closing line.
            </span>
            <button className="primary" onClick={runCalibration} disabled={calLoading}>
              {calLoading ? 'Computing…' : 'Run calibration'}
            </button>
          </div>
          <CalibrationView result={cal} loading={calLoading} />
        </>
      )}

      <footer className="foot">
        Predictions from a gradient-boosted model · evaluated out-of-sample · no lookahead ·
        not betting advice.
      </footer>
    </div>
  )
}
