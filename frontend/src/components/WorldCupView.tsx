import { useEffect, useState } from 'react'
import { api } from '../api'
import type { WcFixture, WcFixtures, WcPrediction, WcQuality } from '../api'
import WDLBar from './WDLBar'

const OUTCOME_LABEL = { H: 'Home win', D: 'Draw', A: 'Away win' }

function winnerLabel(p: WcPrediction | { home: string; away: string; pick: 'H' | 'D' | 'A' }) {
  return p.pick === 'H' ? p.home : p.pick === 'A' ? p.away : 'Draw'
}

function H2H({ teams }: { teams: string[] }) {
  const [home, setHome] = useState('')
  const [away, setAway] = useState('')
  const [pred, setPred] = useState<WcPrediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    if (teams.length && !home) { setHome(teams[0]); setAway(teams[1] ?? teams[0]) }
  }, [teams, home])

  const run = async () => {
    if (home === away) { setErr('Pick two different teams.'); return }
    setErr(null); setLoading(true)
    try { setPred(await api.wcPredict(home, away, true)) }
    catch (e) { setErr(String(e)) } finally { setLoading(false) }
  }

  return (
    <div className="card pad">
      <h3>Pick any two nations</h3>
      <p className="muted small" style={{ marginTop: -6, marginBottom: 14 }}>
        Neutral-venue prediction (as at a World Cup), from Elo + form over 150 years of internationals.
      </p>
      <div className="h2h-controls">
        <select value={home} onChange={(e) => setHome(e.target.value)}>
          {teams.map((t) => <option key={t}>{t}</option>)}
        </select>
        <span className="vs">vs</span>
        <select value={away} onChange={(e) => setAway(e.target.value)}>
          {teams.map((t) => <option key={t}>{t}</option>)}
        </select>
        <button className="primary" onClick={run} disabled={loading}>
          {loading ? 'Predicting…' : 'Predict'}
        </button>
      </div>
      {err && <div className="errorbox card pad" style={{ marginTop: 12 }}>{err}</div>}
      {pred && !err && (
        <div className="pred-result">
          <div className="pred-headline">
            <span className="teamname">{pred.home}</span>
            <span className="pred-score">{winnerLabel(pred) === 'Draw' ? 'Draw' : '→'}</span>
            <span className="teamname right">{pred.away}</span>
          </div>
          <WDLBar wdl={pred.wdl} pick={pred.pick} />
          <div className="pred-meta">
            <div><b>Most likely:</b> {OUTCOME_LABEL[pred.pick]} ({winnerLabel(pred)})</div>
            <div className="snapshots">
              <span>{pred.home}: Elo {pred.elo_home}</span>
              <span>{pred.away}: Elo {pred.elo_away}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default function WorldCupView() {
  const [fx, setFx] = useState<WcFixtures | null>(null)
  const [teams, setTeams] = useState<string[]>([])
  const [quality, setQuality] = useState<WcQuality | null>(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([api.wcFixtures(48), api.wcTeams()])
      .then(([f, t]) => { setFx(f); setTeams(t) })
      .catch((e) => setErr(String(e)))
      .finally(() => setLoading(false))
    api.wcQuality().then(setQuality).catch(() => {})
  }, [])

  if (loading) return <div className="card pad">Loading World Cup model and fixtures… (first load trains on 150 years of matches, ~40s)</div>
  if (err) return <div className="card pad errorbox">{err}</div>

  // Group fixtures by date.
  const byDate = new Map<string, WcFixture[]>()
  fx?.matches.forEach((m) => {
    if (!byDate.has(m.date)) byDate.set(m.date, [])
    byDate.get(m.date)!.push(m)
  })

  return (
    <div className="stack">
      <div className="card pad verdict" style={{ borderColor: '#4f9cf9' }}>
        <span className="verdict-dot" style={{ background: '#4f9cf9' }} />
        <div>
          <strong>FIFA World Cup 2026.</strong> Predictions for {fx?.count ?? 0} upcoming fixtures from a
          national-team Elo + XGBoost model.{' '}
          {quality && (
            <>Out-of-sample accuracy <b>{(quality.accuracy * 100).toFixed(0)}%</b> on{' '}
              {quality.n_test.toLocaleString()} international matches since {quality.test_since.slice(0, 4)}{' '}
              (vs {(quality.baseline * 100).toFixed(0)}% baseline). Honest, not a betting tip.</>
          )}
        </div>
      </div>

      {[...byDate.entries()].map(([date, matches]) => (
        <div key={date} className="card pad">
          <h3>{new Date(date).toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' })}</h3>
          <div className="board-grid">
            {matches.map((m, i) => (
              <div key={i} className="match-card">
                <div className="match-top">
                  <span className="match-date">{m.city ?? ''}</span>
                </div>
                <div className="match-teams">
                  <span>{m.home}</span><span className="muted">v</span><span>{m.away}</span>
                </div>
                <WDLBar wdl={m.wdl} pick={m.pick} compact />
                <div className="match-foot muted small">
                  pick: <b>{m.pick === 'D' ? 'Draw' : (m.pick === 'H' ? m.home : m.away)}</b>
                  {m.elo_home != null && m.elo_away != null && (
                    <> · Elo {m.elo_home} v {m.elo_away}</>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}

      <H2H teams={teams} />
    </div>
  )
}
