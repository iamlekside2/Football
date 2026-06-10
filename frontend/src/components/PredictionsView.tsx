import { useEffect, useState } from 'react'
import { api } from '../api'
import type { League, MatchPrediction } from '../api'
import WDLBar from './WDLBar'

const OUTCOME_LABEL = { H: 'Home win', D: 'Draw', A: 'Away win' }

export default function PredictionsView() {
  const [leagues, setLeagues] = useState<League[]>([])
  const [league, setLeague] = useState('E0')
  const [teams, setTeams] = useState<string[]>([])
  const [home, setHome] = useState('')
  const [away, setAway] = useState('')
  const [pred, setPred] = useState<MatchPrediction | null>(null)
  const [teamsLoading, setTeamsLoading] = useState(false)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    api.leagues().then(setLeagues).catch((e) => setErr(String(e)))
  }, [])

  // Load teams whenever the league changes.
  useEffect(() => {
    setTeamsLoading(true); setPred(null)
    api.leagueTeams(league)
      .then((t) => { setTeams(t); setHome(t[0] ?? ''); setAway(t[1] ?? '') })
      .catch((e) => setErr(String(e)))
      .finally(() => setTeamsLoading(false))
  }, [league])

  const run = async () => {
    if (home === away) { setErr('Pick two different teams.'); return }
    setErr(null); setLoading(true)
    try { setPred(await api.predictLeague(league, home, away)) }
    catch (e) { setErr(String(e)) }
    finally { setLoading(false) }
  }

  return (
    <div className="stack">
      <div className="card pad">
        <h3>Head-to-head predictor</h3>
        <p className="muted small" style={{ marginTop: -6, marginBottom: 14 }}>
          Pick a league, then any two of its teams. Each league has its own trained model.
        </p>
        <div className="h2h-controls">
          <select value={league} onChange={(e) => setLeague(e.target.value)}>
            {leagues.map((l) => (
              <option key={l.code} value={l.code}>{l.name} — {l.country}</option>
            ))}
          </select>
          <select value={home} onChange={(e) => setHome(e.target.value)} disabled={teamsLoading}>
            {teams.map((t) => <option key={t}>{t}</option>)}
          </select>
          <span className="vs">vs</span>
          <select value={away} onChange={(e) => setAway(e.target.value)} disabled={teamsLoading}>
            {teams.map((t) => <option key={t}>{t}</option>)}
          </select>
          <button className="primary" onClick={run} disabled={loading || teamsLoading}>
            {loading ? 'Predicting…' : teamsLoading ? 'Loading teams…' : 'Predict'}
          </button>
        </div>

        {err && <div className="errorbox card pad" style={{ marginTop: 12 }}>{err}</div>}

        {pred && !err && (
          <div className="pred-result">
            <div className="pred-headline">
              <span className="teamname">{pred.home}</span>
              <span className="pred-score">{pred.scoreline ?? '—'}</span>
              <span className="teamname right">{pred.away}</span>
            </div>
            <WDLBar wdl={pred.wdl} pick={pred.pick} />
            <div className="pred-meta">
              <div><b>Most likely:</b> {OUTCOME_LABEL[pred.pick]}
                {pred.scoreline && <> · likeliest score {pred.scoreline}</>}</div>
              {pred.over_2_5 != null && (
                <div><b>Goals:</b> Over 2.5 {Math.round(pred.over_2_5 * 100)}% ·
                  Under 2.5 {Math.round((pred.under_2_5 ?? 0) * 100)}%</div>
              )}
              <div className="snapshots">
                {pred.home_snapshot && (
                  <span>{pred.home}: Elo {pred.home_snapshot.elo}
                    {pred.home_snapshot.form_ppg != null && <> · {pred.home_snapshot.form_ppg} ppg</>}</span>
                )}
                {pred.away_snapshot && (
                  <span>{pred.away}: Elo {pred.away_snapshot.elo}
                    {pred.away_snapshot.form_ppg != null && <> · {pred.away_snapshot.form_ppg} ppg</>}</span>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
