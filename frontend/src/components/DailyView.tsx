import { useEffect, useState } from 'react'
import { api } from '../api'
import type { Daily, DailyMatch } from '../api'
import WDLBar from './WDLBar'

const OUTCOME_LABEL = { H: 'Home win', D: 'Draw', A: 'Away win' }

function MatchCard({ m }: { m: DailyMatch }) {
  const predicted = m.status === 'predicted' && m.wdl && m.pick
  return (
    <div className="match-card">
      <div className="match-top">
        <span className="match-date">{m.league ?? 'Unknown'}{m.country ? ` · ${m.country}` : ''}</span>
        {m.kickoff && <span className="match-date">{new Date(m.kickoff).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>}
      </div>
      <div className="match-teams">
        <span>{m.home}</span><span className="muted">v</span><span>{m.away}</span>
      </div>
      {predicted ? (
        <>
          <WDLBar wdl={m.wdl!} pick={m.pick} compact />
          <div className="match-foot muted small">prediction: <b>{OUTCOME_LABEL[m.pick!]}</b></div>
        </>
      ) : (
        <div className="match-foot muted small nodata">
          {m.status === 'insufficient_data'
            ? 'Not enough history for these teams — no prediction (kept honest).'
            : 'League not yet supported by the model.'}
        </div>
      )}
    </div>
  )
}

export default function DailyView() {
  const [data, setData] = useState<Daily | null>(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    api.daily().then(setData).catch((e) => setErr(String(e))).finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="card pad">Loading today’s fixtures and training league models… (first load ~2 min)</div>
  if (err) return <div className="card pad errorbox">{err}</div>
  if (!data) return null

  const predicted = data.matches.filter((m) => m.status === 'predicted')
  const others = data.matches.filter((m) => m.status !== 'predicted')

  return (
    <div className="stack">
      {data.mode === 'demo' && (
        <div className="card pad verdict" style={{ borderColor: '#d69e2e' }}>
          <span className="verdict-dot" style={{ background: '#d69e2e' }} />
          <div>
            <strong>Demo mode.</strong> These are sample fixtures, not today’s real matches.
            To go live, create a free <code>api-sports.io</code> account and set the
            <code> API_FOOTBALL_KEY</code> environment variable, then restart the backend.
          </div>
        </div>
      )}
      {data.mode === 'error' && (
        <div className="card pad errorbox">Fixtures provider error: {data.error}</div>
      )}

      <div className="card pad">
        <div className="board-head">
          <h3>{data.mode === 'live' ? `Today’s fixtures — ${data.date}` : 'Sample fixtures'}</h3>
          <span className="muted small">
            {data.mode === 'live'
              ? `${data.predicted ?? 0} of ${data.total_fixtures ?? 0} fixtures predictable with our trained leagues.`
              : (data.note ?? '')}
          </span>
        </div>
        <div className="feature-tags" style={{ marginBottom: 14 }}>
          {data.supported_leagues.map((l) => (
            <span key={l.code} className="tag">{l.name} · {l.country}</span>
          ))}
        </div>
        <div className="board-grid">
          {predicted.map((m, i) => <MatchCard key={`p${i}`} m={m} />)}
        </div>
        {predicted.length === 0 && (
          <p className="muted">No predictable matches right now (off-season, or today’s games are
            in leagues we haven’t trained yet).</p>
        )}
      </div>

      {others.length > 0 && (
        <div className="card pad">
          <h3>Other fixtures today ({others.length}) — not predicted</h3>
          <p className="muted small">Shown for completeness; we don’t fake numbers for matches
            outside our trained leagues or with too little team history.</p>
          <div className="board-grid">
            {others.slice(0, 24).map((m, i) => <MatchCard key={`o${i}`} m={m} />)}
          </div>
        </div>
      )}
    </div>
  )
}
