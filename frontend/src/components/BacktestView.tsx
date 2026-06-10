import {
  CartesianGrid, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import type { BacktestResult, Summary } from '../api'
import { pct, units, verdictColor } from '../util'

function StatCard({ label, value, hint, accent }: {
  label: string; value: string; hint?: string; accent?: string
}) {
  return (
    <div className="card stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={accent ? { color: accent } : undefined}>{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  )
}

// Merge the per-bet model curve and per-fixture favourite curve onto one
// date axis so they can be overlaid (connectNulls bridges the gaps).
function mergeSeries(result: BacktestResult) {
  const byDate = new Map<string, { date: string; model: number | null; fav: number | null }>()
  const ensure = (d: string) => {
    if (!byDate.has(d)) byDate.set(d, { date: d, model: null, fav: null })
    return byDate.get(d)!
  }
  for (const p of result.cumulative) ensure(p.date).model = p.cum_profit
  for (const p of result.favourite_cumulative) ensure(p.date).fav = p.cum_profit
  return [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date))
}

const baselineRows: { key: keyof BacktestResult['baselines'] | 'model'; label: string }[] = [
  { key: 'model', label: 'Model value bets' },
  { key: 'favourite', label: 'Always favourite' },
  { key: 'always_home', label: 'Always home' },
  { key: 'random', label: 'Random 1X2' },
]

export default function BacktestView({ result, loading }: {
  result: BacktestResult | null; loading: boolean
}) {
  if (loading) return <div className="card pad">Running walk-forward backtest…</div>
  if (!result) return <div className="card pad">Run a backtest to see results.</div>

  const m = result.model
  const series = mergeSeries(result)
  const getSummary = (k: string): Summary =>
    k === 'model' ? result.model : (result.baselines as Record<string, Summary>)[k]

  return (
    <div className="stack">
      <div className="card pad verdict" style={{ borderColor: verdictColor(result.verdict.level) }}>
        <span className="verdict-dot" style={{ background: verdictColor(result.verdict.level) }} />
        <div>
          <strong>Verdict ({result.season_label}):</strong> {result.verdict.text}
        </div>
      </div>

      <div className="grid stats">
        <StatCard label="ROI" value={pct(m.roi, true)}
          accent={verdictColor(result.verdict.level)} hint="profit ÷ staked — the number that matters" />
        <StatCard label="Value bets" value={String(m.n)} hint={`of ${result.fixtures_evaluated} fixtures`} />
        <StatCard label="Hit rate" value={pct(m.hit_rate)} />
        <StatCard label="Profit / loss" value={units(m.total_profit)} hint="flat 1-unit stakes" />
      </div>

      <div className="card pad">
        <h3>Cumulative profit</h3>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={series} margin={{ top: 8, right: 16, bottom: 8, left: -8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
            <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#9aa4b2' }} minTickGap={40} />
            <YAxis tick={{ fontSize: 11, fill: '#9aa4b2' }} />
            <Tooltip contentStyle={{ background: '#1a1f29', border: '1px solid #2a2f3a' }} />
            <Legend />
            <ReferenceLine y={0} stroke="#4a5568" />
            <Line type="monotone" dataKey="model" name="Model value bets"
              stroke="#4f9cf9" dot={false} strokeWidth={2} connectNulls />
            <Line type="monotone" dataKey="fav" name="Always favourite"
              stroke="#d69e2e" dot={false} strokeWidth={1.5} strokeDasharray="5 4" connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="row">
        <div className="card pad grow">
          <h3>Baselines (same fixtures)</h3>
          <table>
            <thead><tr><th>Strategy</th><th>Bets</th><th>Hit%</th><th>P/L</th><th>ROI</th></tr></thead>
            <tbody>
              {baselineRows.map(({ key, label }) => {
                const s = getSummary(key)
                return (
                  <tr key={key} className={key === 'model' ? 'highlight' : ''}>
                    <td>{label}</td><td>{s.n}</td><td>{pct(s.hit_rate)}</td>
                    <td>{units(s.total_profit)}</td>
                    <td style={{ color: (s.roi ?? 0) >= 0 ? '#1f9d55' : '#e53e3e' }}>{pct(s.roi, true)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        <div className="card pad grow">
          <h3>By flagged edge (do bigger edges win more?)</h3>
          <table>
            <thead><tr><th>Edge bucket</th><th>Bets</th><th>Hit%</th><th>P/L</th><th>ROI</th></tr></thead>
            <tbody>
              {result.edge_buckets.length === 0 && (
                <tr><td colSpan={5} className="muted">No bets flagged.</td></tr>
              )}
              {result.edge_buckets.map((b) => (
                <tr key={b.edge_bucket}>
                  <td>{b.edge_bucket}</td><td>{b.n}</td><td>{pct(b.hit_rate)}</td>
                  <td>{units(b.total_profit)}</td>
                  <td style={{ color: (b.roi ?? 0) >= 0 ? '#1f9d55' : '#e53e3e' }}>{pct(b.roi, true)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
