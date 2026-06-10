import {
  Bar, CartesianGrid, Legend, Line, ComposedChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import type { CalibrationResult } from '../api'
import { num } from '../util'

export default function CalibrationView({ result, loading }: {
  result: CalibrationResult | null; loading: boolean
}) {
  if (loading) return <div className="card pad">Computing calibration across seasons…</div>
  if (!result) return <div className="card pad">Run calibration to compare the model against the closing line.</div>

  const s = result.scores
  const bookBetter = (s.book.brier ?? 1) < (s.model.brier ?? 1)

  // Reliability: predicted probability vs observed frequency. A perfectly
  // calibrated predictor sits on the diagonal (mean_pred == observed).
  const relData = result.model_reliability.map((r, i) => ({
    bin: r.bin,
    diagonal: r.mean_pred,
    model: r.observed,
    book: result.book_reliability[i]?.observed ?? null,
  }))

  return (
    <div className="stack">
      <div className="card pad verdict" style={{ borderColor: bookBetter ? '#e53e3e' : '#1f9d55' }}>
        <span className="verdict-dot" style={{ background: bookBetter ? '#e53e3e' : '#1f9d55' }} />
        <div>
          {bookBetter ? (
            <><strong>The bookmaker’s closing line is better calibrated than the model.</strong>{' '}
              The model cannot have a real 1X2 edge against this line — its “value” flags land
              where it is most wrong, not where the book is.</>
          ) : (
            <><strong>The model scores better than the closing line here.</strong>{' '}
              Surprising — double-check for leakage before believing it.</>
          )}
        </div>
      </div>

      <div className="grid stats">
        <div className="card stat">
          <div className="stat-label">Model Brier</div>
          <div className="stat-value">{num(s.model.brier, 4)}</div>
          <div className="stat-hint">lower is better</div>
        </div>
        <div className="card stat">
          <div className="stat-label">Book Brier</div>
          <div className="stat-value" style={{ color: '#1f9d55' }}>{num(s.book.brier, 4)}</div>
          <div className="stat-hint">closing line (no-vig)</div>
        </div>
        <div className="card stat">
          <div className="stat-label">Model log-loss</div>
          <div className="stat-value">{num(s.model.log_loss, 4)}</div>
        </div>
        <div className="card stat">
          <div className="stat-label">Book log-loss</div>
          <div className="stat-value" style={{ color: '#1f9d55' }}>{num(s.book.log_loss, 4)}</div>
          <div className="stat-hint">{result.n_fixtures} fixtures pooled</div>
        </div>
      </div>

      <div className="card pad">
        <h3>Reliability — predicted probability vs observed frequency</h3>
        <p className="muted small">
          Bars = how often the outcome actually happened in each probability bin.
          The line = what the model predicted (the ideal: bar height equals the line).
          Model bars below its line in the high bins ⇒ overconfidence.
        </p>
        <ResponsiveContainer width="100%" height={340}>
          <ComposedChart data={relData} margin={{ top: 8, right: 16, bottom: 8, left: -8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
            <XAxis dataKey="bin" tick={{ fontSize: 10, fill: '#9aa4b2' }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: '#9aa4b2' }} />
            <Tooltip contentStyle={{ background: '#1a1f29', border: '1px solid #2a2f3a' }} />
            <Legend />
            <Bar dataKey="model" name="Observed (model bins)" fill="#4f9cf9" />
            <Bar dataKey="book" name="Observed (book bins)" fill="#2c5282" />
            <Line type="monotone" dataKey="diagonal" name="Perfect calibration"
              stroke="#d69e2e" dot={false} strokeWidth={2} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="card pad">
        <h3>Model reliability table</h3>
        <table>
          <thead><tr><th>Prob bin</th><th>n</th><th>Mean predicted</th><th>Observed</th><th>Gap</th></tr></thead>
          <tbody>
            {result.model_reliability.map((r) => (
              <tr key={r.bin}>
                <td>{r.bin}</td><td>{r.n}</td><td>{num(r.mean_pred)}</td><td>{num(r.observed)}</td>
                <td style={{ color: Math.abs(r.gap ?? 0) > 0.05 ? '#e53e3e' : '#9aa4b2' }}>
                  {(r.gap ?? 0) >= 0 ? '+' : ''}{num(r.gap)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
