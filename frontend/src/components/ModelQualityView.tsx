import { useEffect, useState } from 'react'
import {
  Bar, CartesianGrid, ComposedChart, Legend, Line,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api } from '../api'
import type { ModelQuality } from '../api'
import { num } from '../util'

function Stat({ label, value, hint, accent }: {
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

export default function ModelQualityView() {
  const [q, setQ] = useState<ModelQuality | null>(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    api.modelQuality().then(setQ).catch((e) => setErr(String(e))).finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="card pad">Evaluating the model out-of-sample…</div>
  if (err) return <div className="card pad errorbox">{err}</div>
  if (!q) return null

  const ml = q.metrics.ml
  const book = q.metrics.book
  const relData = q.reliability.map((r) => ({
    bin: r.bin, observed: r.observed, ideal: r.mean_pred,
  }))

  return (
    <div className="stack">
      <div className="card pad verdict" style={{ borderColor: '#d69e2e' }}>
        <span className="verdict-dot" style={{ background: '#d69e2e' }} />
        <div>
          Evaluated <b>out-of-sample</b> on the {q.holdout_season} season ({q.n_test} matches the
          model never trained on). The model is a genuine predictor — it beats the naive
          baseline — but, honestly, it does <b>not</b> beat the bookmaker’s line. We show both so
          you can trust the numbers rather than take them on faith.
        </div>
      </div>

      <div className="grid stats">
        <Stat label="ML accuracy" value={`${(ml.accuracy * 100).toFixed(1)}%`}
          accent="#4f9cf9" hint="3-way (home/draw/away)" />
        <Stat label="Baseline (always home)"
          value={`${(q.metrics.baseline_home_accuracy * 100).toFixed(1)}%`}
          hint="what guessing 'home' scores" />
        <Stat label="Bookmaker accuracy"
          value={book ? `${(book.accuracy * 100).toFixed(1)}%` : '—'}
          accent="#1f9d55" hint="the line we're measured against" />
        <Stat label="ML Brier / log-loss"
          value={`${num(ml.brier, 3)} / ${num(ml.log_loss, 3)}`}
          hint={book ? `book: ${num(book.brier, 3)} / ${num(book.log_loss, 3)}` : undefined} />
      </div>

      <div className="card pad">
        <h3>Calibration — predicted probability vs what actually happened</h3>
        <p className="muted small">
          Bars = observed frequency in each probability bin; line = what the model predicted.
          Bars sitting below the line in the high bins means the model is a bit overconfident.
        </p>
        <ResponsiveContainer width="100%" height={320}>
          <ComposedChart data={relData} margin={{ top: 8, right: 16, bottom: 8, left: -8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
            <XAxis dataKey="bin" tick={{ fontSize: 10, fill: '#9aa4b2' }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: '#9aa4b2' }} />
            <Tooltip contentStyle={{ background: '#1a1f29', border: '1px solid #2a2f3a' }} />
            <Legend />
            <Bar dataKey="observed" name="Observed frequency" fill="#4f9cf9" />
            <Line type="monotone" dataKey="ideal" name="Model predicted" stroke="#d69e2e" strokeWidth={2} dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="card pad">
        <h3>Features used ({q.features.length})</h3>
        <div className="feature-tags">
          {q.features.map((f) => <span key={f} className="tag">{f}</span>)}
        </div>
        <p className="muted small" style={{ marginTop: 10 }}>
          All features are computed from information available <b>before</b> kick-off
          (rolling window: last {q.form_window} matches). No lookahead.
        </p>
      </div>
    </div>
  )
}
