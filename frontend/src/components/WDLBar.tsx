import type { WDL } from '../api'

const COLORS = { H: '#4f9cf9', D: '#8893a5', A: '#f6915d' }

// Horizontal stacked probability bar for Home / Draw / Away.
export default function WDLBar({ wdl, pick, compact }: {
  wdl: WDL; pick?: 'H' | 'D' | 'A'; compact?: boolean
}) {
  const parts: { k: 'H' | 'D' | 'A'; v: number }[] = [
    { k: 'H', v: wdl.H }, { k: 'D', v: wdl.D }, { k: 'A', v: wdl.A },
  ]
  return (
    <div className={`wdlbar ${compact ? 'compact' : ''}`}>
      <div className="wdlbar-track">
        {parts.map(({ k, v }) => (
          <div key={k} className="wdlbar-seg"
            style={{
              width: `${Math.max(v * 100, 0)}%`,
              background: COLORS[k],
              outline: pick === k ? '2px solid #fff' : 'none',
              outlineOffset: '-2px',
            }}
            title={`${k}: ${(v * 100).toFixed(1)}%`}>
            {v > 0.12 && <span>{Math.round(v * 100)}%</span>}
          </div>
        ))}
      </div>
      {!compact && (
        <div className="wdlbar-legend">
          <span><i style={{ background: COLORS.H }} /> Home {Math.round(wdl.H * 100)}%</span>
          <span><i style={{ background: COLORS.D }} /> Draw {Math.round(wdl.D * 100)}%</span>
          <span><i style={{ background: COLORS.A }} /> Away {Math.round(wdl.A * 100)}%</span>
        </div>
      )}
    </div>
  )
}
