export const pct = (x: number | null | undefined, signed = false): string => {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  const v = x * 100
  return `${signed && v >= 0 ? '+' : ''}${v.toFixed(1)}%`
}

export const units = (x: number | null | undefined): string => {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  return `${x >= 0 ? '+' : ''}${x.toFixed(1)}u`
}

export const num = (x: number | null | undefined, dp = 3): string => {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  return x.toFixed(dp)
}

export const verdictColor = (level: string): string =>
  ({ good: '#1f9d55', weak: '#d69e2e', bad: '#e53e3e', none: '#718096' }[level] ?? '#718096')
