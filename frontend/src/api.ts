// Typed client for the football value-model research API.

export interface Season {
  code: string
  label: string
}

export interface Summary {
  n: number
  hit_rate: number | null
  total_profit: number | null
  staked: number | null
  roi: number | null
}

export interface EdgeBucket {
  edge_bucket: string
  n: number
  hit_rate: number | null
  total_profit: number | null
  roi: number | null
}

export interface CumPoint {
  date: string
  cum_profit: number | null
  home?: string
  away?: string
  outcome?: string
  won?: boolean
}

export interface Verdict {
  level: 'good' | 'weak' | 'bad' | 'none'
  text: string
}

export interface BacktestParams {
  season_code: string
  edge_threshold: number
  min_train_matches: number
  decay: number
  max_goals: number
}

export interface BacktestResult {
  params: BacktestParams
  season_label: string
  fixtures_evaluated: number
  model: Summary
  baselines: {
    favourite: Summary
    always_home: Summary
    random: Summary
  }
  edge_buckets: EdgeBucket[]
  cumulative: CumPoint[]
  favourite_cumulative: CumPoint[]
  verdict: Verdict
}

export interface ReliabilityRow {
  bin: string
  n: number
  mean_pred: number | null
  observed: number | null
  gap: number | null
}

export interface CalibrationResult {
  n_fixtures: number
  seasons: string[]
  scores: {
    model: { brier: number | null; log_loss: number | null }
    book: { brier: number | null; log_loss: number | null }
  }
  model_reliability: ReliabilityRow[]
  book_reliability: ReliabilityRow[]
}

// ---- Prediction site types -------------------------------------------- //
export interface WDL { H: number; D: number; A: number }

export interface TeamSnapshot {
  elo: number
  form_ppg: number | null
  played: number
}

export interface MatchPrediction {
  home: string
  away: string
  wdl: WDL
  pick: 'H' | 'D' | 'A'
  scoreline: string | null
  scoreline_prob: number | null
  over_2_5: number | null
  under_2_5: number | null
  home_snapshot: TeamSnapshot | null
  away_snapshot: TeamSnapshot | null
  league?: string
  country?: string
}

export interface League { code: string; name: string; country: string }

export interface BoardMatch {
  date: string
  home: string
  away: string
  wdl: WDL
  pick: 'H' | 'D' | 'A'
  actual: 'H' | 'D' | 'A'
  correct: boolean
}

export interface Board {
  season: string
  note: string
  matches: BoardMatch[]
}

export interface ModelQuality {
  holdout_season: string
  n_test: number
  metrics: {
    ml: { accuracy: number; brier: number; log_loss: number }
    baseline_home_accuracy: number
    book: { accuracy: number; brier: number; log_loss: number } | null
  }
  reliability: ReliabilityRow[]
  features: string[]
  form_window: number
}

export interface DailyMatch {
  country: string | null
  league: string | null
  home: string
  away: string
  kickoff: string | null
  status: 'predicted' | 'insufficient_data' | 'unsupported_league'
  wdl?: WDL
  pick?: 'H' | 'D' | 'A'
  demo?: boolean
}

export interface Daily {
  mode: 'live' | 'demo' | 'error'
  date: string
  note?: string
  error?: string
  total_fixtures?: number
  predicted?: number
  matches: DailyMatch[]
  supported_leagues: { code: string; name: string; country: string }[]
}

export interface WcFixture {
  date: string
  home: string
  away: string
  city: string | null
  country: string | null
  wdl: WDL
  pick: 'H' | 'D' | 'A'
  elo_home: number | null
  elo_away: number | null
}

export interface WcFixtures { count: number; matches: WcFixture[] }

export interface WcPrediction {
  home: string
  away: string
  wdl: WDL
  pick: 'H' | 'D' | 'A'
  neutral: boolean
  elo_home: number | null
  elo_away: number | null
}

export interface WcQuality {
  accuracy: number
  baseline: number
  brier: number
  log_loss: number
  n_test: number
  test_since: string
}

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${detail}`)
  }
  return res.json() as Promise<T>
}

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

export const api = {
  seasons: () => getJSON<Season[]>('/api/seasons'),
  backtest: (params: Partial<BacktestParams>) =>
    postJSON<BacktestResult>('/api/backtest', params),
  calibration: (body: { season_codes: string[] }) =>
    postJSON<CalibrationResult>('/api/calibration', body),
  // Prediction site
  mlTeams: () => getJSON<string[]>('/api/ml-teams'),
  predict: (home: string, away: string) =>
    getJSON<MatchPrediction>(
      `/api/predict?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`),
  board: (limit = 12) => getJSON<Board>(`/api/board?limit=${limit}`),
  modelQuality: () => getJSON<ModelQuality>('/api/model-quality'),
  daily: (date?: string) => getJSON<Daily>(`/api/daily${date ? `?date=${date}` : ''}`),
  leagues: () => getJSON<League[]>('/api/leagues'),
  leagueTeams: (league: string) =>
    getJSON<string[]>(`/api/league-teams?league=${encodeURIComponent(league)}`),
  predictLeague: (league: string, home: string, away: string) =>
    getJSON<MatchPrediction>(
      `/api/predict-league?league=${encodeURIComponent(league)}&home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`),
  // World Cup
  wcFixtures: (limit = 48) => getJSON<WcFixtures>(`/api/wc/fixtures?limit=${limit}`),
  wcTeams: () => getJSON<string[]>('/api/wc/teams'),
  wcQuality: () => getJSON<WcQuality>('/api/wc/quality'),
  wcPredict: (home: string, away: string, neutral = true) =>
    getJSON<WcPrediction>(
      `/api/wc/predict?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}&neutral=${neutral}`),
}
