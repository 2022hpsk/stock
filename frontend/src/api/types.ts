/**
 * 后端契约类型。
 *
 * 与 `src/quantstock/web/serializers.py` 一一对应。金额字段一律 `string`——
 * 类型上就把 `Decimal` 语义钉住，避免有人顺手写 `price * qty`。
 */

export interface SystemStatus {
  ok: boolean
  version: string
  checked_at: string
  readonly: boolean
  broker: string
  llm: { enabled: boolean; mode: string }
  halt: { halted: boolean; reason: string; halted_at: string | null; halted_by: string }
  components: Array<{ name: string; ok: boolean; detail: string }>
}

export interface Evidence {
  name: string
  value: number | null
  detail: string
  direction: string
}

export interface IntelEvidence {
  title: string
  source: string
  url: string
  published_at: string | null
  domain: string
  sentiment: number | null
  importance: number
  impact: string
  summary: string
}

export interface PositionAnalytics {
  symbol: string
  market_price: string | null
  is_held: boolean
  holding_days: number
  cost_basis: string | null
  unrealized_pnl_pct: number | null
  days_to_tax_free: number | null
  tax_saving_if_wait: string | null
  ma5: number | null
  ma20: number | null
  ma60: number | null
  ma_alignment: string
  stop_loss_price: string | null
  distance_to_stop_pct: number | null
  statements: string[]
}

export interface Rationale {
  verdict: string
  quant_evidence: Evidence[]
  technical: PositionAnalytics
  intel_evidence: IntelEvidence[]
  intel_absent_note: string
  counter_evidence: Evidence[]
  falsification: string[]
  risk_notes: string[]
  confidence: number | null
  confidence_basis: string
  llm_involved: boolean
  llm_adjustment: number | null
  is_complete: boolean
  missing_pillars: string[]
}

export interface TradeIntent {
  intent_id: string
  symbol: string
  side: string
  qty: number
  price_low: string | null
  price_high: string | null
  estimated_amount: string | null
  urgency: string
  stop_loss: string | null
  take_profit: string | null
  rationale: Rationale
}

export interface TradePlan {
  plan_id: string
  account_id: string
  trade_date: string
  generated_at: string
  circuit_state: string
  summary: string
  intents: TradeIntent[]
  rejected: Array<{ symbol: string; reason: string; rule_id: string }>
  incomplete: Array<{ symbol: string; missing: string }>
  total_buy_amount: string | null
  total_sell_amount: string | null
  data_fingerprint: string
  strategy_versions: Record<string, string>
  param_hash: string
  confirmed_by: string
  confirmed_at: string | null
  is_confirmed: boolean
}

export interface AdviceResponse {
  plan: TradePlan
  saved_to: string
  summary: string
  llm_used: boolean
  base_scores: Record<string, number>
  final_scores: Record<string, number>
  llm_notes: Record<string, string>
  skipped: Array<{ symbol: string; reason: string }>
}

export interface IntentPreview {
  intent_id: string
  symbol: string
  side: string
  qty: number
  price_low: string | null
  price_high: string | null
  limit_price: string | null
  estimated_amount: string | null
  urgency: string
  verdict: string
  needs_review: boolean
  drift: {
    reference_price: string | null
    current_price: string | null
    drift_pct: number | null
    exceeded: boolean
  } | null
}

export interface ExecutionPreview {
  plan_id: string
  trade_date: string
  broker: string
  requires_live_flag: boolean
  halted: boolean
  halt_reason: string
  total_buy: string | null
  total_sell: string | null
  review_count: number
  items: IntentPreview[]
}

export interface DataStatus {
  root: string
  symbols: number
  files: number
  bytes_on_disk: number
  latest_date: string | null
  instruments: number
  delisted: number
  is_ready: boolean
  message: string
  health: Array<{
    source: string
    ok: boolean
    checked_at: string
    message: string
    latency_ms: number
    consecutive_failures: number
  }>
}

export interface Bar {
  date: string
  open: string
  high: string
  low: string
  close: string
  volume: number
  amount: string | null
}

export interface IntelItem {
  item_id: string
  source: string
  source_tier: string
  domain: string
  publish_at: string | null
  title: string
  body: string
  url: string
  symbols: string[]
  event_type: string | null
  importance: number
  sentiment: number | null
  classifier: string
  duplicates: string[]
}

export interface IntelDigest {
  trade_date: string
  generated_at: string
  session: string
  by_domain: Record<
    string,
    {
      highlights: string[]
      count: number
      net_sentiment: number
      llm_generated: boolean
      symbols: string[]
      items: IntelItem[]
    }
  >
  top_items: IntelItem[]
  portfolio_alerts: Array<{
    symbol: string
    severity: string
    action_hint: string
    item: IntelItem
  }>
  watchlist_hits: IntelItem[]
  missing_domains: string[]
  failed_sources: string[]
  lines: string[]
}

export interface BacktestReport {
  start: string
  end: string
  trading_days: number
  initial_cash: number
  final_equity: number
  fills: number
  rejections: Record<string, number>
  trial_id: string
  universe: string[]
  llm_mode: string
  explain: string
  warnings: string[]
  stats: {
    total_return: number
    annualized_return: number
    annualized_volatility: number
    sharpe: number
    sortino: number
    calmar: number
    max_drawdown: number
    max_drawdown_duration: number
    win_rate: number
    profit_loss_ratio: number
    trading_days: number
    twr: number
    mwr: number
  }
}

export interface Admission {
  strategy: string
  available: boolean
  message?: string
  admitted?: boolean
  dsr?: number
  pbo?: number
  n_trials?: number
  reasons?: string[]
  explain?: string
}

export interface LlmStatus {
  enabled: boolean
  mode: string
  alpha: number
  prompt_version: string
  cached_entries: number
  cached_cost_usd: number
  daily_spent_usd: number
  monthly_spent_usd: number
  degraded: boolean
  degraded_reason: string
  message: string
  cache_dir: string
  tasks: Array<{ name: string; enabled: boolean; model: string }>
  param_hash_parts: Record<string, string>
}

export interface WsEvent {
  seq: number
  channel: string
  kind: string
  payload: Record<string, unknown>
  at: string
}
