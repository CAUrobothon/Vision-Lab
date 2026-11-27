// How scores read on screen: ratios as % with numerator / denominator, samples with their n,
// differences in %p or units with the bootstrap interval, verdicts in words. Null is "—", never 0.
import type { ScoreDiff, ScoreMetrics, ScoreRatio, ScoreStats, Verdict } from '../api/types'
import { formatMs } from '../lib/format'
import { STATUS_LABEL } from '../ui/controls'

export const DASH = '—'
const MINUS = '−'

export function int(n: number): string {
  return n.toLocaleString('en-US')
}

export function pct(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined ? DASH : `${(v * 100).toFixed(digits)}%`
}

function ratioPct(r: ScoreRatio | null | undefined): string {
  return pct(r?.value)
}

export function ratioNd(r: ScoreRatio | null | undefined): string {
  return r ? `${int(r.numerator)}/${int(r.denominator)}` : DASH
}

/** A batch's coverage: its evaluation images / the GT set's images. */
export function coverageNd(b: { n_eval_images: number }, nImages: number): string {
  return `${int(b.n_eval_images)}/${int(nImages)}`
}

function signed(v: number, digits: number): string {
  const s = Math.abs(v).toFixed(digits)
  if (Number(s) === 0) return s
  return (v > 0 ? '+' : MINUS) + s
}

/** A sample value in its unit (none = the bare number): px and ° with `digits` decimals, ms like
 *  everywhere else. */
export function stat(v: number | null | undefined, unit = '', digits = 2): string {
  if (v === null || v === undefined) return DASH
  if (unit === 'ms') return formatMs(v)
  return unit === '°' ? `${v.toFixed(digits)}°` : `${v.toFixed(digits)} ${unit}`.trim()
}

/** A bootstrap interval's level in percent: 0.95 -> "95%". */
export function levelPct(level: number): string {
  return `${Math.round(level * 100)}%`
}

export type ValueKind = 'ratio' | 'stat'

export function delta(v: number | null | undefined, kind: ValueKind, unit: string): string {
  if (v === null || v === undefined) return DASH
  if (kind === 'ratio') return `${signed(v * 100, 1)}%p`
  return unit === '°' ? `${signed(v, 2)}°` : `${signed(v, unit === 'ms' ? 1 : 2)} ${unit}`
}

export function interval(lo: number | null, hi: number | null, kind: ValueKind): string {
  if (lo === null || hi === null) return DASH
  const f = (v: number) => (kind === 'ratio' ? signed(v * 100, 1) : signed(v, 2))
  return `[${f(lo)}, ${f(hi)}]`
}

export const VERDICT_TEXT: Record<Verdict, string> = { higher: '높음', lower: '낮음', unclear: '차이 불분명' }

/** A run's status in words, plus "missing": the batch has no run for an image. */
export const SCORE_STATUS_LABEL = { ...STATUS_LABEL, missing: '배치에 없음' }

/** A batch in a picker: "#12 HSV 색 범위 (증강1)". */
export function batchLabel(b: { model_name: string; folder: string | null } & ({ id: number } | { batch_id: number })): string {
  return `#${'batch_id' in b ? b.batch_id : b.id} ${b.model_name}${b.folder ? ` (${b.folder})` : ''}`
}

/** Better / worse / no call: the text says the direction, the tone what it means for this metric. */
export function tone(v: Verdict | null, better: 'higher' | 'lower'): 'good' | 'bad' | 'flat' {
  if (v === null || v === 'unclear') return 'flat'
  return v === better ? 'good' : 'bad'
}

// ----- the metric columns, grouped by axis (docs/SCORING.md §3) -----

type Part = 'ratio' | 'median' | 'p95'

export interface MetricCol {
  diffKey: string // ScoreDiff.key
  metric: keyof ScoreMetrics
  part: Part
  kind: ValueKind
  unit: string
  better: 'higher' | 'lower'
  label: string // the metric's name
  sub: string | null // 중앙값 / P95 for samples
  cubeSlice: boolean // defined in cube-level slices
}

const ratio = (metric: keyof ScoreMetrics, label: string, cubeSlice: boolean): MetricCol => ({
  diffKey: metric, metric, part: 'ratio', kind: 'ratio', unit: '', better: 'higher', label, sub: null, cubeSlice,
})
const sample = (metric: keyof ScoreMetrics, label: string, unit: string, cubeSlice: boolean): MetricCol[] =>
  (['median', 'p95'] as const).map((part) => ({
    diffKey: `${metric}_${part}`, metric, part, kind: 'stat', unit, better: 'lower', label,
    sub: part === 'median' ? '중앙값' : 'P95', cubeSlice,
  }))

export const AXES: { axis: string; cols: MetricCol[] }[] = [
  { axis: '종합', cols: [ratio('tolerance_rate', '허용 오차 충족률', true), ratio('image_pass_rate', '이미지 전체 기준 충족률', false)] },
  { axis: '객체 검출', cols: [ratio('recall', 'Recall', true), ratio('precision', 'Precision', false)] },
  { axis: '색상 분류', cols: [ratio('color_accuracy', '색상 정확도', true)] },
  {
    axis: '윗면 추출',
    cols: [...sample('vertex_err', '꼭짓점 RMS', 'px', true), ...sample('center_err', '중심 거리', 'px', true), ...sample('angle_err', '회전각', '°', true)],
  },
  { axis: '처리 성능', cols: sample('time_ms', '이미지당 처리 시간', 'ms', false) },
]

export const METRIC_COLS: MetricCol[] = AXES.flatMap((a) => a.cols)

/** The first column of each axis (drawn with a separator). */
export const AXIS_START = new Set(AXES.map((a) => a.cols[0].diffKey))

export function colName(c: MetricCol): string {
  return c.sub ? `${c.label} ${c.sub}` : c.label
}

/** The column's value in a metrics object as display text and its count line. */
export function cellOf(m: ScoreMetrics, c: MetricCol): { value: string; count: string } {
  const v = m[c.metric] as ScoreRatio | ScoreStats | null
  if (v === null || v === undefined) return { value: DASH, count: '' }
  if (c.kind === 'ratio') {
    const r = v as ScoreRatio
    return { value: ratioPct(r), count: ratioNd(r) }
  }
  const s = v as ScoreStats
  return { value: stat(c.part === 'median' ? s.median : s.p95, c.unit), count: `n ${int(s.n)}` }
}

export function diffOf(diffs: ScoreDiff[] | undefined, c: MetricCol): ScoreDiff | null {
  return diffs?.find((d) => d.key === c.diffKey) ?? null
}
