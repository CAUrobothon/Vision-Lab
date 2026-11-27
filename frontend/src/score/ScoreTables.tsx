// The 채점 page's result tables: 최종 순위 (complete, full-coverage batches by axis, Δ against the
// reference with its paired bootstrap interval), 중간 결과 · 비교 제외 (with the reason in words),
// 조각 비교 (slices × batches for one metric) and 증강 영향 (variant − original per batch).
import { Fragment, useMemo, useState, type ReactNode } from 'react'
import type { ModelInfo, ScoreBatch, ScoreResult, ScoreSlice } from '../api/types'
import { CUBE_STYLE } from '../lib/cubes'
import { changedParams, formatParamValue } from '../lib/params'
import { ClassSwatch } from '../ui/controls'
import { AXES, AXIS_START, cellOf, colName, coverageNd, DASH, diffOf, int, levelPct, METRIC_COLS, pct, SCORE_STATUS_LABEL, stat, type MetricCol } from './format'
import { BatchScope, DiffLine } from './parts'

// ----- batch names -----

function paramsText(b: Pick<ScoreBatch, 'params' | 'model_key'>, models: Map<string, ModelInfo>): { short: string; full: string } {
  const model = models.get(b.model_key)
  if (!model) return { short: '파라미터 비교 불가 (모델 없음)', full: '' }
  const names = changedParams(model.params, b.params)
  if (names.length === 0) return { short: '기본값', full: '기본값' }
  const parts = names.map((n) => {
    const spec = model.params.find((s) => s.name === n)
    return `${spec?.label ?? n} ${formatParamValue(spec, b.params[n])}`
  })
  return { short: `바꾼 값 ${names.length}개`, full: parts.join('\n') }
}

function BatchName({ b, models, children }: { b: ScoreBatch; models: Map<string, ModelInfo>; children?: ReactNode }) {
  const p = paramsText(b, models)
  return (
    <span className="sc-bname">
      <span className="sc-bline">
        <span className="sc-id num">#{b.batch_id}</span>
        <span className="sc-model" title={b.model_name}>
          {b.model_name}
        </span>
      </span>
      <span className="sc-bmeta">
        <BatchScope folder={b.folder} />
        <span className="faint" title={p.full}>
          {p.short}
        </span>
        {children}
      </span>
    </span>
  )
}

// ----- 최종 순위 -----

function MetricHead() {
  return (
    <>
      <tr>
        <th rowSpan={3} className="sc-sticky sc-rank-h r">
          순위
        </th>
        <th rowSpan={3} className="sc-sticky sc-batch-h">
          일괄 실행
        </th>
        {AXES.map((a) => (
          <th key={a.axis} colSpan={a.cols.length} className="sc-axis">
            {a.axis}
          </th>
        ))}
      </tr>
      <tr>
        {AXES.flatMap((a) =>
          a.cols
            .filter((c) => c.part !== 'p95')
            .map((c) => (
              <th
                key={c.diffKey}
                colSpan={c.sub ? 2 : 1}
                rowSpan={c.sub ? 1 : 2}
                className={`sc-mh${c.sub ? ' sc-mh-group' : ''}${AXIS_START.has(c.diffKey) ? ' sc-ax0' : ''}`}
              >
                {c.label}
              </th>
            )),
        )}
      </tr>
      <tr>
        {METRIC_COLS.filter((c) => c.sub).map((c) => (
          <th key={c.diffKey} className={`sc-mh sc-sub${AXIS_START.has(c.diffKey) ? ' sc-ax0' : ''}`}>
            {c.sub}
          </th>
        ))}
      </tr>
    </>
  )
}

function Cols({ lead }: { lead: number[] }) {
  return (
    <colgroup>
      {lead.map((w, i) => (
        <col key={i} style={{ width: w }} />
      ))}
      {METRIC_COLS.map((c) => (
        <col key={c.diffKey} className={c.kind === 'ratio' ? 'sc-col-ratio' : 'sc-col-stat'} />
      ))}
    </colgroup>
  )
}

function MetricCell({ b, c, compared, level }: { b: ScoreBatch; c: MetricCol; compared: boolean; level: number }) {
  const v = cellOf(b.metrics, c)
  return (
    <td data-metric={c.diffKey} className={`sc-m${c.metric === 'time_ms' ? ' vol' : ''}${AXIS_START.has(c.diffKey) ? ' sc-ax0' : ''}`}>
      <span className="sc-v num">{v.value}</span>
      {v.count ? <span className="sc-n num">{v.count}</span> : null}
      {compared ? (
        <DiffLine diff={diffOf(b.comparison?.result?.diffs, c)} kind={c.kind} unit={c.unit} better={c.better} level={level} />
      ) : null}
    </td>
  )
}

export function RankingTable({ result, models }: { result: ScoreResult; models: Map<string, ModelInfo> }) {
  const byId = new Map(result.batches.map((b) => [b.batch_id, b]))
  // Batch ids best first: the rank is the place in the list.
  const rows = result.final_ranking.map((id, i) => ({ rank: i + 1, b: byId.get(id)! })).filter((r) => r.b)
  if (rows.length === 0)
    return (
      <p className="sc-empty">
        최종 순위에 넣을 배치가 없습니다. 순위에는 정답셋 전체({int(result.gt_set.n_images)}장)를 채점하고 모든 실행이 끝난 일괄 실행만 들어갑니다.
      </p>
    )
  return (
    <div className="sc-table-wrap">
      <table className="table sc-table sc-rank">
        <Cols lead={[44, 184]} />
        <thead>
          <MetricHead />
        </thead>
        <tbody>
          {rows.map(({ rank, b }) => {
            const role = b.comparison?.role
            return (
              <tr key={b.batch_id} data-batch={b.batch_id} className={role === 'reference' ? 'is-ref' : undefined}>
                <td className="sc-sticky sc-rank-c r num">{rank}</td>
                <td className="sc-sticky sc-batch-c">
                  <BatchName b={b} models={models}>
                    {role === 'reference' ? (
                      <span className="sc-role ref" title="다른 배치의 Δ는 이 배치와의 차이입니다">
                        기준
                      </span>
                    ) : role === 'not_comparable' ? (
                      <span className="sc-role" title={b.comparison?.reason ?? undefined}>
                        비교 안 함
                      </span>
                    ) : null}
                  </BatchName>
                  {role === 'not_comparable' && b.comparison?.reason ? <span className="sc-why">{b.comparison.reason}</span> : null}
                </td>
                {METRIC_COLS.map((c) => (
                  <MetricCell key={c.diffKey} b={b} c={c} compared={role === 'compared'} level={result.criteria.bootstrap_level} />
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ----- 중간 결과 · 비교 제외 -----

// Unfinished images by reason, in this order; a count says 취소 rather than the status' 취소됨.
const PENDING = ['running', 'queued', 'cancelled', 'missing'] as const
const PENDING_LABEL: Record<(typeof PENDING)[number], string> = { ...SCORE_STATUS_LABEL, cancelled: '취소' }

/** Why a batch is not in the final ranking, in words (`nImages`: the GT set's images). */
function whyNotRanked(b: ScoreBatch, nImages: number): { state: string; reason: string } {
  if (b.status === 'interim') {
    const parts = PENDING.filter((k) => b.pending[k] > 0).map((k) => `${PENDING_LABEL[k]} ${int(b.pending[k])}`)
    return { state: '중간 결과', reason: parts.join(' · ') }
  }
  if (!b.full_coverage) {
    const where = b.folder
      ? ` (폴더 ${b.folder})`
      : nImages > b.n_eval_images
        ? ` (연결 안 된 정답 이미지 ${int(nImages - b.n_eval_images)})`
        : ''
    return { state: '비교 제외', reason: `부분 범위 ${coverageNd(b, nImages)}${where}` }
  }
  return { state: '비교 제외', reason: b.comparison?.reason ?? '' }
}

const INTERIM_COLS = METRIC_COLS.filter((c) => c.kind === 'ratio' || c.diffKey === 'time_ms_median')

export function InterimTable({ result, models }: { result: ScoreResult; models: Map<string, ModelInfo> }) {
  const ranked = new Set(result.final_ranking)
  const rows = result.batches.filter((b) => !ranked.has(b.batch_id))
  if (rows.length === 0) return <p className="sc-empty">고른 일괄 실행은 모두 최종 순위에 있습니다.</p>
  return (
    <div className="sc-table-wrap">
      <table className="table sc-table sc-interim">
        <colgroup>
          <col style={{ width: 184 }} />
          <col style={{ width: 220 }} />
          <col style={{ width: 92 }} />
          {INTERIM_COLS.map((c) => (
            <col key={c.diffKey} className={c.kind === 'ratio' ? 'sc-col-ratio' : 'sc-col-stat'} />
          ))}
        </colgroup>
        <thead>
          <tr>
            <th rowSpan={2} className="sc-sticky sc-batch-h0">
              일괄 실행
            </th>
            <th rowSpan={2}>상태 · 이유</th>
            <th rowSpan={2} className="r">
              채점된 이미지
            </th>
            <th colSpan={INTERIM_COLS.length} className="sc-axis">
              중간 점수 — 끝난 이미지만, 순위 · 차이 판정에 넣지 않음
            </th>
          </tr>
          <tr>
            {INTERIM_COLS.map((c) => (
              <th key={c.diffKey} className="sc-mh">
                {colName(c)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((b) => {
            const w = whyNotRanked(b, result.gt_set.n_images)
            return (
              <tr key={b.batch_id} data-batch={b.batch_id}>
                <td className="sc-sticky sc-batch-c0">
                  <BatchName b={b} models={models} />
                </td>
                <td className="sc-state">
                  <span className="sc-state-line">
                    <strong>{w.state}</strong>
                    <span className="tag-interim">중간 점수</span>
                  </span>
                  <span className="sc-reason num">{w.reason}</span>
                  {b.metrics.n_failed ? <span className="sc-reason num">실패 {int(b.metrics.n_failed)} (검출 0개로 셈)</span> : null}
                  {b.videos_skipped > 0 ? <span className="sc-reason num">영상 {int(b.videos_skipped)}개는 정답이 없어 건너뜀</span> : null}
                </td>
                <td className="r num sc-m">
                  <span className="sc-v">
                    {int(b.metrics.n_images ?? 0)}/{int(b.n_eval_images)}
                  </span>
                  <span className="sc-n">정답 {int(result.gt_set.n_images)}장 중</span>
                </td>
                {INTERIM_COLS.map((c) => {
                  const v = cellOf(b.metrics, c)
                  return (
                    <td key={c.diffKey} data-metric={c.diffKey} className={`sc-m${c.metric === 'time_ms' ? ' vol' : ''}`}>
                      <span className="sc-v num">{v.value}</span>
                      {v.count ? <span className="sc-n num">{v.count}</span> : null}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ----- metric picker shared by the slice and augmentation tables -----

function MetricPicker({ label, value, onChange, cols = METRIC_COLS }: { label: string; value: string; onChange: (k: string) => void; cols?: MetricCol[] }) {
  return (
    <select className="select sc-picker" aria-label={label} value={value} onChange={(e) => onChange(e.target.value)}>
      {AXES.map((a) => (
        <optgroup key={a.axis} label={a.axis}>
          {a.cols
            .filter((c) => cols.includes(c))
            .map((c) => (
              <option key={c.diffKey} value={c.diffKey}>
                {colName(c)}
                {c.cubeSlice ? '' : ' (이미지 단위만)'}
              </option>
            ))}
        </optgroup>
      ))}
    </select>
  )
}

function BatchHeads({ batches }: { batches: ScoreBatch[] }) {
  return (
    <>
      {batches.map((b) => (
        <th key={b.batch_id} className="sc-bh" title={b.model_name}>
          <span className="sc-id num">#{b.batch_id}</span> <span className="sc-bh-name">{b.model_name}</span>
          {b.status === 'interim' || !b.full_coverage ? <span className="tag-interim">중간</span> : null}
        </th>
      ))}
    </>
  )
}

// ----- 조각 비교 -----

function sliceLabel(dim: string, value: ScoreSlice['value']): ReactNode {
  if (value === null) return '값 없음'
  if (dim === 'color' && typeof value === 'string' && value in CUBE_STYLE) return <ClassSwatch color={value as keyof typeof CUBE_STYLE} withName />
  if (dim === 'near_contact') return value ? '근접 쌍' : '근접 아님'
  if (dim === 'weak_boundary') return value ? '경계 약함' : '경계 보통'
  return String(value)
}

export function SliceTable({ result }: { result: ScoreResult }) {
  const [key, setKey] = useState('tolerance_rate')
  const col = METRIC_COLS.find((c) => c.diffKey === key) ?? METRIC_COLS[0]
  const batches = useMemo(() => result.batches.filter((b) => (b.metrics.n_images ?? 0) > 0), [result.batches])
  // Every slice value any batch has, in the server's order (the first batch that has it).
  const groups = useMemo(() => {
    return result.slice_dimensions.map((d) => {
      const seen = new Map<string, ScoreSlice['value']>()
      for (const b of batches) for (const s of b.slices ?? []) if (s.dimension === d.key && !seen.has(String(s.value))) seen.set(String(s.value), s.value)
      return { dim: d, values: [...seen.values()] }
    })
  }, [result.slice_dimensions, batches])
  if (batches.length === 0) return <p className="sc-empty">채점된 이미지가 있는 배치가 없습니다.</p>
  return (
    <>
      <div className="sc-tools">
        <MetricPicker label="조각 지표" value={col.diffKey} onChange={setKey} />
        <span className="faint">칸마다 분자/분모 또는 표본 수 n. 위치 · 좌우 · 색 · 근접 · 경계 약함은 정답 큐브 단위입니다.</span>
      </div>
      <div className="sc-table-wrap fit">
        <table className="table sc-table sc-slices">
          <colgroup>
            <col style={{ width: 176 }} />
            {batches.map((b) => (
              <col key={b.batch_id} className="sc-col-batch" />
            ))}
          </colgroup>
          <thead>
            <tr>
              <th className="sc-sticky sc-batch-h0">조각</th>
              <BatchHeads batches={batches} />
            </tr>
          </thead>
          <tbody>
            {groups.map(({ dim, values }) => (
              <Fragment key={dim.key}>
                <tr className="sc-group">
                  <th colSpan={batches.length + 1} className="sc-sticky-row">
                    {dim.label} <span className="faint">({dim.unit === 'image' ? '이미지 단위' : '큐브 단위'})</span>
                  </th>
                </tr>
                {dim.unit === 'cube' && !col.cubeSlice ? (
                  <tr className="sc-undefined">
                    <td colSpan={batches.length + 1}>{col.label}은(는) 큐브 단위 조각에서 정의되지 않습니다 (이미지 단위 지표).</td>
                  </tr>
                ) : (
                  values.map((value) => {
                    const cells = batches.map((b) => b.slices?.find((s) => s.dimension === dim.key && String(s.value) === String(value)) ?? null)
                    const ns = [...new Set(cells.filter(Boolean).map((s) => s!.n))]
                    const unit = dim.unit === 'image' ? '이미지' : '큐브'
                    const uniform = ns.length === 1
                    return (
                      <tr key={String(value)} data-slice={`${dim.key}:${String(value)}`}>
                        <th scope="row" className="sc-sticky sc-slice-h">
                          <span className="sc-slice-name">{sliceLabel(dim.key, value)}</span>
                          {uniform ? <span className="sc-n num">{`${unit} ${int(ns[0])}`}</span> : null}
                        </th>
                        {cells.map((s, i) => {
                          if (!s)
                            return (
                              <td key={batches[i].batch_id} className="sc-m faint">
                                {DASH}
                              </td>
                            )
                          const v = cellOf(s.metrics, col)
                          return (
                            <td key={batches[i].batch_id} className={`sc-m${col.metric === 'time_ms' ? ' vol' : ''}`}>
                              <span className="sc-v num">{v.value}</span>
                              {v.count ? <span className="sc-n num">{v.count}</span> : null}
                              {uniform ? null : <span className="sc-n num">{`${unit} ${int(s.n)}`}</span>}
                            </td>
                          )
                        })}
                      </tr>
                    )
                  })
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

// ----- 증강 영향 -----

export function AugmentationTable({ result }: { result: ScoreResult }) {
  const [key, setKey] = useState('tolerance_rate')
  const col = METRIC_COLS.find((c) => c.diffKey === key) ?? METRIC_COLS[0]
  const batches = useMemo(() => result.batches.filter((b) => (b.augmentation_effect ?? []).length > 0), [result.batches])
  const variants = useMemo(() => {
    const order = result.gt_set.variants.map((v) => v.name)
    const present = new Set(batches.flatMap((b) => (b.augmentation_effect ?? []).map((e) => e.variant)))
    return [...order.filter((v) => present.has(v)), ...[...present].filter((v) => !order.includes(v))]
  }, [result.gt_set.variants, batches])
  if (batches.length === 0)
    return <p className="sc-empty">증강 이미지와 그 원본이 함께 채점된 배치가 없습니다 (증강 정보가 있는 정답셋에서만 나옵니다).</p>
  return (
    <>
      <div className="sc-tools">
        <MetricPicker label="증강 영향 지표" value={col.diffKey} onChange={setKey} />
        <span className="faint">
          Δ = 증강 이미지 − 같은 원본(source_sha256)의 original. 구간은 layout_group 단위 paired bootstrap ({levelPct(result.criteria.bootstrap_level)}).
        </span>
      </div>
      <div className="sc-table-wrap fit">
        <table className="table sc-table sc-aug">
          <colgroup>
            <col style={{ width: 176 }} />
            {batches.map((b) => (
              <col key={b.batch_id} className="sc-col-batch" />
            ))}
          </colgroup>
          <thead>
            <tr>
              <th className="sc-sticky sc-batch-h0">증강</th>
              <BatchHeads batches={batches} />
            </tr>
          </thead>
          <tbody>
            {variants.map((v) => (
              <tr key={v} data-variant={v}>
                <th scope="row" className="sc-sticky sc-slice-h">
                  <span className="sc-slice-name">{v}</span>
                </th>
                {batches.map((b) => {
                  const e = b.augmentation_effect?.find((x) => x.variant === v)
                  if (!e)
                    return (
                      <td key={b.batch_id} className="sc-m faint">
                        {DASH}
                      </td>
                    )
                  const d = diffOf(e.comparison.diffs, col)
                  return (
                    <td key={b.batch_id} className={`sc-m${col.metric === 'time_ms' ? ' vol' : ''}`}>
                      <span className="sc-v num" title="original → 증강">
                        {d && d.a !== null && d.b !== null ? `${fmtValue(d.a, col)} → ${fmtValue(d.b, col)}` : DASH}
                      </span>
                      <DiffLine diff={d} kind={col.kind} unit={col.unit} better={col.better} level={e.comparison.level} />
                      <span className="sc-n num">
                        원본 {int(e.n_sources)}쌍{e.n_unpaired ? ` · 원본 없는 ${int(e.n_unpaired)}장 뺌` : ''}
                      </span>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

function fmtValue(v: number, c: MetricCol): string {
  return c.kind === 'ratio' ? pct(v) : stat(v, c.unit)
}
