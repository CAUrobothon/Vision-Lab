// Result of what the viewer shows: counts per colour, the cube table (hover/click ↔ viewer
// highlight), processing time with per-stage times, and the pipeline's notes. For an image with
// ground truth, the score of those cubes (a run's own, or the live one of a preview): the image's
// pass, counts, each detection's outcome and a 정답 tab with every GT cube's errors.
import { memo, useEffect, useRef, useState } from 'react'
import { AlertTriangle, Check, SlidersHorizontal, X } from 'lucide-react'
import type { Cube, ScoreImageDetail, Timing } from '../api/types'
import { CUBE_COLORS } from '../api/types'
import { countColors } from '../lib/cubes'
import { formatMs } from '../lib/format'
import { loadJson, saveJson } from '../lib/storage'
import { useCriteria } from '../score/criteria'
import { stat } from '../score/format'
import { gtOfPred } from '../score/image'
import { CountsLine, CriteriaFields, CriteriaLine, PassBadge } from '../score/parts'
import { ClassSwatch } from '../ui/controls'

export interface ResultTiming {
  ms: number | null // this frame
  stages: Record<string, number> | null
  rtt: number | null // preview round trip seen by the browser
  run: Timing | null
}

/** The score of the cubes shown, for an image with ground truth (null = it has none). */
export interface LiveScore {
  detail: ScoreImageDetail | null // null until the first answer
  pending: boolean
  error: string | null
}

function Outcome({ d, pred }: { d: ScoreImageDetail; pred: number }) {
  const gi = gtOfPred(d, pred)
  if (gi === null)
    return d.false_positives.includes(pred) ? (
      <span className="ls-mark fail-text" title="오검출: 게이트 안에 짝이 되는 정답이 없음">
        오검출
      </span>
    ) : (
      <span className="faint">—</span>
    )
  const c = d.cubes[gi]
  const o = c.outcome
  const title = `정답 ${d.gt[gi].id} · 짝 거리 ${stat(c.distance, 'px')} · 중심 ${stat(c.center_err, 'px')} · 회전각 ${stat(c.angle_err, '°', 1)} · 꼭짓점 RMS ${stat(c.vertex_err, 'px')}${c.color_ok ? '' : ' · 색 틀림'}`
  return o === 'pass' ? (
    <Check size={13} strokeWidth={2.5} className="ls-mark ok-text" aria-label={`통과 (${title})`} />
  ) : (
    <X size={13} strokeWidth={2.5} className="ls-mark fail-text" aria-label={`미달 (${title})`} />
  )
}

function LiveHead({ score }: { score: LiveScore }) {
  const [editing, setEditing] = useState(false)
  const [c] = useCriteria()
  const d = score.detail
  return (
    <section className={`live-score${score.pending ? ' pending' : ''}`} aria-label="정답 비교">
      <div className="ls-line num">
        {score.error ? (
          <span className="error-text">{score.error}</span>
        ) : !d ? (
          <span className="faint">정답과 비교하는 중…</span>
        ) : !d.scored ? (
          <span className="faint">실행이 끝나면 정답과 비교합니다</span>
        ) : (
          <>
            <PassBadge passed={!!d.passed} className="ls-verdict" />
            <CountsLine d={d} short className="ls-counts" title={`정답셋 ${d.gt_set_name} · ${d.filename}`} />
          </>
        )}
        <button
          type="button"
          className="icon-btn sm ls-edit"
          aria-expanded={editing}
          aria-label="채점 기준"
          title="채점 기준 (모든 화면이 같이 씁니다)"
          onClick={() => setEditing((v) => !v)}
        >
          <SlidersHorizontal size={12} strokeWidth={1.75} />
        </button>
      </div>
      {editing ? (
        <CriteriaFields bootstrap={false} />
      ) : (
        <CriteriaLine c={c} className="ls-crit faint num" />
      )}
    </section>
  )
}

function GtTable({ d, highlight, onHighlight }: { d: ScoreImageDetail; highlight: number | null; onHighlight: (i: number | null) => void }) {
  return (
    <table className="table result-table gt-table" onMouseLeave={() => onHighlight(null)}>
      <thead>
        <tr>
          <th>정답</th>
          <th className="r">검출</th>
          <th className="r">중심</th>
          <th className="r">각도</th>
          <th className="r">RMS</th>
          <th>결과</th>
        </tr>
      </thead>
      <tbody>
        {d.gt.map((g, i) => {
          const c = d.cubes[i]
          const o = c?.outcome ?? null
          const pred = c?.pred_index ?? null
          return (
            <tr
              key={g.id}
              className={pred !== null && pred === highlight ? 'hl' : undefined}
              onMouseEnter={() => onHighlight(pred)}
              title={`${g.id} · 짝 거리 ${stat(c?.distance, 'px')}${c && !c.color_ok && c.matched ? ` · 검출 색 ${c.pred_color}` : ''}${g.valid_geometry ? '' : ' · 기하 무효: 짝과 색만 봄'}`}
            >
              <td>
                <ClassSwatch color={g.color} /> <span className="gt-id">{g.id}</span>
              </td>
              <td className="r num faint">{pred === null ? '—' : pred + 1}</td>
              <td className="r num">{stat(c?.center_err)}</td>
              <td className="r num">{stat(c?.angle_err, '°', 1)}</td>
              <td className="r num">{stat(c?.vertex_err)}</td>
              <td>
                {o === 'pass' ? (
                  <span className="ls-mark ok-text">통과</span>
                ) : o === 'missed' ? (
                  <span className="ls-mark fail-text">미검출</span>
                ) : o === 'fail' ? (
                  <span className="ls-mark fail-text">{c.matched && !c.color_ok ? '색 틀림' : '미달'}</span>
                ) : (
                  <span className="faint">—</span>
                )}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

interface ResultPanelProps {
  title: string
  cubes: Cube[] | null // null = nothing to show yet
  highlight: number | null
  onHighlight: (i: number | null) => void
  timing: ResultTiming | null
  notes: string[]
  error: string | null
  empty: string
  score?: LiveScore | null // the media has ground truth
}

export const ResultPanel = memo(function ResultPanel({ title, cubes, highlight, onHighlight, timing, notes, error, empty, score = null }: ResultPanelProps) {
  const rowRefs = useRef<(HTMLTableRowElement | null)[]>([])
  const [tab, setTabRaw] = useState<'cubes' | 'gt'>(() => (loadJson<string>('result.tab', 'cubes') === 'gt' ? 'gt' : 'cubes'))
  const setTab = (t: 'cubes' | 'gt') => {
    setTabRaw(t)
    saveJson('result.tab', t)
  }
  const d = score?.detail ?? null
  const showGt = score !== null && tab === 'gt' && d !== null
  useEffect(() => {
    if (highlight !== null) rowRefs.current[highlight]?.scrollIntoView({ block: 'nearest' })
  }, [highlight])
  const counts = cubes ? countColors(cubes) : null

  return (
    <section className={`result${score ? ' has-gt' : ''}`} aria-label="결과">
      <div className="result-head">
        <h2 className="panel-title">결과</h2>
        {score ? (
          <div className="segmented result-tabs" role="group" aria-label="결과 표">
            <button type="button" aria-pressed={tab === 'cubes'} onClick={() => setTab('cubes')}>
              검출
            </button>
            <button type="button" aria-pressed={tab === 'gt'} onClick={() => setTab('gt')}>
              정답
            </button>
          </div>
        ) : null}
        <span className="faint result-src num">{title}</span>
      </div>

      {score ? <LiveHead score={score} /> : null}

      {error ? (
        <div className="notice error">
          <AlertTriangle size={14} strokeWidth={1.75} />
          <span>{error}</span>
        </div>
      ) : null}

      {counts && !showGt ? (
        <div className="result-counts num" aria-label="색별 개수">
          <span className="rc-total">
            <strong>{cubes!.length}</strong> 개
          </span>
          {CUBE_COLORS.map((c) =>
            counts[c] > 0 || c !== 'unknown' ? (
              <span key={c} className={`rc${counts[c] === 0 ? ' zero' : ''}`}>
                <ClassSwatch color={c} /> {counts[c]}
              </span>
            ) : null,
          )}
        </div>
      ) : null}

      {timing && !showGt ? (
        <div className="result-time num">
          {timing.run ? (
            <>
              <span>
                p50 <strong>{formatMs(timing.run.p50_ms)}</strong>
              </span>
              <span>p95 {formatMs(timing.run.p95_ms)}</span>
              <span>첫 프레임 {formatMs(timing.run.first_ms)}</span>
              {timing.ms !== null ? <span>이 프레임 {formatMs(timing.ms)}</span> : null}
            </>
          ) : (
            <>
              <span>
                처리 <strong>{formatMs(timing.ms)}</strong>
              </span>
              {timing.rtt !== null ? <span>왕복 {formatMs(timing.rtt)}</span> : null}
            </>
          )}
          {timing.stages && Object.keys(timing.stages).length > 0 ? (
            <span className="result-stages">
              {Object.entries(timing.stages).map(([k, v]) => (
                <span key={k}>
                  {k} {v.toFixed(1)}
                </span>
              ))}
            </span>
          ) : null}
        </div>
      ) : null}

      {notes.length > 0 ? (
        <ul className="result-notes">
          {notes.map((n, i) => (
            <li key={i}>
              <AlertTriangle size={12} strokeWidth={1.75} />
              {n}
            </li>
          ))}
        </ul>
      ) : null}

      <div className="result-table-wrap">
        {showGt ? (
          <GtTable d={d} highlight={highlight} onHighlight={onHighlight} />
        ) : cubes && cubes.length > 0 ? (
          <table className="table result-table" onMouseLeave={() => onHighlight(null)}>
            <thead>
              <tr>
                <th className="r">#</th>
                <th>색</th>
                <th className="r">중심 x · y</th>
                <th className="r">각도</th>
                <th className="r">점수</th>
                {d ? <th className="c">정답</th> : null}
              </tr>
            </thead>
            <tbody>
              {cubes.map((c, i) => (
                <tr
                  key={i}
                  ref={(el) => {
                    rowRefs.current[i] = el
                  }}
                  className={i === highlight ? 'hl' : undefined}
                  onMouseEnter={() => onHighlight(i)}
                  onClick={() => onHighlight(i === highlight ? null : i)}
                >
                  <td className="r faint num">{i + 1}</td>
                  <td>
                    <ClassSwatch color={c.color} withName />
                  </td>
                  <td className="r num">
                    {c.center[0].toFixed(1)} · {c.center[1].toFixed(1)}
                  </td>
                  <td className="r num">{c.angle === null ? '—' : `${c.angle.toFixed(1)}°`}</td>
                  <td className="r num">{c.score.toFixed(2)}</td>
                  {d ? (
                    <td className="c">
                      <Outcome d={d} pred={i} />
                    </td>
                  ) : null}
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="result-empty faint">{cubes ? '이 프레임에서 찾은 큐브가 없습니다.' : empty}</p>
        )}
      </div>
    </section>
  )
})
