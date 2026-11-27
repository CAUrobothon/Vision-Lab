// One batch image by image (GET /api/score/images): 원본별 결과 (rows = originals, columns =
// variants, cell = image pass with missed / false-positive counts) and 이미지별 결과 (failures
// first). Every image opens its run in the workbench with the 정답 layer on.
import { useMemo } from 'react'
import { useNavigate } from 'react-router'
import { Check, ExternalLink, X } from 'lucide-react'
import type { ScoreImageRow } from '../api/types'
import { formatMs } from '../lib/format'
import { DASH, int, SCORE_STATUS_LABEL } from './format'

/** Opens a run in the workbench with the 정답 layer on and the rail on its dataset (the workbench
 *  reads `dataset` and `gt` from its URL). */
export function useOpenInViewer() {
  const navigate = useNavigate()
  return (row: Pick<ScoreImageRow, 'media_id' | 'run_id'>, datasetId: number | null) => {
    const q = new URLSearchParams({ media: String(row.media_id) })
    if (row.run_id !== null) q.set('run', String(row.run_id))
    if (datasetId !== null) q.set('dataset', String(datasetId))
    q.set('gt', '1')
    navigate(`/?${q}`)
  }
}

/** An image's failure counts: the row's field, its letter in a grid cell and its column name. */
const COUNTS = [
  ['n_missed', '미', '미검출'],
  ['n_fp', '오', '오검출'],
  ['n_color_wrong', '색', '색 틀림'],
  ['n_geom_fail', '기', '기하 실패'],
] as const

function badness(r: ScoreImageRow): number {
  return COUNTS.reduce((sum, [k]) => sum + (r[k] ?? 0), 0)
}

function Counts({ r }: { r: ScoreImageRow }) {
  const parts = COUNTS.filter(([k]) => r[k]).map(([k, letter]) => `${letter}${r[k]}`)
  return parts.length ? <span className="ps-counts num">{parts.join(' ')}</span> : null
}

function PassMark({ passed }: { passed: boolean | null }) {
  if (passed === null) return null
  return passed ? (
    <Check size={13} strokeWidth={2.5} className="pass-icon ok-text" aria-label="통과" />
  ) : (
    <X size={13} strokeWidth={2.5} className="pass-icon fail-text" aria-label="미달" />
  )
}

function rowTitle(r: ScoreImageRow): string {
  if (r.passed === null) return `${r.filename}: ${SCORE_STATUS_LABEL[r.status]} (아직 채점 안 됨)`
  return `${r.filename}: 이미지 ${r.passed ? '통과' : '미달'} · ${COUNTS.map(([k, , name]) => `${name} ${r[k]}`).join(' · ')}`
}

export function SourceGrid({ rows, variants, onOpen }: { rows: ScoreImageRow[]; variants: string[]; onOpen: (r: ScoreImageRow) => void }) {
  const sources = useMemo(() => {
    const by = new Map<string, Map<string, ScoreImageRow>>()
    for (const r of rows) {
      const m = by.get(r.source_sha256) ?? new Map<string, ScoreImageRow>()
      m.set(r.variant, r)
      by.set(r.source_sha256, m)
    }
    return [...by.entries()].map(([sha, m]) => ({ sha, cells: m, label: (m.get('original') ?? [...m.values()][0]).filename, hasOriginal: m.has('original') }))
  }, [rows])
  const cols = variants.filter((v) => rows.some((r) => r.variant === v))
  return (
    <div className="sc-table-wrap fit ps-wrap">
      <table className="table sc-table ps-table">
        <thead>
          <tr>
            <th className="sc-sticky ps-src-h">원본</th>
            {cols.map((v) => (
              <th key={v} className="ps-vh" title={v}>
                {v}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sources.map((s) => (
            <tr key={s.sha}>
              <th scope="row" className="sc-sticky ps-src" title={s.label}>
                {s.label}
                {s.hasOriginal ? null : <span className="faint"> (원본 없음)</span>}
              </th>
              {cols.map((v) => {
                const r = s.cells.get(v)
                if (!r)
                  return (
                    <td key={v} data-variant={v} className="ps-cell ps-none">
                      {DASH}
                    </td>
                  )
                return (
                  <td key={v} data-variant={v} data-pass={r.passed === null ? '' : String(r.passed)} className={`ps-cell${r.passed === false ? ' fail' : ''}`}>
                    <button type="button" className="ps-open" title={`${rowTitle(r)} — 작업대에서 정답과 함께 보기`} onClick={() => onOpen(r)}>
                      {r.passed === null ? <span className="faint">{SCORE_STATUS_LABEL[r.status]}</span> : <PassMark passed={r.passed} />}
                      <Counts r={r} />
                    </button>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function ImageList({ rows, onOpen }: { rows: ScoreImageRow[]; onOpen: (r: ScoreImageRow) => void }) {
  const sorted = useMemo(
    () =>
      [...rows].sort(
        (a, b) =>
          badness(b) - badness(a) ||
          Number(a.passed !== false) - Number(b.passed !== false) ||
          Number(a.passed === null) - Number(b.passed === null) ||
          a.gt_index - b.gt_index,
      ),
    [rows],
  )
  return (
    <div className="sc-table-wrap im-wrap">
      <table className="table sc-table im-table">
        <thead>
          <tr>
            <th>이미지</th>
            <th>증강</th>
            <th>결과</th>
            <th className="r">정답</th>
            {COUNTS.map(([k, , name]) => (
              <th key={k} className="r">
                {name}
              </th>
            ))}
            <th className="r">처리 시간</th>
            <th aria-label="열기" />
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.gt_index} className={r.passed === false ? 'fail' : undefined} onClick={() => onOpen(r)} title={rowTitle(r)}>
              <td className="im-name-c">
                {r.folder ? <span className="media-folder">{r.folder}/</span> : null}
                <span className="im-name">{r.filename}</span>
                {r.media_name !== r.filename ? <span className="faint"> ← {r.media_name}</span> : null}
              </td>
              <td className="faint">{r.variant}</td>
              <td>
                {r.passed === null ? (
                  <span className="faint">{SCORE_STATUS_LABEL[r.status]}</span>
                ) : (
                  <span className={`im-pass ${r.passed ? 'ok' : 'fail'}`}>
                    <PassMark passed={r.passed} />
                    {r.passed ? '통과' : '미달'}
                  </span>
                )}
              </td>
              <td className="r num">{int(r.n_gt)}</td>
              {COUNTS.map(([k]) => (
                <td key={k} className={`r num${r[k] ? ' im-bad' : ''}`}>
                  {r[k] === null ? DASH : int(r[k]!)}
                </td>
              ))}
              <td className="r num vol">{formatMs(r.ms)}</td>
              <td className="r">
                <button
                  type="button"
                  className="icon-btn sm"
                  aria-label={`${r.filename} 작업대에서 정답과 함께 보기`}
                  title="작업대에서 정답과 함께 보기"
                  onClick={(e) => {
                    e.stopPropagation()
                    onOpen(r)
                  }}
                >
                  <ExternalLink size={12} strokeWidth={1.75} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
