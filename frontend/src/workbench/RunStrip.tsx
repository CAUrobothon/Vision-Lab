// Strip of this media's runs under the viewer: click shows a run in the viewer, check two to compare.
import { memo } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Link, useNavigate } from 'react-router'
import { GitCompareArrows, Square } from 'lucide-react'
import { api } from '../api/client'
import { invalidateRuns, isActive, useRuns } from '../api/queries'
import type { Run } from '../api/types'
import { formatMs } from '../lib/format'
import { StarButton, StatusMark } from '../ui/controls'
import { useToast } from '../ui/toast'

interface RunStripProps {
  mediaId: number
  mediaThumb: string
  selectedId: number | null
  onSelect: (id: number | null) => void
  checked: number[]
  onCheck: (ids: number[]) => void
}

const RunTile = memo(function RunTile({
  run,
  fallbackThumb,
  selected,
  checked,
  onSelect,
  onCheck,
}: {
  run: Run
  fallbackThumb: string
  selected: boolean
  checked: boolean
  onSelect: (id: number | null) => void
  onCheck: (id: number, on: boolean) => void
}) {
  const toast = useToast()
  const star = useMutation({
    mutationFn: () => api.patchRun(run.id, { starred: !run.starred }),
    onSuccess: invalidateRuns,
    onError: (e: Error) => toast.error(e.message),
  })
  const cancel = useMutation({
    mutationFn: () => api.cancelRun(run.id),
    onSuccess: invalidateRuns,
    onError: (e: Error) => toast.error(e.message),
  })
  const pct = run.frames_total ? Math.round((run.frames_done / run.frames_total) * 100) : 0
  return (
    <li className={`run-tile${selected ? ' selected reg' : ''}`}>
      <label className="run-check" title="비교할 기록으로 선택">
        <input type="checkbox" checked={checked} onChange={(e) => onCheck(run.id, e.target.checked)} aria-label={`기록 #${run.id} 비교 선택`} />
      </label>
      <button
        type="button"
        className="run-main"
        aria-pressed={selected}
        onClick={() => onSelect(selected ? null : run.id)}
        title={`#${run.id} ${run.model_name} — ${selected ? '미리보기로 돌아가기' : '뷰어에 표시'}`}
      >
        <span className="run-thumb">
          <img src={run.thumb_url ?? fallbackThumb} alt="" loading="lazy" />
        </span>
        <span className="run-text">
          <span className="run-title">
            <span className="num faint">#{run.id}</span> {run.model_name}
          </span>
          <StatusMark status={run.status} />
          <span className="run-meta num">
            {isActive(run) ? `${run.frames_done} / ${run.frames_total ?? '?'}` : `p50 ${formatMs(run.timing?.p50_ms)}`}
            {run.media_kind === 'video' && run.stride > 1 && !isActive(run) ? ` · ${run.stride}프레임마다` : ''}
          </span>
        </span>
      </button>
      <div className="run-side">
        <StarButton on={run.starred} label={`기록 #${run.id}`} onToggle={() => star.mutate()} />
        {isActive(run) ? (
          <button type="button" className="icon-btn sm" aria-label={`기록 #${run.id} 취소`} title="취소" onClick={() => cancel.mutate()}>
            <Square size={11} strokeWidth={2} />
          </button>
        ) : null}
      </div>
      {isActive(run) ? (
        <div className="progress run-progress" aria-label={`진행 ${pct}%`}>
          <span style={{ width: `${pct}%` }} />
        </div>
      ) : null}
    </li>
  )
})

export function RunStrip({ mediaId, mediaThumb, selectedId, onSelect, checked, onCheck }: RunStripProps) {
  const navigate = useNavigate()
  const runs = useRuns({ media_id: mediaId, limit: 50 })
  const items = runs.data?.items ?? []
  const toggle = (id: number, on: boolean) => {
    const next = on ? [...checked.filter((x) => x !== id), id].slice(-2) : checked.filter((x) => x !== id)
    onCheck(next)
  }
  return (
    <div className="strip">
      <div className="strip-head">
        <h2 className="panel-title">이 미디어의 기록</h2>
        <span className="faint num">{runs.data ? `${runs.data.total}개` : ''}</span>
        <span className="strip-spacer" />
        {checked.length > 0 ? <span className="faint">비교할 기록 {checked.length}/2</span> : null}
        <button
          type="button"
          className="btn btn-sm"
          disabled={checked.length !== 2}
          onClick={() => navigate(`/compare?a=${checked[0]}&b=${checked[1]}`)}
        >
          <GitCompareArrows size={13} strokeWidth={1.75} /> 비교
        </button>
        <Link className="btn btn-sm btn-quiet" to={`/runs?media=${mediaId}`}>
          기록에서 보기
        </Link>
      </div>
      {items.length === 0 ? (
        <p className="strip-empty faint">
          {runs.isPending ? '불러오는 중…' : '아직 기록이 없습니다. 오른쪽에서 “기록으로 저장” 또는 “실행”을 누르면 여기에 쌓입니다.'}
        </p>
      ) : (
        <ul className="run-tiles">
          {items.map((r) => (
            <RunTile
              key={r.id}
              run={r}
              fallbackThumb={mediaThumb}
              selected={r.id === selectedId}
              checked={checked.includes(r.id)}
              onSelect={onSelect}
              onCheck={toggle}
            />
          ))}
        </ul>
      )}
    </div>
  )
}
