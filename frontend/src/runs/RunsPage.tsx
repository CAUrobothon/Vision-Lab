// 기록 (History) — `/runs?media&model&status&starred&tag&q`. Filters live in the URL.
import { memo, useMemo, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Link, useNavigate, useSearchParams } from 'react-router'
import { Download, ExternalLink, GitCompareArrows, RefreshCw, Search, X } from 'lucide-react'
import { api } from '../api/client'
import { invalidateRuns, isActive, useMediaList, useModels, useRuns } from '../api/queries'
import type { ModelInfo, Run, RunStatus } from '../api/types'
import { formatDate, formatMs, formatNumber } from '../lib/format'
import { changedParams, formatParamValue } from '../lib/params'
import { ConfirmButton, InlineText, STATUS_OPTIONS, StarButton, StatusMark, TagEditor } from '../ui/controls'
import { useToast } from '../ui/toast'
import './runs.css'

function numOrUndef(v: string | null): number | undefined {
  const n = v === null ? NaN : Number(v)
  return Number.isFinite(n) && n > 0 ? n : undefined
}

function ChangedParams({ run, model }: { run: Run; model: ModelInfo | undefined }) {
  if (!model) return <span className="faint">모델이 지워져 기본값과 비교할 수 없음</span>
  const names = changedParams(model.params, run.params)
  if (names.length === 0) return <span className="faint">기본값</span>
  const parts = names.map((n) => {
    const spec = model.params.find((s) => s.name === n)
    return `${spec?.label ?? n} ${formatParamValue(spec, run.params[n])}`
  })
  return (
    <span className="changed-params" title={parts.join('\n')}>
      {parts.map((p, i) => (
        <span key={i} className="cp">
          {p}
        </span>
      ))}
    </span>
  )
}

const RunRow = memo(function RunRow({
  run,
  model,
  checked,
  onCheck,
}: {
  run: Run
  model: ModelInfo | undefined
  checked: boolean
  onCheck: (id: number, on: boolean) => void
}) {
  const toast = useToast()
  const navigate = useNavigate()
  const patch = useMutation({
    mutationFn: (p: { note?: string; tags?: string[]; starred?: boolean }) => api.patchRun(run.id, p),
    onSuccess: invalidateRuns,
    onError: (e: Error) => toast.error(e.message),
  })
  const remove = useMutation({
    mutationFn: () => api.deleteRun(run.id),
    onSuccess: invalidateRuns,
    onError: (e: Error) => toast.error(e.message),
  })
  const exportRun = useMutation({
    mutationFn: () => api.exportRun(run.id),
    onSuccess: (info) => {
      invalidateRuns()
      window.location.assign(info.url)
    },
    onError: (e: Error) => toast.error(e.message),
  })
  const open = () => navigate(`/?media=${run.media_id}&run=${run.id}`)
  const active = isActive(run)
  return (
    <li className={`run-row${checked ? ' checked' : ''}`}>
      <label className="rr-check">
        <input type="checkbox" checked={checked} onChange={(e) => onCheck(run.id, e.target.checked)} aria-label={`기록 #${run.id} 선택`} />
      </label>
      <button type="button" className="rr-thumb" onClick={open} aria-label={`기록 #${run.id}을 작업대에서 열기`}>
        {run.thumb_url ? <img src={run.thumb_url} alt="" loading="lazy" /> : <span className="faint">미리보기 없음</span>}
      </button>
      <div className="rr-id">
        <span className="num">#{run.id}</span>
        <span className="faint num">{formatDate(run.created_at)}</span>
        {run.parent_run_id ? <span className="faint num">← #{run.parent_run_id}</span> : null}
      </div>
      <div className="rr-media">
        <span className="rr-strong" title={run.media_name}>
          {run.media_name}
        </span>
        <span className="faint num">
          {run.media_kind === 'video'
            ? `f ${run.frame_start}–${run.frame_end === null ? '끝' : run.frame_end - 1}${run.stride > 1 ? ` · ${run.stride}프레임마다` : ''}`
            : '이미지'}
        </span>
      </div>
      <div className="rr-model">
        <span className="rr-strong">{run.model_name}</span>
        <ChangedParams run={run} model={model} />
      </div>
      <div className="rr-num num">
        <span className="rr-strong">{formatNumber(run.summary?.cubes_mean)}</span>
        <span className="faint">{run.summary ? `${run.summary.cubes_min}–${run.summary.cubes_max}` : ''}</span>
      </div>
      <div className="rr-num num">
        <span className="rr-strong">{formatMs(run.timing?.p50_ms)}</span>
        <span className="faint">{run.device ?? ''}</span>
      </div>
      <div className="rr-status">
        <StatusMark status={run.status} />
        {active ? (
          <div className="progress">
            <span style={{ width: `${run.frames_total ? (run.frames_done / run.frames_total) * 100 : 0}%` }} />
          </div>
        ) : run.error ? (
          <span className="error-text rr-error" title={run.error}>
            {run.error}
          </span>
        ) : null}
      </div>
      <div className="rr-notes">
        <InlineText value={run.note} placeholder="메모 추가" label={`기록 #${run.id} 메모`} onSave={(note) => patch.mutate({ note })} />
        <TagEditor tags={run.tags} onChange={(tags) => patch.mutate({ tags })} />
      </div>
      <div className="rr-actions">
        <StarButton on={run.starred} label={`기록 #${run.id}`} onToggle={() => patch.mutate({ starred: !run.starred })} />
        <button type="button" className="icon-btn" onClick={open} aria-label="작업대에서 열기" title="작업대에서 열기">
          <ExternalLink size={14} strokeWidth={1.75} />
        </button>
        <button
          type="button"
          className="icon-btn"
          onClick={() => navigate(`/?media=${run.media_id}&rerun=${run.id}`)}
          aria-label="파라미터 바꿔 다시 실행"
          title="파라미터 바꿔 다시 실행"
        >
          <RefreshCw size={14} strokeWidth={1.75} />
        </button>
        {run.export_url ? (
          <a className="icon-btn" href={run.export_url} download aria-label="내보낸 파일 받기" title="내보낸 파일 받기">
            <Download size={14} strokeWidth={1.75} />
          </a>
        ) : (
          <button
            type="button"
            className="icon-btn"
            disabled={run.status !== 'done' || exportRun.isPending}
            onClick={() => exportRun.mutate()}
            aria-label="결과를 그린 파일로 내보내기"
            title={run.media_kind === 'video' ? 'MP4로 내보내기' : 'PNG로 내보내기'}
          >
            <Download size={14} strokeWidth={1.75} />
          </button>
        )}
        <ConfirmButton label={`기록 #${run.id} 삭제`} onConfirm={() => remove.mutate()} quiet />
      </div>
    </li>
  )
})

export default function RunsPage() {
  const navigate = useNavigate()
  const toast = useToast()
  const [sp, setSp] = useSearchParams()
  const [limit, setLimit] = useState(100)
  const filter = {
    media_id: numOrUndef(sp.get('media')),
    model_key: sp.get('model') || undefined,
    status: (sp.get('status') as RunStatus | null) || undefined,
    starred: sp.get('starred') === '1' ? true : undefined,
    tag: sp.get('tag') || undefined,
    q: sp.get('q')?.trim() || undefined,
    limit,
  }
  const runs = useRuns(filter)
  const media = useMediaList({})
  const models = useModels()
  const modelMap = useMemo(() => new Map((models.data ?? []).map((m) => [m.key, m])), [models.data])
  const [picked, setSelected] = useState<number[]>([])
  const [q, setQ] = useState(sp.get('q') ?? '')
  const [tagDraft, setTagDraft] = useState(sp.get('tag') ?? '')

  const set = (key: string, value: string | null) =>
    setSp(
      (prev) => {
        const next = new URLSearchParams(prev)
        if (value === null || value === '') next.delete(key)
        else next.set(key, value)
        return next
      },
      { replace: true },
    )

  const items = runs.data?.items ?? []
  // Only rows the filter shows count as selected: a bulk delete must never reach a run the user
  // cannot see (found by exploratory QA).
  const selected = picked.filter((id) => items.some((r) => r.id === id))

  const onCheck = (id: number, on: boolean) => setSelected((prev) => (on ? [...prev, id] : prev.filter((x) => x !== id)))
  const bulkDelete = useMutation({
    mutationFn: async (ids: number[]) => {
      for (const id of ids) await api.deleteRun(id)
    },
    onSuccess: (_, ids) => {
      toast.ok(`기록 ${ids.length}개를 지웠습니다`)
      setSelected([])
      invalidateRuns()
    },
    onError: (e: Error) => {
      toast.error(e.message)
      invalidateRuns()
    },
  })

  const filtered = !!(filter.media_id || filter.model_key || filter.status || filter.starred || filter.tag || filter.q)
  const allVisibleSelected = items.length > 0 && items.every((r) => selected.includes(r.id))

  return (
    <div className="page">
      <div className="page-inner runs-page">
        <div className="page-head">
          <h1>기록</h1>
          <span className="faint num">{runs.data ? `${runs.data.total}개` : ''}</span>
        </div>

        <div className="filter-bar" role="search">
          <select className="select" aria-label="미디어" value={filter.media_id ?? ''} onChange={(e) => set('media', e.target.value)}>
            <option value="">모든 미디어</option>
            {media.data?.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
          <select className="select" aria-label="모델" value={filter.model_key ?? ''} onChange={(e) => set('model', e.target.value)}>
            <option value="">모든 모델</option>
            {models.data?.map((m) => (
              <option key={m.key} value={m.key}>
                {m.name}
              </option>
            ))}
          </select>
          <select className="select" aria-label="상태" value={filter.status ?? ''} onChange={(e) => set('status', e.target.value)}>
            <option value="">모든 상태</option>
            {STATUS_OPTIONS.map(([v, label]) => (
              <option key={v} value={v}>
                {label}
              </option>
            ))}
          </select>
          <button type="button" className="chip" aria-pressed={!!filter.starred} onClick={() => set('starred', filter.starred ? null : '1')}>
            별표만
          </button>
          <input
            className="input tag-filter"
            value={tagDraft}
            placeholder="태그"
            aria-label="태그로 거르기"
            onChange={(e) => setTagDraft(e.target.value)}
            onBlur={() => set('tag', tagDraft.trim())}
            onKeyDown={(e) => e.key === 'Enter' && set('tag', tagDraft.trim())}
          />
          <label className="search grow">
            <Search size={13} strokeWidth={1.75} aria-hidden="true" />
            <input
              className="input"
              type="search"
              value={q}
              placeholder="#번호, 미디어, 모델, 메모로 찾기"
              aria-label="기록 검색"
              onChange={(e) => {
                setQ(e.target.value)
                set('q', e.target.value)
              }}
            />
          </label>
          {filtered ? (
            <button
              type="button"
              className="btn btn-sm btn-quiet"
              onClick={() => {
                setQ('')
                setTagDraft('')
                setSp({}, { replace: true })
              }}
            >
              <X size={12} strokeWidth={1.75} /> 필터 지우기
            </button>
          ) : null}
        </div>

        <div className={`bulk-bar${selected.length ? ' on' : ''}`} aria-live="polite">
          <label className="rr-check">
            <input
              type="checkbox"
              checked={allVisibleSelected}
              aria-label="보이는 기록 모두 선택"
              onChange={(e) => setSelected(e.target.checked ? items.map((r) => r.id) : [])}
            />
          </label>
          {selected.length ? (
            <>
              <span className="num">{selected.length}개 선택</span>
              <button
                type="button"
                className="btn btn-sm"
                disabled={selected.length !== 2}
                onClick={() => navigate(`/compare?a=${selected[0]}&b=${selected[1]}`)}
                title={selected.length !== 2 ? '두 개를 고르면 비교할 수 있습니다' : undefined}
              >
                <GitCompareArrows size={13} strokeWidth={1.75} /> 비교
              </button>
              <span className="bulk-gap" />
              <ConfirmButton label={`${selected.length}개 삭제`} confirmLabel={`정말 ${selected.length}개 삭제`} onConfirm={() => bulkDelete.mutate(selected)} />
              <button type="button" className="btn btn-sm btn-quiet" onClick={() => setSelected([])}>
                선택 해제
              </button>
            </>
          ) : (
            <span className="faint">기록을 골라 비교하거나 한꺼번에 지울 수 있습니다.</span>
          )}
        </div>

        <div className="run-head" aria-hidden="true">
          <span />
          <span />
          <span>번호 · 시각</span>
          <span>미디어</span>
          <span>모델 · 바꾼 파라미터</span>
          <span className="r">큐브/프레임</span>
          <span className="r">p50</span>
          <span>상태</span>
          <span>메모 · 태그</span>
          <span />
        </div>

        {runs.isPending ? (
          <ul className="run-list" aria-hidden="true">
            {[0, 1, 2].map((i) => (
              <li key={i} className="run-row">
                <span />
                <span className="rr-thumb skeleton" />
              </li>
            ))}
          </ul>
        ) : runs.isError ? (
          <p className="error-text empty">{runs.error.message}</p>
        ) : items.length === 0 ? (
          <div className="empty">
            {filtered ? (
              <>
                <strong>조건에 맞는 기록이 없습니다</strong>
                <span>필터를 지우면 모든 기록이 보입니다.</span>
              </>
            ) : (
              <>
                <strong>아직 기록이 없습니다</strong>
                <span>
                  <Link to="/">작업대</Link>에서 미리보기를 “기록으로 저장”하거나 모델을 “실행”하면 여기에 쌓입니다.
                </span>
              </>
            )}
          </div>
        ) : (
          <ul className="run-list">
            {items.map((r) => (
              <RunRow key={r.id} run={r} model={modelMap.get(r.model_key)} checked={selected.includes(r.id)} onCheck={onCheck} />
            ))}
          </ul>
        )}
        {runs.data && runs.data.total > limit ? (
          <div className="more">
            <button type="button" className="btn" onClick={() => setLimit((l) => l + 100)}>
              더 보기 ({runs.data.total - limit}개 남음)
            </button>
          </div>
        ) : null}
      </div>
    </div>
  )
}
