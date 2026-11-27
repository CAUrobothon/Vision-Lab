// Actions that create runs: save the preview as a run (pipelines), run a weights model, and for
// videos a frame range with a stride.
import { useState } from 'react'
import { Play, Save, X } from 'lucide-react'
import type { Media, ModelInfo } from '../api/types'
import { NumberField } from '../ui/controls'

export interface RunRequest {
  frame_start: number
  frame_end: number | null
  stride: number
}

interface RunActionsProps {
  media: Media | null
  model: ModelInfo | null
  frame: number
  pending: boolean
  parentRunId: number | null
  onClearParent: () => void
  onCreate: (req: RunRequest) => void
}

export function RunActions({ media, model, frame, pending, parentRunId, onClearParent, onCreate }: RunActionsProps) {
  const isVideo = media?.kind === 'video'
  const isPipeline = model?.kind === 'pipeline'
  const total = media?.frame_count ?? 1
  const [scope, setScope] = useState<'all' | 'range'>('all')
  const [start, setStart] = useState(0)
  const [end, setEnd] = useState(total)
  const [stride, setStride] = useState(isPipeline ? 1 : 3)
  const disabled = !media || !model || model.state !== 'ready' || pending

  const s = scope === 'all' ? 0 : Math.min(start, total - 1)
  const e = scope === 'all' ? total : Math.min(end, total)
  // An end at or before the start is shown as a mistake, not quietly turned into one frame.
  const badRange = scope === 'range' && e <= s
  const count = badRange ? 0 : Math.ceil((e - s) / stride)

  const single = () => onCreate({ frame_start: isVideo ? frame : 0, frame_end: isVideo ? frame + 1 : null, stride: 1 })

  return (
    <div className="actions">
      {parentRunId !== null ? (
        <div className="parent-note">
          <span>
            기록 <span className="num">#{parentRunId}</span>의 파라미터를 바꿔 다시 실행
          </span>
          <button type="button" className="icon-btn sm" aria-label="원본 기록 연결 해제" onClick={onClearParent}>
            <X size={12} strokeWidth={1.75} />
          </button>
        </div>
      ) : null}

      {isVideo ? (
        <div className="video-run">
          <div className="vr-row">
            <span className="vr-label">범위</span>
            <div className="segmented" role="group" aria-label="실행 범위">
              <button type="button" aria-pressed={scope === 'all'} onClick={() => setScope('all')}>
                전체
              </button>
              <button
                type="button"
                aria-pressed={scope === 'range'}
                onClick={() => {
                  if (scope !== 'range') {
                    setStart(frame)
                    setEnd(total)
                  }
                  setScope('range')
                }}
              >
                구간
              </button>
            </div>
            <label className="vr-stride">
              <NumberField integer label="프레임 간격" value={stride} min={1} max={1000} onCommit={setStride} />
              <span>프레임마다</span>
            </label>
          </div>
          {scope === 'range' ? (
            <div className="vr-row">
              <span className="vr-label">구간</span>
              <span className="vr-range num">
                <NumberField integer label="시작 프레임 (포함)" value={start} min={0} max={total - 1} onCommit={setStart} />
                <button type="button" className="btn btn-sm btn-quiet" onClick={() => setStart(frame)} title="현재 프레임을 시작으로">
                  현재
                </button>
                <span className="faint">→</span>
                <NumberField integer label="끝 프레임 (제외)" value={end} min={1} max={total} onCommit={setEnd} />
                <button type="button" className="btn btn-sm btn-quiet" onClick={() => setEnd(frame + 1)} title="현재 프레임까지">
                  현재
                </button>
              </span>
            </div>
          ) : null}
          {badRange ? (
            <p className="vr-summary error-text">끝 프레임({e})이 시작({s})보다 커야 합니다. 끝은 그 프레임을 빼고 셉니다.</p>
          ) : (
            <p className="vr-summary faint num">
              f {s} → {e - 1} 중 {count}프레임 처리{!isPipeline ? ' · 작업자가 순서대로 처리합니다' : ''}
            </p>
          )}
        </div>
      ) : null}

      <div className="action-buttons">
        {isVideo ? (
          <>
            <button type="button" className="btn" disabled={disabled} onClick={single} title={`현재 프레임 f ${frame}만`}>
              {isPipeline ? <Save size={14} strokeWidth={1.75} /> : <Play size={14} strokeWidth={1.75} />}
              {isPipeline ? '이 프레임 저장' : '이 프레임만'}
            </button>
            <button
              type="button"
              className="btn btn-primary"
              disabled={disabled || badRange}
              onClick={() => onCreate({ frame_start: s, frame_end: scope === 'all' ? null : e, stride })}
            >
              <Play size={14} strokeWidth={2} />
              {scope === 'all' ? '영상 전체 실행' : '구간 실행'}
            </button>
          </>
        ) : (
          <button type="button" className="btn btn-primary btn-block" disabled={disabled} onClick={single}>
            {isPipeline ? <Save size={14} strokeWidth={2} /> : <Play size={14} strokeWidth={2} />}
            {isPipeline ? '기록으로 저장' : '실행'}
          </button>
        )}
      </div>
    </div>
  )
}
