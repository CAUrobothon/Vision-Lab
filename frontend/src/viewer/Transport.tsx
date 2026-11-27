// Video transport: play/pause, frame step, frame number, time and a scrubber that marks the frames
// a run has results for.
import { memo, useMemo } from 'react'
import { ChevronLeft, ChevronRight, Pause, Play } from 'lucide-react'
import { formatDuration } from '../lib/format'

interface TransportProps {
  frame: number
  frameCount: number
  fps: number
  paused: boolean
  marks: number[] | null
  onToggle: () => void
  onStep: (delta: number) => void
  onSeek: (frame: number) => void
}

const Marks = memo(function Marks({ marks, frameCount }: { marks: number[]; frameCount: number }) {
  const d = useMemo(() => marks.map((f) => `M${f + 0.5} 0V1`).join(''), [marks])
  return (
    <svg className="scrub-marks" viewBox={`0 0 ${Math.max(frameCount, 1)} 1`} preserveAspectRatio="none" aria-hidden="true">
      <path d={d} />
    </svg>
  )
})

export function Transport({ frame, frameCount, fps, paused, marks, onToggle, onStep, onSeek }: TransportProps) {
  const last = Math.max(frameCount - 1, 0)
  const pct = last > 0 ? (frame / last) * 100 : 0
  return (
    <div className="transport">
      <button type="button" className="icon-btn" onClick={onToggle} aria-label={paused ? '재생' : '일시정지'} title={paused ? '재생 (Space)' : '일시정지 (Space)'}>
        {paused ? <Play size={16} strokeWidth={1.75} /> : <Pause size={16} strokeWidth={1.75} />}
      </button>
      <button type="button" className="icon-btn" onClick={() => onStep(-1)} aria-label="이전 프레임" title="이전 프레임 (←, Shift 10프레임)">
        <ChevronLeft size={16} strokeWidth={1.75} />
      </button>
      <button type="button" className="icon-btn" onClick={() => onStep(1)} aria-label="다음 프레임" title="다음 프레임 (→, Shift 10프레임)">
        <ChevronRight size={16} strokeWidth={1.75} />
      </button>
      <span className="transport-frame num" aria-live="off">
        f {frame} <span className="faint">/ {last}</span>
      </span>
      <div className="scrub">
        {marks && marks.length > 0 ? <Marks marks={marks} frameCount={frameCount} /> : null}
        <input
          type="range"
          className="slider"
          min={0}
          max={last}
          step={1}
          value={frame}
          style={{ ['--pct' as string]: `${pct}%` }}
          onChange={(e) => onSeek(Number(e.target.value))}
          aria-label="재생 위치 (프레임)"
        />
      </div>
      <span className="transport-time num faint">
        {formatDuration(frame / fps)} / {formatDuration(frameCount / fps)}
      </span>
    </div>
  )
}
