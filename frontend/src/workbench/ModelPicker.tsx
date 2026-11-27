// Model picker grouped "내장 파이프라인" / "내 가중치", each with its summary.
import { useEffect, useRef, useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { Link } from 'react-router'
import type { ModelInfo } from '../api/types'

const TASK_LABEL: Record<string, string> = { detect: '검출', segment: '분할', obb: '회전 박스' }

export function modelKindLabel(m: ModelInfo): string {
  return m.kind === 'pipeline' ? 'OpenCV' : `가중치 · ${TASK_LABEL[m.task ?? ''] ?? '?'}`
}

export function ModelPicker({ models, value, onChange }: { models: ModelInfo[]; value: string | null; onChange: (key: string) => void }) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const current = models.find((m) => m.key === value) ?? null
  const groups: [string, ModelInfo[]][] = [
    ['내장 파이프라인', models.filter((m) => m.kind === 'pipeline')],
    ['내 가중치', models.filter((m) => m.kind === 'weights')],
  ]

  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    document.addEventListener('pointerdown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('pointerdown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <div className="model-picker" ref={rootRef}>
      <button type="button" className="model-current" aria-expanded={open} aria-haspopup="listbox" onClick={() => setOpen((v) => !v)}>
        <span className="mc-top">
          <strong>{current?.name ?? '모델 선택'}</strong>
          <span className="mc-kind">{current ? modelKindLabel(current) : ''}</span>
          <ChevronDown size={14} strokeWidth={1.75} className="mc-chevron" />
        </span>
        {current ? <span className="mc-summary">{current.summary}</span> : null}
      </button>
      {open ? (
        <div className="model-list" role="listbox" aria-label="모델">
          {groups.map(([title, items]) => (
            <div key={title} role="group" aria-label={title}>
              <div className="ml-group">{title}</div>
              {items.length === 0 ? (
                <p className="ml-empty">
                  아직 없습니다. <Link to="/models">모델 · 저장공간</Link>에서 <code>.pt</code>를 올리세요.
                </p>
              ) : (
                items.map((m) => (
                  <button
                    key={m.key}
                    type="button"
                    role="option"
                    aria-selected={m.key === value}
                    className={`ml-item${m.key === value ? ' reg' : ''}`}
                    disabled={m.state === 'error'}
                    onClick={() => {
                      onChange(m.key)
                      setOpen(false)
                    }}
                  >
                    <span className="ml-name">
                      {m.name}
                      <span className="mc-kind">{modelKindLabel(m)}</span>
                    </span>
                    <span className="ml-summary">{m.state === 'error' ? m.error : m.summary}</span>
                  </button>
                ))
              )}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
}
