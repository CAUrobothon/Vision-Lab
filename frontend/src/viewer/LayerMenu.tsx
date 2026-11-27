// Toggle menu for overlay layers. Rendered inside the viewer so it also works in fullscreen.
import { useEffect, useRef, useState } from 'react'
import { Layers } from 'lucide-react'
import type { DebugLayer } from '../api/types'
import type { LayerPrefs } from './layers'

interface LayerMenuProps {
  prefs: LayerPrefs
  onChange: (patch: Partial<LayerPrefs>) => void
  debugLayers: DebugLayer[] | null // null = this view has no preview (debug unavailable)
  gtAvailable?: boolean // the media has ground truth: offer the 정답 layer
}

const TOGGLES: [keyof LayerPrefs, string][] = [
  ['quad', '윗면 사각형'],
  ['center', '중심 · 각도'],
  ['label', '라벨 (색 + 각도)'],
  ['raw', '원 출력 (폴리곤 · 박스)'],
  ['roi', '검출 영역 (ROI)'],
]

export function LayerMenu({ prefs, onChange, debugLayers, gtAvailable = false }: LayerMenuProps) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('pointerdown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('pointerdown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const activeDebug = debugLayers?.find((l) => l.name === prefs.debugName) ?? debugLayers?.[0] ?? null

  return (
    <div className="layer-menu" ref={rootRef}>
      <button
        type="button"
        className="icon-btn"
        aria-expanded={open}
        aria-label="오버레이 레이어"
        title="오버레이 레이어"
        onClick={() => setOpen((v) => !v)}
      >
        <Layers size={16} strokeWidth={1.75} />
      </button>
      {open ? (
        <div className="layer-panel" role="group" aria-label="오버레이 레이어">
          {TOGGLES.map(([key, label]) => (
            <label key={key} className="layer-row">
              <input type="checkbox" checked={prefs[key] as boolean} onChange={(e) => onChange({ [key]: e.target.checked })} />
              <span>{label}</span>
            </label>
          ))}
          {gtAvailable ? (
            <>
              <div className="layer-sep" />
              <label className="layer-row">
                <input type="checkbox" checked={prefs.gt} onChange={(e) => onChange({ gt: e.target.checked })} />
                <span>정답</span>
              </label>
              <p className="layer-hint">점선 = 정답 윗면, 가는 선 = 짝, 미검출 · 오검출은 글자로 표시. 큐브에 올리면 오차가 나옵니다.</p>
            </>
          ) : null}
          <div className="layer-sep" />
          <label className="layer-row">
            <input
              type="checkbox"
              checked={prefs.debug}
              disabled={debugLayers === null}
              onChange={(e) => onChange({ debug: e.target.checked })}
            />
            <span>디버그 레이어</span>
          </label>
          {debugLayers === null ? (
            <p className="layer-hint">파이프라인 미리보기에서만 볼 수 있습니다.</p>
          ) : prefs.debug ? (
            debugLayers.length === 0 ? (
              <p className="layer-hint">다음 미리보기에서 불러옵니다.</p>
            ) : (
              <div className="layer-debug">
                <select
                  className="select"
                  value={activeDebug?.name ?? ''}
                  onChange={(e) => onChange({ debugName: e.target.value })}
                  aria-label="디버그 레이어 선택"
                >
                  {debugLayers.map((l) => (
                    <option key={l.name} value={l.name}>
                      {l.label}
                    </option>
                  ))}
                </select>
                <label className="layer-opacity">
                  <span className="faint">불투명도</span>
                  <input
                    type="range"
                    className="slider"
                    min={0.1}
                    max={1}
                    step={0.05}
                    value={prefs.debugOpacity}
                    style={{ ['--pct' as string]: `${((prefs.debugOpacity - 0.1) / 0.9) * 100}%` }}
                    onChange={(e) => onChange({ debugOpacity: Number(e.target.value) })}
                  />
                  <span className="num">{Math.round(prefs.debugOpacity * 100)}%</span>
                </label>
                <div className="layer-blend">
                  <span className="faint">합성</span>
                  <div className="segmented" role="group" aria-label="디버그 이미지 합성 방식">
                    <button type="button" aria-pressed={prefs.debugBlend === 'normal'} onClick={() => onChange({ debugBlend: 'normal' })}>
                      보통
                    </button>
                    <button
                      type="button"
                      aria-pressed={prefs.debugBlend === 'screen'}
                      onClick={() => onChange({ debugBlend: 'screen' })}
                      title="검은 부분은 투명하게, 밝은 부분만 겹칩니다"
                    >
                      밝은 부분만
                    </button>
                  </div>
                </div>
              </div>
            )
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
