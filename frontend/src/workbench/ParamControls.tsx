// Parameter controls generated from ParamSpec, grouped by `group` (collapsible), each with label,
// value, unit and the help line. Colour params can be picked from the image; ROI params are drawn
// on the viewer.
import { memo, useState, type ReactNode } from 'react'
import { ChevronDown, Pipette, Plus, RotateCcw, Trash2 } from 'lucide-react'
import type { ParamSpec, ParamValue, Params, RoiPolygon } from '../api/types'
import { groupSpecs, paramDecimals, sameValue } from '../lib/params'
import { loadJson, saveJson } from '../lib/storage'

export type ParamTool = { kind: 'pick'; param: string } | { kind: 'roi'; param: string; mode: 'include' | 'exclude' } | null

interface ParamControlsProps {
  specs: ParamSpec[]
  values: Params
  onChange: (name: string, value: ParamValue) => void
  onResetGroup: (names: string[]) => void
  tool: ParamTool
  onTool: (t: ParamTool) => void
  canUseViewer: boolean // false while no media is selected
}

function stepOf(spec: ParamSpec): number {
  if (spec.step) return spec.step
  if (spec.type === 'int') return 1
  return ((spec.max ?? 1) - (spec.min ?? 0)) / 100
}

function pct(v: number, spec: ParamSpec): string {
  const lo = spec.min ?? 0
  const hi = spec.max ?? 1
  return `${hi > lo ? ((v - lo) / (hi - lo)) * 100 : 0}%`
}

function NumberField({ value, spec, onCommit, label }: { value: number; spec: ParamSpec; onCommit: (v: number) => void; label: string }) {
  const shown = value.toFixed(paramDecimals(spec))
  const [draft, setDraft] = useState<string | null>(null)
  const commit = () => {
    if (draft === null) return
    const v = Number(draft)
    setDraft(null)
    // An emptied field goes back to its value; Number('') is 0, which used to be applied.
    if (draft.trim() === '' || !Number.isFinite(v)) return
    const lo = spec.min ?? -Infinity
    const hi = spec.max ?? Infinity
    onCommit(Math.min(hi, Math.max(lo, spec.type === 'int' ? Math.round(v) : v)))
  }
  return (
    <input
      className="input num-input"
      inputMode="decimal"
      aria-label={label}
      value={draft ?? shown}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') commit()
        if (e.key === 'Escape') setDraft(null)
      }}
    />
  )
}

function NumberControl({ spec, label, value, onChange }: { spec: ParamSpec; label: ReactNode; value: number; onChange: (v: number) => void }) {
  const hasRange = spec.min !== null && spec.max !== null
  return (
    <>
      <div className="pc-head">
        <label className="pc-label" htmlFor={`p-${spec.name}`}>
          {label}
        </label>
        <span className="pc-value">
          <NumberField value={value} spec={spec} onCommit={onChange} label={`${spec.label} 값`} />
          {spec.unit ? <span className="pc-unit">{spec.unit}</span> : null}
        </span>
      </div>
      {hasRange ? (
        <input
          id={`p-${spec.name}`}
          type="range"
          className="slider"
          min={spec.min!}
          max={spec.max!}
          step={stepOf(spec)}
          value={value}
          style={{ ['--pct' as string]: pct(value, spec) }}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      ) : null}
    </>
  )
}

function RangeControl({
  spec,
  label,
  value,
  onChange,
}: {
  spec: ParamSpec
  label: ReactNode
  value: [number, number]
  onChange: (v: [number, number]) => void
}) {
  const [lo, hi] = value
  const set = (a: number, b: number) => onChange(a <= b ? [a, b] : [b, a])
  return (
    <>
      <div className="pc-head">
        <span className="pc-label" id={`p-${spec.name}`}>
          {label}
        </span>
        <span className="pc-value">
          <NumberField value={lo} spec={spec} onCommit={(v) => set(v, hi)} label={`${spec.label} 최소`} />
          <span className="faint">–</span>
          <NumberField value={hi} spec={spec} onCommit={(v) => set(lo, v)} label={`${spec.label} 최대`} />
          {spec.unit ? <span className="pc-unit">{spec.unit}</span> : null}
        </span>
      </div>
      {spec.min !== null && spec.max !== null ? (
        <div className="dual" role="group" aria-labelledby={`p-${spec.name}`}>
          <div className="dual-track">
            <span style={{ left: pct(lo, spec), right: `calc(100% - ${pct(hi, spec)})` }} />
          </div>
          <input
            type="range"
            className="slider"
            min={spec.min}
            max={spec.max}
            step={stepOf(spec)}
            value={lo}
            aria-label={`${spec.label} 최소`}
            onChange={(e) => onChange([Math.min(Number(e.target.value), hi), hi])}
          />
          <input
            type="range"
            className="slider"
            min={spec.min}
            max={spec.max}
            step={stepOf(spec)}
            value={hi}
            aria-label={`${spec.label} 최대`}
            onChange={(e) => onChange([lo, Math.max(Number(e.target.value), lo)])}
          />
        </div>
      ) : null}
    </>
  )
}

function SelectControl({ spec, label, value, onChange }: { spec: ParamSpec; label: ReactNode; value: string; onChange: (v: string) => void }) {
  const options = spec.options ?? []
  const compact = options.length <= 4 && options.reduce((n, o) => n + o.label.length, 0) <= 18
  return (
    <div className="pc-head">
      <span className="pc-label" id={`p-${spec.name}`}>
        {label}
      </span>
      {compact ? (
        <div className="segmented" role="group" aria-labelledby={`p-${spec.name}`}>
          {options.map((o) => (
            <button key={o.value} type="button" aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
              {o.label}
            </button>
          ))}
        </div>
      ) : (
        <select className="select pc-select" value={value} aria-labelledby={`p-${spec.name}`} onChange={(e) => onChange(e.target.value)}>
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      )}
    </div>
  )
}

function MultiSelectControl({
  spec,
  label,
  value,
  onChange,
}: {
  spec: ParamSpec
  label: ReactNode
  value: string[]
  onChange: (v: string[]) => void
}) {
  const options = spec.options ?? []
  const toggle = (v: string) => {
    const next = value.includes(v) ? value.filter((x) => x !== v) : [...value, v]
    onChange(options.map((o) => o.value).filter((x) => next.includes(x)))
  }
  return (
    <>
      <div className="pc-head">
        <span className="pc-label" id={`p-${spec.name}`}>
          {label}
        </span>
        <span className="faint pc-meta">{value.length === 0 ? '전체' : `${value.length}개 고름`}</span>
      </div>
      {options.length > 0 ? (
        <div className="pc-chips" role="group" aria-labelledby={`p-${spec.name}`}>
          {options.map((o) => (
            <button key={o.value} type="button" className="chip" aria-pressed={value.includes(o.value)} onClick={() => toggle(o.value)}>
              {o.label}
            </button>
          ))}
        </div>
      ) : (
        <p className="faint pc-help">고를 항목이 없습니다</p>
      )}
    </>
  )
}

function ColorControl({
  spec,
  label,
  value,
  onChange,
  picking,
  onPick,
  canPick,
}: {
  spec: ParamSpec
  label: ReactNode
  value: string
  onChange: (v: string) => void
  picking: boolean
  onPick: () => void
  canPick: boolean
}) {
  const [draft, setDraft] = useState<string | null>(null)
  const commit = () => {
    if (draft === null) return
    const v = draft.trim().toLowerCase()
    setDraft(null)
    if (/^#?[0-9a-f]{6}$/.test(v)) onChange(v.startsWith('#') ? v : '#' + v)
  }
  return (
    <div className="pc-head">
      <label className="pc-label" htmlFor={`p-${spec.name}`}>
        {label}
      </label>
      <span className="pc-value">
        <input
          id={`p-${spec.name}`}
          type="color"
          className="swatch-input"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          aria-label={`${spec.label} 색 선택`}
        />
        <input
          className="input hex-input"
          value={draft ?? value}
          aria-label={`${spec.label} 16진수`}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => e.key === 'Enter' && commit()}
        />
        <button
          type="button"
          className="icon-btn"
          aria-pressed={picking}
          disabled={!canPick}
          aria-label={`${spec.label}: 이미지에서 고르기`}
          title="이미지에서 고르기"
          onClick={onPick}
        >
          <Pipette size={14} strokeWidth={1.75} />
        </button>
      </span>
    </div>
  )
}

function RoiControl({
  spec,
  label,
  value,
  onChange,
  tool,
  onTool,
  canDraw,
}: {
  spec: ParamSpec
  label: ReactNode
  value: RoiPolygon[]
  onChange: (v: RoiPolygon[]) => void
  tool: ParamTool
  onTool: (t: ParamTool) => void
  canDraw: boolean
}) {
  const drawing = tool?.kind === 'roi' && tool.param === spec.name ? tool.mode : null
  const draw = (mode: 'include' | 'exclude') =>
    onTool(drawing === mode ? null : { kind: 'roi', param: spec.name, mode })
  return (
    <>
      <div className="pc-head">
        {/* The group heading usually already says "검출 영역"; then the summary takes the label's place. */}
        {spec.label !== spec.group ? <span className="pc-label">{label}</span> : null}
        <span className="faint pc-meta">{value.length === 0 ? '영역 없음 · 이미지 전체에서 찾습니다' : `다각형 ${value.length}개`}</span>
      </div>
      {value.length > 0 ? (
        <ul className="roi-list">
          {value.map((p, i) => (
            <li key={i}>
              <span className={`roi-mode ${p.mode}`}>{p.mode === 'include' ? '포함' : '제외'}</span>
              <span className="num faint">점 {p.points.length}개</span>
              <button
                type="button"
                className="icon-btn sm"
                aria-label={`${p.mode === 'include' ? '포함' : '제외'} 영역 ${i + 1} 지우기`}
                onClick={() => onChange(value.filter((_, j) => j !== i))}
              >
                <Trash2 size={12} strokeWidth={1.75} />
              </button>
            </li>
          ))}
        </ul>
      ) : null}
      <div className="roi-actions">
        <button type="button" className="btn btn-sm" aria-pressed={drawing === 'include'} disabled={!canDraw} onClick={() => draw('include')}>
          <Plus size={12} strokeWidth={2} /> 포함 영역 그리기
        </button>
        <button type="button" className="btn btn-sm" aria-pressed={drawing === 'exclude'} disabled={!canDraw} onClick={() => draw('exclude')}>
          <Plus size={12} strokeWidth={2} /> 제외 영역 그리기
        </button>
      </div>
    </>
  )
}

function Control({
  spec,
  value,
  onChange,
  tool,
  onTool,
  canUseViewer,
}: {
  spec: ParamSpec
  value: ParamValue
  onChange: (name: string, v: ParamValue) => void
  tool: ParamTool
  onTool: (t: ParamTool) => void
  canUseViewer: boolean
}) {
  const changed = !sameValue(value, spec.default)
  const set = (v: ParamValue) => onChange(spec.name, v)
  const label = (
    <>
      {spec.label}
      {changed ? (
        <span className="pc-changed" title="기본값과 다릅니다">
          변경
        </span>
      ) : null}
    </>
  )
  let body
  switch (spec.type) {
    case 'int':
    case 'float':
      body = <NumberControl spec={spec} label={label} value={Number(value)} onChange={set} />
      break
    case 'range':
      body = <RangeControl spec={spec} label={label} value={value as [number, number]} onChange={set} />
      break
    case 'bool':
      body = (
        <div className="pc-head">
          <label className="pc-label" htmlFor={`p-${spec.name}`}>
            {label}
          </label>
          <span className="switch">
            <input id={`p-${spec.name}`} type="checkbox" role="switch" checked={Boolean(value)} onChange={(e) => set(e.target.checked)} />
            <span />
          </span>
        </div>
      )
      break
    case 'select':
      body = <SelectControl spec={spec} label={label} value={String(value)} onChange={set} />
      break
    case 'multiselect':
      body = <MultiSelectControl spec={spec} label={label} value={Array.isArray(value) ? (value as string[]) : []} onChange={set} />
      break
    case 'color':
      body = (
        <ColorControl
          spec={spec}
          label={label}
          value={String(value)}
          onChange={set}
          canPick={canUseViewer}
          picking={tool?.kind === 'pick' && tool.param === spec.name}
          onPick={() => onTool(tool?.kind === 'pick' && tool.param === spec.name ? null : { kind: 'pick', param: spec.name })}
        />
      )
      break
    case 'roi':
      body = (
        <RoiControl spec={spec} label={label} value={(value as RoiPolygon[]) ?? []} onChange={set} tool={tool} onTool={onTool} canDraw={canUseViewer} />
      )
      break
  }
  return (
    <div className="pc" data-param={spec.name}>
      {body}
      {spec.help ? <p className="pc-help">{spec.help}</p> : null}
    </div>
  )
}

export const ParamControls = memo(function ParamControls({
  specs,
  values,
  onChange,
  onResetGroup,
  tool,
  onTool,
  canUseViewer,
}: ParamControlsProps) {
  const [collapsed, setCollapsed] = useState<string[]>(() => loadJson<string[]>('collapsedGroups', []))
  const toggle = (g: string) =>
    setCollapsed((prev) => {
      const next = prev.includes(g) ? prev.filter((x) => x !== g) : [...prev, g]
      saveJson('collapsedGroups', next)
      return next
    })
  if (specs.length === 0) return <p className="faint pc-empty">조절할 파라미터가 없습니다.</p>
  return (
    <div className="param-groups">
      {groupSpecs(specs).map(([group, items]) => {
        const open = !collapsed.includes(group)
        const changed = items.filter((s) => !sameValue(values[s.name], s.default))
        return (
          <section key={group} className="pg">
            <div className="pg-head">
              <button type="button" className="pg-toggle" aria-expanded={open} onClick={() => toggle(group)}>
                <ChevronDown size={14} strokeWidth={1.75} className="pg-chevron" />
                <span>{group}</span>
                {changed.length > 0 ? <span className="pg-count num">변경 {changed.length}</span> : null}
              </button>
              <button
                type="button"
                className="btn btn-sm btn-quiet"
                disabled={changed.length === 0}
                onClick={() => onResetGroup(items.map((s) => s.name))}
                title={`${group}: 기본값으로`}
              >
                <RotateCcw size={12} strokeWidth={1.75} /> 기본값
              </button>
            </div>
            {open ? (
              <div className="pg-body">
                {items.map((s) => (
                  <Control
                    key={s.name}
                    spec={s}
                    value={values[s.name] ?? s.default}
                    onChange={onChange}
                    tool={tool}
                    onTool={onTool}
                    canUseViewer={canUseViewer}
                  />
                ))}
              </div>
            ) : null}
          </section>
        )
      })}
    </div>
  )
})
