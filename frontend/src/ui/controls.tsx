// Small reusable controls: a two-step delete button, a star toggle, inline note and tag editors,
// cube class swatches, run status marks, number and name fields and a modal dialog.
import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties, type MouseEvent as ReactMouseEvent, type ReactNode } from 'react'
import { Star, Trash2, X } from 'lucide-react'
import type { CubeColor, RunStatus } from '../api/types'
import { CUBE_STYLE } from '../lib/cubes'

/** Destructive action that needs a second click within 4 s ("삭제" → "정말 삭제"). */
export function ConfirmButton({
  label,
  confirmLabel = '정말 삭제',
  onConfirm,
  disabled,
  small = true,
  icon = true,
  quiet = false,
}: {
  label: string
  confirmLabel?: string
  onConfirm: () => void
  disabled?: boolean
  small?: boolean
  icon?: boolean
  quiet?: boolean // a plain trash icon until the first click arms it
}) {
  const [armed, setArmed] = useState(false)
  useEffect(() => {
    if (!armed) return
    const t = window.setTimeout(() => setArmed(false), 4000)
    return () => window.clearTimeout(t)
  }, [armed])
  if (quiet && !armed)
    return (
      <button type="button" className="icon-btn" disabled={disabled} aria-label={label} title={label} onClick={() => setArmed(true)}>
        <Trash2 size={14} strokeWidth={1.75} />
      </button>
    )
  return (
    <button
      type="button"
      className={`btn btn-danger${small ? ' btn-sm' : ''}${armed ? ' armed' : ''}`}
      disabled={disabled}
      autoFocus={quiet}
      onClick={() => {
        if (armed) {
          setArmed(false)
          onConfirm()
        } else setArmed(true)
      }}
      onBlur={() => setArmed(false)}
    >
      {icon ? <Trash2 size={13} strokeWidth={1.75} /> : null}
      {armed ? confirmLabel : label}
    </button>
  )
}

export function StarButton({ on, onToggle, label }: { on: boolean; onToggle: () => void; label: string }) {
  return (
    <button
      type="button"
      className={`star-btn${on ? ' on' : ''}`}
      aria-pressed={on}
      aria-label={on ? `${label} 별표 해제` : `${label} 별표`}
      title={on ? '별표 해제' : '별표'}
      onClick={(e) => {
        e.stopPropagation()
        onToggle()
      }}
    >
      <Star size={14} strokeWidth={1.75} fill={on ? 'currentColor' : 'none'} />
    </button>
  )
}

export function ClassSwatch({ color, withName = false }: { color: CubeColor; withName?: boolean }) {
  const st = CUBE_STYLE[color]
  return (
    <span className="cls">
      <i style={{ background: st.stroke }}>{st.short}</i>
      {withName ? st.name : null}
    </span>
  )
}

export const STATUS_LABEL: Record<RunStatus, string> = {
  queued: '대기',
  running: '실행 중',
  done: '완료',
  failed: '실패',
  cancelled: '취소됨',
}

export function StatusMark({ status }: { status: RunStatus }) {
  return <span className={`status ${status}`}>{STATUS_LABEL[status]}</span>
}

export const STATUS_OPTIONS = Object.entries(STATUS_LABEL) as [RunStatus, string][]

/** Text that becomes an input on click; saves on Enter or blur, Esc cancels. */
export function InlineText({
  value,
  placeholder,
  onSave,
  label,
}: {
  value: string
  placeholder: string
  onSave: (v: string) => void
  label: string
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(value)
  const ref = useRef<HTMLInputElement>(null)
  useEffect(() => {
    if (editing) ref.current?.select()
  }, [editing])
  if (!editing)
    return (
      <button
        type="button"
        className={`inline-text${value ? '' : ' empty'}`}
        onClick={() => {
          setDraft(value)
          setEditing(true)
        }}
        aria-label={`${label} 편집`}
        title={value || placeholder}
      >
        {value || placeholder}
      </button>
    )
  const commit = () => {
    setEditing(false)
    if (draft.trim() !== value) onSave(draft.trim())
  }
  return (
    <input
      ref={ref}
      className="input inline-input"
      value={draft}
      aria-label={label}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') commit()
        if (e.key === 'Escape') setEditing(false)
      }}
    />
  )
}

export function TagEditor({ tags, onChange }: { tags: string[]; onChange: (tags: string[]) => void }) {
  const [draft, setDraft] = useState('')
  const add = () => {
    const t = draft.trim()
    setDraft('')
    if (t && !tags.includes(t)) onChange([...tags, t])
  }
  return (
    <div className="tag-editor">
      {tags.map((t) => (
        <span key={t} className="tag">
          {t}
          <button type="button" aria-label={`태그 ${t} 지우기`} onClick={() => onChange(tags.filter((x) => x !== t))}>
            <X size={10} strokeWidth={2} />
          </button>
        </span>
      ))}
      <input
        className="tag-input"
        value={draft}
        placeholder="+ 태그"
        aria-label="태그 추가"
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') add()
          if (e.key === 'Backspace' && !draft && tags.length) onChange(tags.slice(0, -1))
        }}
        onBlur={add}
      />
    </div>
  )
}

/** A number field typed as text: a valid number inside [min, max] applies as it is typed;
 *  anything else (empty, "-", "0.", out of range) waits, and on blur or Enter is clamped, or goes
 *  back to the value when unreadable. Clamping on every keystroke made fields impossible to clear.
 *  `integer` rounds what is committed; decimals also accept a comma ("0,5"). A bound may be
 *  ±Infinity (none) or exclusive: a value clamped onto an exclusive bound goes back as well. */
export function NumberField({
  value,
  min,
  max,
  minExclusive = false,
  maxExclusive = false,
  onCommit,
  label,
  integer = false,
  className = 'input num-input',
}: {
  value: number
  min: number
  max: number
  minExclusive?: boolean
  maxExclusive?: boolean
  onCommit: (v: number) => void
  label: string
  integer?: boolean
  className?: string
}) {
  const [draft, setDraft] = useState<string | null>(null)
  const read = (text: string) => Number(integer ? text : text.replace(',', '.'))
  const inside = (n: number) => (minExclusive ? n > min : n >= min) && (maxExclusive ? n < max : n <= max)
  const commit = () => {
    if (draft === null) return
    const n = integer ? Math.round(read(draft)) : read(draft)
    setDraft(null)
    const clamped = Math.min(max, Math.max(min, n))
    if (draft.trim() !== '' && Number.isFinite(n) && inside(clamped)) onCommit(clamped)
  }
  return (
    <input
      className={className}
      inputMode={integer ? 'numeric' : 'decimal'}
      aria-label={label}
      value={draft ?? String(value)}
      onChange={(e) => {
        const text = e.target.value
        setDraft(text)
        const n = read(text)
        const complete = integer ? Number.isInteger(n) : !text.endsWith('.') && Number.isFinite(n)
        if (text.trim() !== '' && complete && inside(n)) onCommit(n)
      }}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') commit()
        if (e.key === 'Escape') setDraft(null)
      }}
    />
  )
}

/** A name typed into a one-line form (a new dataset, a renamed folder): Enter submits the trimmed
 *  name when there is one, Esc cancels. `cancel` is the cancel button: the word, an X, or none
 *  (Esc only). */
export function NameForm({
  initial = '',
  label,
  placeholder,
  maxLength = 80,
  submitLabel,
  pending,
  onSubmit,
  onCancel,
  cancel = 'text',
  selectOnFocus = false,
  className,
  style,
}: {
  initial?: string
  label: string
  placeholder?: string
  maxLength?: number
  submitLabel: string
  pending: boolean
  onSubmit: (name: string) => void
  onCancel: () => void
  cancel?: 'text' | 'icon' | 'none'
  selectOnFocus?: boolean
  className?: string
  style?: CSSProperties
}) {
  const [name, setName] = useState(initial)
  return (
    <form
      className={className}
      style={style}
      onSubmit={(e) => {
        e.preventDefault()
        if (name.trim()) onSubmit(name.trim())
      }}
    >
      <input
        className="input"
        autoFocus
        value={name}
        maxLength={maxLength}
        placeholder={placeholder}
        aria-label={label}
        onChange={(e) => setName(e.target.value)}
        onFocus={selectOnFocus ? (e) => e.target.select() : undefined}
        onKeyDown={(e) => e.key === 'Escape' && onCancel()}
      />
      <button type="submit" className="btn btn-sm" disabled={!name.trim() || pending}>
        {submitLabel}
      </button>
      {cancel === 'text' ? (
        <button type="button" className="btn btn-sm btn-quiet" onClick={onCancel}>
          취소
        </button>
      ) : cancel === 'icon' ? (
        <button type="button" className="icon-btn sm" aria-label="취소" title="취소 (Esc)" onClick={onCancel}>
          <X size={12} strokeWidth={1.75} />
        </button>
      ) : null}
    </form>
  )
}

/** Closes the dialog a button sits in (its `onClose` then runs). */
export function closeDialog(e: ReactMouseEvent<HTMLElement>) {
  e.currentTarget.closest('dialog')?.close()
}

/** A modal dialog, shown when it mounts: a header with the title and an X; Esc, the X or a click
 *  on the backdrop close it, and `onClose` should then unmount it. `top` anchors it below the top
 *  edge instead of centring it, for content that grows and shrinks (a centred dialog would jump).
 *  Styled by .pj-dialog / .pj-body / .pj-head (workbench.css). */
export function Modal({
  title,
  className,
  top = false,
  onClose,
  children,
}: {
  title: ReactNode
  className?: string
  top?: boolean
  onClose: () => void
  children: ReactNode
}) {
  const ref = useRef<HTMLDialogElement>(null)
  const titleId = useId()
  useLayoutEffect(() => {
    const el = ref.current
    if (el && !el.open) el.showModal()
  }, [])
  return (
    <dialog
      ref={ref}
      className={`pj-dialog${top ? ' top' : ''}${className ? ` ${className}` : ''}`}
      aria-labelledby={titleId}
      onClose={onClose}
      onClick={(e) => {
        if (e.target === e.currentTarget) e.currentTarget.close() // the backdrop
      }}
    >
      <div className="pj-body">
        <header className="pj-head">
          <h2 id={titleId}>{title}</h2>
          <button type="button" className="icon-btn" aria-label="닫기" onClick={closeDialog}>
            <X size={16} strokeWidth={1.75} />
          </button>
        </header>
        {children}
      </div>
    </dialog>
  )
}
