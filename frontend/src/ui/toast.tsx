// Short-lived messages for things that finish outside the user's focus (uploads, saves, errors).
import { createContext, use, useCallback, useMemo, useState, type ReactNode } from 'react'
import { AlertTriangle, Check, X } from 'lucide-react'

interface Toast {
  id: number
  kind: 'ok' | 'error'
  text: string
}

interface ToastApi {
  ok: (text: string) => void
  error: (text: string) => void
}

const ToastContext = createContext<ToastApi | null>(null)

let nextId = 1

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const dismiss = useCallback((id: number) => setToasts((ts) => ts.filter((t) => t.id !== id)), [])
  const push = useCallback(
    (kind: Toast['kind'], text: string) => {
      const id = nextId++
      setToasts((ts) => [...ts.slice(-3), { id, kind, text }])
      window.setTimeout(() => dismiss(id), kind === 'error' ? 8000 : 4000)
    },
    [dismiss],
  )
  const api = useMemo<ToastApi>(() => ({ ok: (t) => push('ok', t), error: (t) => push('error', t) }), [push])
  return (
    <ToastContext value={api}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind}`}>
            {t.kind === 'error' ? <AlertTriangle size={14} strokeWidth={1.75} /> : <Check size={14} strokeWidth={1.75} />}
            <p>{t.text}</p>
            <button type="button" className="icon-btn sm" aria-label="알림 닫기" onClick={() => dismiss(t.id)}>
              <X size={12} strokeWidth={1.75} />
            </button>
          </div>
        ))}
      </div>
    </ToastContext>
  )
}

export function useToast(): ToastApi {
  const ctx = use(ToastContext)
  if (!ctx) throw new Error('useToast outside ToastProvider')
  return ctx
}
