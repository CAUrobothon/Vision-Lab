// localStorage with a version prefix. Storage can be missing or throw (private windows, blocked
// site data), so every access is wrapped and the app works without it.

const PREFIX = 'vl1.'

export function loadJson<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(PREFIX + key)
    return raw === null ? fallback : (JSON.parse(raw) as T)
  } catch {
    return fallback
  }
}

export function saveJson(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(PREFIX + key, JSON.stringify(value))
  } catch {
    // Storage is a convenience only.
  }
}

/** A value in localStorage that every component using it shares (one value per key, also across
 *  tabs through the storage event). `read` checks what was stored and falls back where needed. */
export interface Shared<T> {
  get: () => T
  set: (next: T) => void
  subscribe: (listener: () => void) => () => void
}

export function sharedValue<T>(key: string, fallback: T, read: (raw: unknown) => T = (raw) => raw as T): Shared<T> {
  const load = (): T => {
    try {
      return read(loadJson<unknown>(key, fallback))
    } catch {
      return fallback
    }
  }
  let value = load()
  const listeners = new Set<() => void>()
  const notify = () => listeners.forEach((l) => l())
  try {
    window.addEventListener('storage', (e) => {
      if (e.key === PREFIX + key) {
        value = load()
        notify()
      }
    })
  } catch {
    // No window events (never in the browser): sharing within the tab still works.
  }
  return {
    get: () => value,
    set: (next) => {
      value = next
      saveJson(key, next)
      notify()
    },
    subscribe: (listener) => {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
  }
}
