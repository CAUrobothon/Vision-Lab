import { useEffect, useState } from 'react'

/** `value` once it has stayed the same for `ms` (typing 1 2 → 12 asks the server once). Values
 *  compare by JSON, so a new object with the same numbers is no change. */
export function useSettled<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value)
  const key = JSON.stringify(value)
  useEffect(() => {
    const t = window.setTimeout(() => setSettled(JSON.parse(key) as T), ms)
    return () => window.clearTimeout(t)
  }, [key, ms])
  return settled
}
