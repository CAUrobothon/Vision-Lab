// The scoring criteria (docs/SCORING.md §3, §5) shared by every screen that scores: the 채점 page,
// the rail's batch chips, the viewer's 정답 layer and the workbench's live score. One value per
// viewer, kept in localStorage; edits reach every screen at once.
import { useMemo, useSyncExternalStore } from 'react'
import { useScoreCriteriaInfo } from '../api/queries'
import type { Batch, ScoreCriteria, ScoreCriteriaInfo, ScoreCriterionLimit } from '../api/types'
import { useSettled } from '../lib/settled'
import { sharedValue } from '../lib/storage'

type Key = keyof ScoreCriteria

// Defaults and limits come from the server (GET /api/score/criteria). Until it answers (the first
// paint, or a server without scoring) these copies of its values stand in.
const tau = { min: 0, max: null, min_exclusive: false, max_exclusive: false, step: 0.5, integer: false }
const FALLBACK: ScoreCriteriaInfo = {
  defaults: { center_px: 5, angle_deg: 5, vertex_px: 5, gate_px: 29, bootstrap_reps: 2000, bootstrap_level: 0.95, bootstrap_seed: 0 },
  limits: {
    center_px: tau,
    angle_deg: tau,
    vertex_px: tau,
    gate_px: { ...tau, step: 1 },
    bootstrap_reps: { min: 1, max: 20000, min_exclusive: false, max_exclusive: false, step: 100, integer: true },
    bootstrap_level: { min: 0, max: 1, min_exclusive: true, max_exclusive: true, step: 0.01, integer: false },
    bootstrap_seed: { min: 0, max: null, min_exclusive: false, max_exclusive: false, step: 1, integer: true },
  },
}

const KEYS = Object.keys(FALLBACK.defaults) as Key[]

/** The server's criteria defaults and limits (the stand-in until it has answered). */
export function useCriteriaInfo(): ScoreCriteriaInfo {
  return useScoreCriteriaInfo().data ?? FALLBACK
}

/** `v` is a value the server accepts for a field with these limits. */
export function fits(v: unknown, lim: ScoreCriterionLimit): v is number {
  return (
    typeof v === 'number' &&
    Number.isFinite(v) &&
    (lim.min === null || (lim.min_exclusive ? v > lim.min : v >= lim.min)) &&
    (lim.max === null || (lim.max_exclusive ? v < lim.max : v <= lim.max)) &&
    (!lim.integer || Number.isInteger(v))
  )
}

// Only the fields the viewer changed are kept, so a later change of a server default reaches every
// field they never touched (an older full copy reads the same way).
const store = sharedValue<Partial<ScoreCriteria>>('score.criteria', {}, (raw) =>
  raw && typeof raw === 'object' ? (raw as Partial<ScoreCriteria>) : {},
)

/** The criteria in use (the viewer's changes on the server's defaults) and a setter. */
export function useCriteria(): [ScoreCriteria, (patch: Partial<ScoreCriteria>) => void] {
  const { defaults, limits } = useCriteriaInfo()
  const changed = useSyncExternalStore(store.subscribe, store.get)
  const value = useMemo(() => {
    const out = { ...defaults }
    for (const k of KEYS) if (fits(changed[k], limits[k])) out[k] = changed[k]
    return out
  }, [defaults, limits, changed])
  const set = (patch: Partial<ScoreCriteria>) => {
    const next = { ...value, ...patch }
    store.set(Object.fromEntries(KEYS.filter((k) => next[k] !== defaults[k]).map((k) => [k, next[k]])))
  }
  return [value, set]
}

export function resetCriteria() {
  store.set({})
}

export function isDefaultCriteria(c: ScoreCriteria, defaults: ScoreCriteria): boolean {
  return KEYS.every((k) => c[k] === defaults[k])
}

/** The four values one image's score depends on (bootstrap settings never change it). */
export function thresholdsOf(c: ScoreCriteria): Pick<ScoreCriteria, 'center_px' | 'angle_deg' | 'vertex_px' | 'gate_px'> {
  return { center_px: c.center_px, angle_deg: c.angle_deg, vertex_px: c.vertex_px, gate_px: c.gate_px }
}

/** The criteria once they have stopped changing for 400 ms (typing 1 2 → 12 asks the server
 *  once), with their four thresholds; `settling` while an edit is still waiting. */
export function useSettledCriteria(): {
  criteria: ScoreCriteria
  thresholds: ReturnType<typeof thresholdsOf>
  settling: boolean
} {
  const [now] = useCriteria()
  const criteria = useSettled(now, 400)
  const thresholds = useMemo(() => thresholdsOf(criteria), [criteria])
  return { criteria, thresholds, settling: JSON.stringify(criteria) !== JSON.stringify(now) }
}

/** The GT set picked for each dataset (the rail's 정답셋 row; the 채점 page starts from it). */
const picked = sharedValue<Record<string, number>>('score.gtSet', {}, (raw) =>
  raw && typeof raw === 'object' ? (raw as Record<string, number>) : {},
)

export function useGtPick(datasetId: number | null): [number | null, (id: number) => void] {
  const all = useSyncExternalStore(picked.subscribe, picked.get)
  const id = datasetId === null ? null : (all[String(datasetId)] ?? null)
  return [id, (gid) => datasetId !== null && picked.set({ ...picked.get(), [String(datasetId)]: gid })]
}

/** The picked set if it still exists, else the newest one. */
export function chooseGt<T extends { id: number }>(sets: T[] | undefined, pickedId: number | null): T | null {
  if (!sets || sets.length === 0) return null
  return sets.find((s) => s.id === pickedId) ?? sets[0]
}

/** Names the state of a dataset's batches: which exist and which are finished. A score made for
 *  one stamp is reused until a batch is added, removed or finishes (a running one is polled). */
export function batchStamp(list: Batch[] | undefined): string {
  return (list ?? []).map((b) => `${b.id}${b.finished ? '' : '*'}`).join(',')
}
