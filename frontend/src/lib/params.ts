// Parameter helpers: defaults from ParamSpec, merging stored values, and diffs for the history.
import type { ParamSpec, ParamValue, Params } from '../api/types'

export function defaultParams(specs: ParamSpec[]): Params {
  const out: Params = {}
  for (const s of specs) out[s.name] = s.default
  return out
}

/** Stored or preset values on top of the defaults; keys the model no longer has are dropped. */
export function mergeParams(specs: ParamSpec[], given: Params | null | undefined): Params {
  const out = defaultParams(specs)
  if (!given) return out
  for (const s of specs) if (s.name in given) out[s.name] = given[s.name]
  return out
}

export function sameValue(a: ParamValue | undefined, b: ParamValue | undefined): boolean {
  return JSON.stringify(a) === JSON.stringify(b)
}

/** True when `a` and `b` hold the same value for every param of the model. */
export function sameParams(specs: ParamSpec[], a: Params, b: Params): boolean {
  return specs.every((s) => sameValue(a[s.name] ?? s.default, b[s.name] ?? s.default))
}

/** The params whose value differs between `from` and `to`, in spec order. */
export function diffParams(specs: ParamSpec[], from: Params, to: Params): { spec: ParamSpec; from: ParamValue; to: ParamValue }[] {
  return specs
    .filter((s) => !sameValue(from[s.name] ?? s.default, to[s.name] ?? s.default))
    .map((s) => ({ spec: s, from: from[s.name] ?? s.default, to: to[s.name] ?? s.default }))
}

/** Names of params whose value differs from the spec default, in spec order. */
export function changedParams(specs: ParamSpec[], params: Params): string[] {
  return specs.filter((s) => s.name in params && !sameValue(params[s.name], s.default)).map((s) => s.name)
}

/** Only the values that differ from the defaults. This is what gets remembered locally, so a later
 *  change of a default reaches every param the user never touched (and removed params vanish). */
export function paramOverrides(specs: ParamSpec[], params: Params): Params {
  const out: Params = {}
  for (const name of changedParams(specs, params)) out[name] = params[name]
  return out
}

/** Decimals needed to show every multiple of `step` exactly: 0.125 -> 3, 0.05 -> 2, 2.5 -> 1, 5 -> 0. */
export function stepDecimals(step: number): number {
  for (let d = 0; d < 6; d++) {
    const scaled = Math.abs(step) * 10 ** d
    if (Math.abs(scaled - Math.round(scaled)) < 1e-6 * Math.max(1, scaled)) return d
  }
  return 6
}

/** Decimals shown for a number param: from its step, or (no step) from 1/100 of its range. */
export function paramDecimals(spec: ParamSpec): number {
  if (spec.type === 'int') return 0
  if (spec.step) return stepDecimals(spec.step)
  const span = (spec.max ?? 1) - (spec.min ?? 0)
  return span > 0 ? Math.min(4, Math.max(0, Math.ceil(-Math.log10(span / 100) - 1e-9))) : 2
}

export function formatParamValue(spec: ParamSpec | undefined, v: ParamValue | undefined): string {
  if (v === undefined) return '—'
  if (Array.isArray(v)) {
    if (spec?.type === 'multiselect') {
      if (v.length === 0) return '전체'
      const labels = (v as string[]).map((x) => spec.options?.find((o) => o.value === x)?.label ?? x)
      return labels.join(', ')
    }
    if (v.length === 2 && typeof v[0] === 'number') return `${v[0]}–${v[1]}`
    return `영역 ${v.length}개`
  }
  if (typeof v === 'boolean') return v ? '켬' : '끔'
  if (spec?.type === 'select') return spec.options?.find((o) => o.value === v)?.label ?? String(v)
  if (typeof v === 'number') {
    return `${spec?.type === 'float' ? v.toFixed(paramDecimals(spec)) : v}${spec?.unit ? ' ' + spec.unit : ''}`
  }
  return String(v)
}

/** Groups in first-seen order, so the pipeline author decides the order. */
export function groupSpecs(specs: ParamSpec[]): [string, ParamSpec[]][] {
  const groups = new Map<string, ParamSpec[]>()
  for (const s of specs) {
    const g = groups.get(s.group)
    if (g) g.push(s)
    else groups.set(s.group, [s])
  }
  return [...groups.entries()]
}
