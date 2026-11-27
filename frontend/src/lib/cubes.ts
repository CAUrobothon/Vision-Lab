// Cube class presentation and frame lookup, shared by the viewer, tables and Compare.
import type { Cube, CubeColor, FrameRecord, Pt } from '../api/types'

// Overlay colours match cubevision/draw.py (OVERLAY_BGR) so baked exports look the same.
// Every class also carries a letter, so the overlay never relies on colour alone.
export const CUBE_STYLE: Record<CubeColor, { stroke: string; short: string; name: string }> = {
  red: { stroke: '#eb2828', short: 'R', name: '빨강' },
  green: { stroke: '#28be3c', short: 'G', name: '초록' },
  blue: { stroke: '#146ee6', short: 'B', name: '파랑' },
  wood: { stroke: '#d7963c', short: 'W', name: '나무' },
  unknown: { stroke: '#c800c8', short: '?', name: '미분류' },
}

/** A class name in words; a colour this list does not know (a GT label's own) as it is. */
export function colorName(color: string | null): string {
  return CUBE_STYLE[color as CubeColor]?.name ?? String(color)
}

export function cubeLabel(c: Cube): string {
  const s = CUBE_STYLE[c.color].short
  return c.angle === null ? s : `${s} ${Math.round(c.angle)}°`
}

/** Index of the stored frame with the largest number ≤ `frame` (frames sorted ascending), or -1. */
export function frameAtOrBefore(frames: FrameRecord[], frame: number): number {
  let lo = 0
  let hi = frames.length - 1
  let best = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (frames[mid].frame <= frame) {
      best = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  return best
}

/** The frames a run was asked to process: start (inclusive), end (exclusive, null = to the end), every `stride`th. */
export interface FrameSpan {
  start: number
  end: number | null
  stride: number
}

export function spanOf(run: { frame_start: number; frame_end: number | null; stride: number }): FrameSpan {
  return { start: run.frame_start, end: run.frame_end, stride: run.stride }
}

export function outsideSpan(span: FrameSpan, frame: number): boolean {
  return frame < span.start || (span.end !== null && frame >= span.end)
}

/**
 * The stored record that describes `frame`: the latest one at or before it, but only while `frame`
 * is inside the run's range and within that record's stride (the frames the run skipped after
 * it). Outside that, nothing describes the frame and null is returned. Without a span (images,
 * unknown range) the latest record at or before the frame is used.
 */
export function recordFor(frames: FrameRecord[], frame: number, span?: FrameSpan | null): FrameRecord | null {
  const i = frameAtOrBefore(frames, frame)
  if (i < 0) return null
  const rec = frames[i]
  if (span && (outsideSpan(span, frame) || frame >= rec.frame + Math.max(1, span.stride))) return null
  return rec
}

export function countColors(cubes: Cube[]): Record<CubeColor, number> {
  const out: Record<CubeColor, number> = { red: 0, green: 0, blue: 0, wood: 0, unknown: 0 }
  for (const c of cubes) out[c.color]++
  return out
}

/** Even-odd point-in-polygon test in image pixels. */
export function pointInPolygon([x, y]: Pt, poly: Pt[]): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]
    const [xj, yj] = poly[j]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

/** The outline used for hit-testing and highlighting: quad, else box, else polygon. */
export function cubeOutline(c: Cube): Pt[] | null {
  if (c.quad) return c.quad
  if (c.box) {
    const [x1, y1, x2, y2] = c.box
    return [
      [x1, y1],
      [x2, y1],
      [x2, y2],
      [x1, y2],
    ]
  }
  return c.polygon
}

/** Index of the cube under an image point (smallest area wins), or null. */
export function hitTestCubes(cubes: Cube[], pt: Pt): number | null {
  let best: number | null = null
  let bestArea = Infinity
  cubes.forEach((c, i) => {
    const outline = cubeOutline(c)
    const hit = outline
      ? pointInPolygon(pt, outline)
      : Math.hypot(pt[0] - c.center[0], pt[1] - c.center[1]) < 12
    if (hit && c.area < bestArea) {
      best = i
      bestArea = c.area
    }
  })
  return best
}
