// SVG overlay in media pixels (viewBox = 0 0 W H). It lives inside the same transformed box as the
// image/video, so it zooms and pans with it. Stroke widths, dashes, dots and labels are divided by
// the zoom (k = 1 / zoom) so they keep a constant size on screen.
import { memo, useId } from 'react'
import type { Cube, Pt, RoiPolygon, ScoreImageDetail } from '../api/types'
import { CUBE_STYLE, cubeLabel, cubeOutline } from '../lib/cubes'
import type { LayerPrefs } from './layers'

export interface ShownSet {
  key: string
  cubes: Cube[]
  dashed: boolean
  tag: string | null // "A" / "B" in Compare's overlay mode
}

export interface DebugImage {
  url: string
  isMask: boolean // 1-channel masks are tinted here; colour layers are an <img> under the overlay
}

export interface RoiDraft {
  mode: 'include' | 'exclude'
  points: Pt[]
  cursor: Pt | null
}

interface OverlayProps {
  width: number
  height: number
  scale: number
  sets: ShownSet[]
  layers: LayerPrefs
  roi: RoiPolygon[] | null
  debug: DebugImage | null
  highlight: number | null // index into sets[0].cubes
  draft: RoiDraft | null
  gt?: ScoreImageDetail | null // the 정답 layer: the media's GT against what is drawn
}

const toPoints = (p: Pt[]) => p.map(([x, y]) => `${x},${y}`).join(' ')
const toPath = (p: Pt[]) => 'M' + p.map(([x, y]) => `${x} ${y}`).join('L') + 'Z'

function bbox(points: Pt[]): [number, number, number, number] {
  let x1 = Infinity
  let y1 = Infinity
  let x2 = -Infinity
  let y2 = -Infinity
  for (const [x, y] of points) {
    x1 = Math.min(x1, x)
    y1 = Math.min(y1, y)
    x2 = Math.max(x2, x)
    y2 = Math.max(y2, y)
  }
  return [x1, y1, x2, y2]
}

function CubeMark({ cube, layers, k, dashed, tag }: { cube: Cube; layers: LayerPrefs; k: number; dashed: boolean; tag: string | null }) {
  const col = CUBE_STYLE[cube.color].stroke
  const outline = cube.quad ?? (cube.box ? cubeOutline(cube) : null)
  const [cx, cy] = cube.center
  const dash = dashed ? `${7 * k} ${5 * k}` : undefined
  const tick = 0.35 * Math.sqrt(Math.max(cube.area, 1))
  const rad = cube.angle === null ? 0 : (cube.angle * Math.PI) / 180
  return (
    <g>
      {layers.raw && cube.polygon ? (
        <polygon className="ov-raw" points={toPoints(cube.polygon)} strokeDasharray={`${2 * k} ${3 * k}`} />
      ) : null}
      {layers.raw && cube.box ? (
        <rect
          className="ov-raw box"
          strokeDasharray={`${2 * k} ${3 * k}`}
          x={cube.box[0]}
          y={cube.box[1]}
          width={cube.box[2] - cube.box[0]}
          height={cube.box[3] - cube.box[1]}
        />
      ) : null}
      {layers.quad && outline ? (
        <>
          <polygon className="ov-halo" points={toPoints(outline)} strokeDasharray={dash} />
          <polygon className="ov-quad" points={toPoints(outline)} stroke={col} strokeDasharray={dash} />
        </>
      ) : null}
      {layers.center ? (
        <>
          {cube.angle !== null ? (
            <>
              <line className="ov-halo" x1={cx} y1={cy} x2={cx + tick * Math.cos(rad)} y2={cy + tick * Math.sin(rad)} />
              <line className="ov-tick" x1={cx} y1={cy} x2={cx + tick * Math.cos(rad)} y2={cy + tick * Math.sin(rad)} stroke={col} />
            </>
          ) : null}
          <circle className="ov-dot" cx={cx} cy={cy} r={3.5 * k} fill={col} />
        </>
      ) : null}
      {layers.label ? (
        <g transform={`translate(${cx} ${cy}) scale(${k})`}>
          <text className="ov-label" x={7} y={tag && dashed ? 19 : -7}>
            {tag ? `${tag}: ` : ''}
            {cubeLabel(cube)}
          </text>
        </g>
      ) : null}
    </g>
  )
}

/** The 정답 layer: GT top faces dashed (light = passes, warm = fails or missed), a thin line from
 *  each detection's centre to its GT match point, missed GT and false positives named in words. */
function GtMarks({ d, k }: { d: ScoreImageDetail; k: number }) {
  const preds = d.predictions
  const dash = `${5 * k} ${4 * k}`
  const tag = (x: number, y: number, text: string, cls: string) => (
    <g transform={`translate(${x} ${y}) scale(${k})`}>
      <text className={`ov-label ov-gt-text ${cls}`} x={8} y={16}>
        {text}
      </text>
    </g>
  )
  return (
    <g className="ov-gt">
      {d.gt.map((g, i) => {
        const c = d.cubes[i]
        const state = d.scored && c ? c.outcome : 'pending'
        const [mx, my] = g.match_point
        return (
          <g key={g.id}>
            {g.quad ? (
              <>
                <polygon className="ov-halo" points={toPoints(g.quad)} strokeDasharray={dash} />
                <polygon className={`ov-gt-quad ${state}`} points={toPoints(g.quad)} strokeDasharray={dash} />
              </>
            ) : (
              <rect
                className={`ov-gt-point ${state}`}
                x={mx - 5 * k}
                y={my - 5 * k}
                width={10 * k}
                height={10 * k}
                transform={`rotate(45 ${mx} ${my})`}
              />
            )}
            {state === 'missed' ? (
              <g className="ov-gt-miss">
                <line className="ov-halo" x1={mx - 7 * k} y1={my - 7 * k} x2={mx + 7 * k} y2={my + 7 * k} />
                <line className="ov-halo" x1={mx - 7 * k} y1={my + 7 * k} x2={mx + 7 * k} y2={my - 7 * k} />
                <line className="ov-gt-x" x1={mx - 7 * k} y1={my - 7 * k} x2={mx + 7 * k} y2={my + 7 * k} />
                <line className="ov-gt-x" x1={mx - 7 * k} y1={my + 7 * k} x2={mx + 7 * k} y2={my - 7 * k} />
                {tag(mx, my, `미검출 ${g.id}`, 'bad')}
              </g>
            ) : null}
          </g>
        )
      })}
      {d.cubes.map((c, i) => {
        const from = c.matched && c.pred_index !== null ? preds[c.pred_index]?.center : null
        if (!from) return null
        const [gx, gy] = d.gt[i].match_point
        return (
          <g key={`${i}-${c.pred_index}`}>
            <line className="ov-halo thin" x1={from[0]} y1={from[1]} x2={gx} y2={gy} />
            <line className="ov-gt-link" x1={from[0]} y1={from[1]} x2={gx} y2={gy} />
            <circle className="ov-gt-mp" cx={gx} cy={gy} r={2.5 * k} />
          </g>
        )
      })}
      {d.false_positives.map((j) => {
        const c = preds[j]
        if (!c) return null
        const r = Math.max(12 * k, 0.75 * Math.sqrt(Math.max(c.area, 1)))
        return (
          <g key={`fp-${j}`} className="ov-gt-fp">
            <circle className="ov-halo" cx={c.center[0]} cy={c.center[1]} r={r} strokeDasharray={`${3 * k} ${3 * k}`} />
            <circle className="ov-gt-ring" cx={c.center[0]} cy={c.center[1]} r={r} strokeDasharray={`${3 * k} ${3 * k}`} />
            {tag(c.center[0] + r * 0.7, c.center[1] + r * 0.4, '오검출', 'bad')}
          </g>
        )
      })}
    </g>
  )
}

function Highlight({ cube, k }: { cube: Cube; k: number }) {
  const outline = cubeOutline(cube)
  const [x1, y1, x2, y2] = outline ? bbox(outline) : [cube.center[0] - 8, cube.center[1] - 8, cube.center[0] + 8, cube.center[1] + 8]
  const pad = 6 * k
  const l = 10 * k
  const [a, b, c, d] = [x1 - pad, y1 - pad, x2 + pad, y2 + pad]
  const path = `M${a} ${b + l}V${b}H${a + l}M${c - l} ${b}H${c}V${b + l}M${c} ${d - l}V${d}H${c - l}M${a + l} ${d}H${a}V${d - l}`
  return (
    <g>
      {outline ? <polygon className="ov-hl-fill" points={toPoints(outline)} /> : null}
      <path className="ov-halo hl" d={path} />
      <path className="ov-hl" d={path} />
    </g>
  )
}

export const Overlay = memo(function Overlay({ width, height, scale, sets, layers, roi, debug, highlight, draft, gt = null }: OverlayProps) {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, '')
  const k = 1 / scale
  // Polygons with fewer than 3 points carve nothing out of the region (cubevision/roi.py skips them).
  const includes = roi?.filter((p) => p.mode === 'include' && p.points.length >= 3) ?? []
  const excludes = roi?.filter((p) => p.mode === 'exclude' && p.points.length >= 3) ?? []
  const showRoi = layers.roi && includes.length + excludes.length > 0
  const hlCube = highlight !== null ? sets[0]?.cubes[highlight] : undefined
  return (
    <svg
      className="overlay"
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      style={{ ['--k' as string]: k }}
      aria-hidden="true"
    >
      <defs>
        <pattern id={`hatch-${uid}`} width={10 * k} height={10 * k} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2={10 * k} stroke="rgb(255 255 255 / 0.45)" strokeWidth={2 * k} />
        </pattern>
        {debug?.isMask ? (
          <mask id={`mask-${uid}`} maskUnits="userSpaceOnUse" x="0" y="0" width={width} height={height}>
            <image href={debug.url} width={width} height={height} className="ov-debug-img" />
          </mask>
        ) : null}
        {showRoi ? (
          // White = dimmed (outside the region). Each polygon is filled on its own with the default
          // nonzero rule, like cubevision/roi.py: union of includes, minus the union of excludes.
          <mask id={`roi-${uid}`} maskUnits="userSpaceOnUse" x="0" y="0" width={width} height={height}>
            <rect width={width} height={height} fill={includes.length > 0 ? 'white' : 'black'} />
            {includes.map((p, i) => (
              <path key={`i${i}`} d={toPath(p.points)} fill="black" />
            ))}
            {excludes.map((p, i) => (
              <path key={`x${i}`} d={toPath(p.points)} fill="white" />
            ))}
          </mask>
        ) : null}
      </defs>

      {debug?.isMask ? (
        <rect width={width} height={height} className="ov-debug-tint" mask={`url(#mask-${uid})`} opacity={layers.debugOpacity} />
      ) : null}

      {showRoi ? (
        <g>
          <rect className="ov-roi-dim" width={width} height={height} mask={`url(#roi-${uid})`} />
          {excludes.map((p, i) => (
            <g key={`x${i}`}>
              <path d={toPath(p.points)} fill={`url(#hatch-${uid})`} />
              <path className="ov-roi-line" d={toPath(p.points)} strokeDasharray={`${6 * k} ${4 * k}`} />
            </g>
          ))}
          {includes.map((p, i) => (
            <path key={`i${i}`} className="ov-roi-line" d={toPath(p.points)} strokeDasharray={`${6 * k} ${4 * k}`} />
          ))}
        </g>
      ) : null}

      {layers.gt && gt ? <GtMarks d={gt} k={k} /> : null}

      {sets.map((set) => (
        <g key={set.key}>
          {set.cubes.map((c, i) => (
            <CubeMark key={i} cube={c} layers={layers} k={k} dashed={set.dashed} tag={set.tag} />
          ))}
        </g>
      ))}

      {hlCube ? <Highlight cube={hlCube} k={k} /> : null}

      {draft ? (
        <g>
          {draft.points.length > 0 ? (
            <polyline
              className="ov-draft"
              points={toPoints(draft.cursor ? [...draft.points, draft.cursor] : draft.points)}
              strokeDasharray={draft.mode === 'exclude' ? `${6 * k} ${4 * k}` : undefined}
            />
          ) : null}
          {draft.points.map(([x, y], i) => (
            <rect
              key={i}
              className={i === 0 ? 'ov-vertex first' : 'ov-vertex'}
              x={x - (i === 0 ? 5 : 3.5) * k}
              y={y - (i === 0 ? 5 : 3.5) * k}
              width={(i === 0 ? 10 : 7) * k}
              height={(i === 0 ? 10 : 7) * k}
            />
          ))}
        </g>
      ) : null}
    </svg>
  )
})
