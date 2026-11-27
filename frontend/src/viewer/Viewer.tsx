// The viewer: one component for images and videos, reused by the Workbench and Compare.
//
// The <img>/<video> and the SVG overlay share one box of W x H media pixels, moved with
// `transform: translate() scale()`. Zoom is about the cursor; drag pans.
//
// Video frame index of what is on screen = round(metadata.mediaTime * fps) from
// requestVideoFrameCallback (fallback: currentTime). Seeking to frame n sets
// currentTime = (n + 0.5) / fps, the middle of that frame's display interval, so float error can
// never land on the neighbour. Run results are drawn from the stored frame with the largest
// index <= the frame on screen.
import {
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  type Ref,
} from 'react'
import { Crosshair, Maximize, Minimize } from 'lucide-react'
import type { Cube, DebugLayer, FrameRecord, Media, Pt, RoiPolygon, ScoreImageDetail } from '../api/types'
import { hitTestCubes, outsideSpan, recordFor, type FrameSpan } from '../lib/cubes'
import { medianRgb, rgbToHex, rgbToHsvCv, rgbToLabCv, type Rgb } from '../lib/colour'
import { Overlay, type DebugImage, type ShownSet } from './Overlay'
import { Transport } from './Transport'
import { LayerMenu } from './LayerMenu'
import { GtCard } from './GtCard'
import type { LayerPrefs } from './layers'
import './viewer.css'

export interface OverlaySet {
  key: string
  frames: FrameRecord[] // sorted by frame
  exact?: boolean // a preview: draw only on its own frame
  span?: FrameSpan | null // a run's range and stride: draw nothing outside what it processed
  dashed?: boolean
  tag?: string
}

/** Zoom and pan in screen pixels: media point p is drawn at p * s + (tx, ty). */
export interface ViewState {
  s: number
  tx: number
  ty: number
}

/**
 * Zoom and pan in media-normalised units, so viewers of differently sized media can share it
 * (Compare): z = zoom relative to "fit", (u, v) = the media point (0..1 of width, height) at the
 * centre of the stage.
 */
export interface LinkedView {
  z: number
  u: number
  v: number
}

function toLinked(view: ViewState, fit: ViewState, cw: number, ch: number, w: number, h: number): LinkedView {
  return { z: view.s / fit.s, u: (cw / 2 - view.tx) / view.s / w, v: (ch / 2 - view.ty) / view.s / h }
}

function fromLinked(link: LinkedView, fit: ViewState, cw: number, ch: number, w: number, h: number): ViewState {
  const s = link.z * fit.s
  return { s, tx: cw / 2 - link.u * w * s, ty: ch / 2 - link.v * h * s }
}

export type ViewerTool =
  | { kind: 'pick'; label: string; onPick: (hex: string) => void; onCancel: () => void }
  | { kind: 'roi'; mode: 'include' | 'exclude'; onComplete: (points: Pt[]) => void; onCancel: () => void }

export interface ViewerHandle {
  seekFrame: (n: number) => void
  play: () => void
  pause: () => void
}

export interface ViewerProps {
  media: Media
  sets: OverlaySet[]
  layers: LayerPrefs
  onLayersChange: (patch: Partial<LayerPrefs>) => void
  debugLayers?: DebugLayer[] | null
  roi?: RoiPolygon[] | null
  highlight?: number | null
  onHighlight?: (i: number | null) => void
  onFrameChange?: (frame: number) => void
  onPausedChange?: (paused: boolean) => void
  badge?: ReactNode
  tool?: ViewerTool | null
  view?: LinkedView | null // controlled (Compare links two viewers; null = fit); omit for uncontrolled
  onViewChange?: (v: LinkedView | null) => void
  transport?: boolean
  keyboard?: boolean
  ref?: Ref<ViewerHandle>
  gtAvailable?: boolean // the media has ground truth: the layer menu offers 정답
  gt?: ScoreImageDetail | null // what the 정답 layer draws (when it is on)
}

const EMPTY: Cube[] = []
const MAX_SCALE = 32

function fitView(cw: number, ch: number, w: number, h: number): ViewState {
  if (cw <= 0 || ch <= 0) return { s: 1, tx: 0, ty: 0 }
  const pad = 12
  const s = Math.max(0.01, Math.min((cw - 2 * pad) / w, (ch - 2 * pad) / h))
  return { s, tx: (cw - w * s) / 2, ty: (ch - h * s) / 2 }
}

/** PNG colour type from a data URL header: 0 / 4 = greyscale (a mask). */
function isMaskPng(url: string): boolean {
  const i = url.indexOf('base64,')
  if (i < 0) return false
  try {
    const head = atob(url.slice(i + 7, i + 7 + 44))
    const type = head.charCodeAt(25)
    return type === 0 || type === 4
  } catch {
    return false
  }
}

function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null
  if (!el) return false
  const tag = el.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable
}

interface Readout {
  x: number
  y: number
  rgb: Rgb | null
}

export function Viewer(props: ViewerProps) {
  const {
    media,
    sets,
    layers,
    onLayersChange,
    debugLayers = null,
    roi = null,
    highlight = null,
    onHighlight,
    badge,
    tool = null,
    transport = true,
    keyboard = true,
    gtAvailable = false,
    gt = null,
  } = props
  const W = media.width
  const H = media.height
  const isVideo = media.kind === 'video'
  const fps = media.fps ?? 30
  const frameCount = media.frame_count ?? 1

  const rootRef = useRef<HTMLDivElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const [video, setVideo] = useState<HTMLVideoElement | null>(null)

  // ----- view (zoom / pan) -----
  const [size, setSize] = useState({ w: 0, h: 0 })
  useLayoutEffect(() => {
    const el = stageRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => setSize({ w: entry.contentRect.width, h: entry.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  const fit = useMemo(() => fitView(size.w, size.h, W, H), [size.w, size.h, W, H])
  const [ownView, setOwnView] = useState<ViewState | null>(null)
  const controlled = props.view !== undefined
  const linked = props.view ?? null
  const eff = useMemo(
    () => (controlled ? (linked ? fromLinked(linked, fit, size.w, size.h, W, H) : fit) : (ownView ?? fit)),
    [controlled, linked, fit, ownView, size.w, size.h, W, H],
  )

  const latest = useRef({ eff, fit, size, controlled, onViewChange: props.onViewChange })
  useLayoutEffect(() => {
    latest.current = { eff, fit, size, controlled, onViewChange: props.onViewChange }
  })

  const setView = useCallback(
    (v: ViewState | null) => {
      const { fit: f, size: sz, controlled: c, onViewChange } = latest.current
      if (!c) setOwnView(v)
      onViewChange?.(v ? toLinked(v, f, sz.w, sz.h, W, H) : null)
    },
    [W, H],
  )

  const zoomAt = useCallback(
    (px: number, py: number, target: number) => {
      const { eff: v, fit: f } = latest.current
      const s = Math.min(MAX_SCALE, Math.max(Math.min(f.s * 0.25, 1), target))
      setView({ s, tx: px - ((px - v.tx) * s) / v.s, ty: py - ((py - v.ty) * s) / v.s })
    },
    [setView],
  )

  const zoomCentre = useCallback(
    (target: number) => {
      const el = stageRef.current
      if (el) zoomAt(el.clientWidth / 2, el.clientHeight / 2, target)
    },
    [zoomAt],
  )

  useEffect(() => {
    const el = stageRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const unit = e.deltaMode === 1 ? 16 : 1
      const factor = Math.exp(-e.deltaY * unit * (e.ctrlKey ? 0.01 : 0.0015))
      zoomAt(e.clientX - rect.left, e.clientY - rect.top, latest.current.eff.s * factor)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [zoomAt])

  // ----- video clock -----
  const [frame, setFrame] = useState(0)
  const [paused, setPaused] = useState(true)
  const [pendingSeek, setPendingSeek] = useState<number | null>(null)
  // -1 = nothing reported yet, so the first frame on screen (usually 0) is reported too: a viewer
  // that mounts in place of another (Compare's mode switch) must tell its parent where it is.
  const frameRef = useRef(-1)
  const cbs = useRef({ onFrameChange: props.onFrameChange, onPausedChange: props.onPausedChange })
  useLayoutEffect(() => {
    cbs.current = { onFrameChange: props.onFrameChange, onPausedChange: props.onPausedChange }
  })

  useEffect(() => {
    if (!video) return
    const report = (f: number) => {
      const n = Math.max(0, Math.min(frameCount - 1, f))
      setPendingSeek((p) => (p === n ? null : p))
      if (n === frameRef.current) return
      frameRef.current = n
      setFrame(n)
      cbs.current.onFrameChange?.(n)
    }
    let handle = 0
    const hasRvfc = typeof video.requestVideoFrameCallback === 'function'
    const onVideoFrame: VideoFrameRequestCallback = (_now, meta) => {
      report(Math.round(meta.mediaTime * fps))
      handle = video.requestVideoFrameCallback(onVideoFrame)
    }
    const fromTime = () => report(Math.floor(video.currentTime * fps + 1e-3))
    const onPlay = () => {
      setPaused(false)
      cbs.current.onPausedChange?.(false)
    }
    const onPause = () => {
      setPaused(true)
      cbs.current.onPausedChange?.(true)
    }
    if (hasRvfc) handle = video.requestVideoFrameCallback(onVideoFrame)
    else video.addEventListener('timeupdate', fromTime)
    video.addEventListener('loadeddata', fromTime)
    video.addEventListener('seeked', fromTime)
    video.addEventListener('play', onPlay)
    video.addEventListener('pause', onPause)
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) fromTime()
    return () => {
      if (hasRvfc) video.cancelVideoFrameCallback(handle)
      else video.removeEventListener('timeupdate', fromTime)
      video.removeEventListener('loadeddata', fromTime)
      video.removeEventListener('seeked', fromTime)
      video.removeEventListener('play', onPlay)
      video.removeEventListener('pause', onPause)
    }
  }, [video, fps, frameCount])

  const seekFrame = useCallback(
    (n: number) => {
      if (!video) return
      const target = Math.max(0, Math.min(frameCount - 1, Math.round(n)))
      setPendingSeek(target)
      video.currentTime = (target + 0.5) / fps
    },
    [video, fps, frameCount],
  )
  const play = useCallback(() => {
    if (!video) return
    if (frameRef.current >= frameCount - 1) video.currentTime = 0.5 / fps
    void video.play().catch(() => undefined)
  }, [video, fps, frameCount])
  const pause = useCallback(() => video?.pause(), [video])
  const togglePlay = useCallback(() => (video?.paused ? play() : pause()), [video, play, pause])
  const step = useCallback((d: number) => {
    video?.pause()
    seekFrame((pendingSeek ?? Math.max(0, frameRef.current)) + d)
  }, [video, seekFrame, pendingSeek])

  useImperativeHandle(props.ref, () => ({ seekFrame, play, pause }), [seekFrame, play, pause])

  const shownFrame = isVideo ? frame : 0

  // ----- what to draw -----
  const shown = useMemo(() => {
    return sets.map((set) => {
      let rec: FrameRecord | null = recordFor(set.frames, shownFrame, isVideo ? set.span : null)
      if (rec && set.exact && rec.frame !== shownFrame) rec = null
      return { key: set.key, cubes: rec?.cubes ?? EMPTY, dashed: !!set.dashed, tag: set.tag ?? null, at: rec?.frame ?? null }
    })
  }, [sets, shownFrame, isVideo])
  const shownSets: ShownSet[] = useMemo(
    () => shown.map(({ key, cubes, dashed, tag }) => ({ key, cubes, dashed, tag })),
    [shown],
  )
  const primary = shown[0]
  const primaryCubes = primary?.cubes ?? EMPTY

  const debugImage: DebugImage | null = useMemo(() => {
    if (!layers.debug || !debugLayers || debugLayers.length === 0) return null
    const layer = debugLayers.find((l) => l.name === layers.debugName) ?? debugLayers[0]
    return { url: layer.url, isMask: isMaskPng(layer.url) }
  }, [layers.debug, layers.debugName, debugLayers])

  // ----- pixel sampling (inspector, colour picking) -----
  const [inspect, setInspect] = useState(false)
  const [readout, setReadout] = useState<Readout | null>(null)
  const sample = useRef<{ canvas: HTMLCanvasElement | null; data: ImageData | null; key: string }>({
    canvas: null,
    data: null,
    key: '',
  })

  const sampleData = useCallback(
    (force: boolean): ImageData | null => {
      const el: CanvasImageSource | null = isVideo ? video : imgRef.current
      if (!el) return null
      const key = isVideo ? `f${frameRef.current}` : 'img'
      const s = sample.current
      if (s.data && s.key === key && !force) return s.data
      if (isVideo && video && !video.paused && !force) return null
      if (!isVideo && !(imgRef.current?.complete && imgRef.current.naturalWidth > 0)) return null
      const canvas = s.canvas ?? document.createElement('canvas')
      canvas.width = W
      canvas.height = H
      const ctx = canvas.getContext('2d', { willReadFrequently: true })
      if (!ctx) return null
      ctx.drawImage(el, 0, 0, W, H)
      sample.current = { canvas, data: ctx.getImageData(0, 0, W, H), key }
      return sample.current.data
    },
    [isVideo, video, W, H],
  )

  const pixelAt = (data: ImageData, x: number, y: number): Rgb => {
    const i = (y * W + x) * 4
    return [data.data[i], data.data[i + 1], data.data[i + 2]]
  }

  // ----- ROI drafting -----
  const [draft, setDraft] = useState<Pt[]>([])
  const [cursor, setCursor] = useState<Pt | null>(null)
  const toolKey = tool ? (tool.kind === 'roi' ? `roi-${tool.mode}` : 'pick') : ''
  const [draftKey, setDraftKey] = useState(toolKey)
  if (draftKey !== toolKey) {
    setDraftKey(toolKey)
    setDraft([])
  }

  const finishRoi = useCallback(
    (points: Pt[]) => {
      if (tool?.kind !== 'roi' || points.length < 3) return
      tool.onComplete(points.map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10] as Pt))
      setDraft([])
    },
    [tool],
  )

  // ----- pointer -----
  const drag = useRef<{ x: number; y: number; tx: number; ty: number; moved: boolean } | null>(null)
  const hoverHl = useRef(false)

  const toMedia = (e: ReactPointerEvent | PointerEvent | MouseEvent): { px: number; py: number; pt: Pt } => {
    const rect = stageRef.current!.getBoundingClientRect()
    const px = e.clientX - rect.left
    const py = e.clientY - rect.top
    const v = latest.current.eff
    return { px, py, pt: [(px - v.tx) / v.s, (py - v.ty) / v.s] }
  }

  const inside = ([x, y]: Pt) => x >= 0 && y >= 0 && x < W && y < H

  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return
    e.currentTarget.setPointerCapture(e.pointerId)
    const v = latest.current.eff
    drag.current = { x: e.clientX, y: e.clientY, tx: v.tx, ty: v.ty, moved: false }
  }

  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    const d = drag.current
    if (d) {
      const dx = e.clientX - d.x
      const dy = e.clientY - d.y
      if (!d.moved && Math.hypot(dx, dy) > 4) d.moved = true
      if (d.moved) {
        setView({ s: latest.current.eff.s, tx: d.tx + dx, ty: d.ty + dy })
        return
      }
    }
    const { pt } = toMedia(e)
    if (tool?.kind === 'roi') setCursor(pt)
    if (inspect || tool?.kind === 'pick') {
      if (inside(pt)) {
        const x = Math.floor(pt[0])
        const y = Math.floor(pt[1])
        const data = sampleData(false)
        setReadout({ x, y, rgb: data ? pixelAt(data, x, y) : null })
      } else setReadout(null)
    }
    if (!tool && onHighlight) {
      const hit = inside(pt) ? hitTestCubes(primaryCubes, pt) : null
      if (hit !== highlight && (hit !== null || hoverHl.current)) {
        hoverHl.current = hit !== null
        onHighlight(hit)
      }
    }
  }

  const onPointerUp = (e: ReactPointerEvent<HTMLDivElement>) => {
    const d = drag.current
    drag.current = null
    if (!d || d.moved) return
    const { pt } = toMedia(e)
    if (tool?.kind === 'pick') {
      if (!inside(pt)) return
      const data = sampleData(true)
      if (!data) return
      const x0 = Math.max(0, Math.min(W - 5, Math.floor(pt[0]) - 2))
      const y0 = Math.max(0, Math.min(H - 5, Math.floor(pt[1]) - 2))
      const patch = sample.current.canvas!.getContext('2d')!.getImageData(x0, y0, 5, 5)
      tool.onPick(rgbToHex(medianRgb(patch.data)))
      return
    }
    if (tool?.kind === 'roi') {
      const clamped: Pt = [Math.max(0, Math.min(W, pt[0])), Math.max(0, Math.min(H, pt[1]))]
      setDraft((p) => [...p, clamped])
      return
    }
    if (onHighlight) {
      const hit = inside(pt) ? hitTestCubes(primaryCubes, pt) : null
      hoverHl.current = false
      onHighlight(hit)
    }
  }

  const onDoubleClick = () => {
    if (tool?.kind !== 'roi') return
    // The two clicks of a double-click already added the same point twice.
    const pts = draft.filter((p, i) => i === 0 || Math.hypot(p[0] - draft[i - 1][0], p[1] - draft[i - 1][1]) > 2)
    finishRoi(pts)
  }

  const onPointerLeave = () => {
    setReadout(null)
    setCursor(null)
    if (hoverHl.current && onHighlight) {
      hoverHl.current = false
      onHighlight(null)
    }
  }

  // ----- fullscreen -----
  const [fullscreen, setFullscreen] = useState(false)
  useEffect(() => {
    const onChange = () => setFullscreen(document.fullscreenElement === rootRef.current)
    document.addEventListener('fullscreenchange', onChange)
    return () => document.removeEventListener('fullscreenchange', onChange)
  }, [])
  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement) void document.exitFullscreen()
    else void rootRef.current?.requestFullscreen().catch(() => undefined)
  }, [])

  // ----- keyboard -----
  const keyState = useRef({ draft, tool, finishRoi, togglePlay, step, zoomCentre, toggleFullscreen, setView })
  useLayoutEffect(() => {
    keyState.current = { draft, tool, finishRoi, togglePlay, step, zoomCentre, toggleFullscreen, setView }
  })
  useEffect(() => {
    if (!keyboard) return
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || isTyping(e.target)) return
      // A modal dialog (the JSON editor) owns the keyboard: the viewer behind it must not step
      // frames, zoom or go full screen (found by exploratory QA).
      if (document.querySelector('dialog[open]')) return
      const k = keyState.current
      if (k.tool?.kind === 'roi') {
        if (e.key === 'Enter') {
          e.preventDefault()
          k.finishRoi(k.draft)
          return
        }
        if (e.key === 'Backspace') {
          e.preventDefault()
          setDraft((p) => p.slice(0, -1))
          return
        }
      }
      if (e.key === 'Escape' && k.tool) {
        e.preventDefault()
        setDraft([])
        k.tool.onCancel()
        return
      }
      const onButton = (e.target as HTMLElement | null)?.closest('button, a, [role="button"]')
      switch (e.key) {
        case '0':
          k.setView(null)
          break
        case '1':
          k.zoomCentre(1)
          break
        case '+':
        case '=':
          k.zoomCentre(latest.current.eff.s * 1.25)
          break
        case '-':
          k.zoomCentre(latest.current.eff.s / 1.25)
          break
        case 'f':
        case 'F':
          k.toggleFullscreen()
          break
        case ' ':
          if (!isVideo || onButton) return
          k.togglePlay()
          break
        case 'ArrowLeft':
          if (!isVideo) return
          k.step(e.shiftKey ? -10 : -1)
          break
        case 'ArrowRight':
          if (!isVideo) return
          k.step(e.shiftKey ? 10 : 1)
          break
        default:
          return
      }
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [keyboard, isVideo])

  // ----- render -----
  const pctLabel = `${Math.round(eff.s * 100)}%`
  const span = sets[0]?.span ?? null
  const processedNote =
    isVideo && primary && sets[0] && !sets[0].exact
      ? primary.at === null
        ? span && outsideSpan(span, shownFrame)
          ? '이 구간은 처리하지 않음'
          : sets[0].frames.length > 0
            ? '이 프레임의 처리 결과 없음'
            : null
        : primary.at !== shownFrame
          ? `처리 프레임 f ${primary.at}`
          : null
      : null
  const marks = useMemo(
    () => (isVideo && sets[0] && !sets[0].exact ? sets[0].frames.map((f) => f.frame) : null),
    [isVideo, sets],
  )

  return (
    <div className={`viewer${fullscreen ? ' is-fullscreen' : ''}`} ref={rootRef}>
      <div
        className={tool ? 'viewer-stage tool' : 'viewer-stage'}
        ref={stageRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => (drag.current = null)}
        onPointerLeave={onPointerLeave}
        onDoubleClick={onDoubleClick}
        role="img"
        aria-label={`${media.name} 결과 보기`}
      >
        <div className="media-frame reg" style={{ left: eff.tx - 1, top: eff.ty - 1, width: W * eff.s + 2, height: H * eff.s + 2 }} />
        <div
          className={`viewer-content${eff.s >= 2 ? ' pixelated' : ''}`}
          style={{ width: W, height: H, transform: `translate(${eff.tx}px, ${eff.ty}px) scale(${eff.s})` }}
        >
          {isVideo ? (
            <video ref={setVideo} src={media.file_url} muted playsInline preload="auto" />
          ) : (
            <img ref={imgRef} src={media.file_url} alt="" draggable={false} />
          )}
          {debugImage && !debugImage.isMask ? (
            <img
              className="debug-img"
              src={debugImage.url}
              alt=""
              draggable={false}
              style={{ opacity: layers.debugOpacity, mixBlendMode: layers.debugBlend }}
            />
          ) : null}
          <Overlay
            width={W}
            height={H}
            scale={eff.s}
            sets={shownSets}
            layers={layers}
            roi={roi}
            debug={debugImage}
            highlight={highlight}
            draft={tool?.kind === 'roi' ? { mode: tool.mode, points: draft, cursor } : null}
            gt={gtAvailable ? gt : null}
          />
        </div>
        {eff.s >= 8 ? (
          <div
            className="pixel-grid"
            style={{ left: eff.tx, top: eff.ty, width: W * eff.s, height: H * eff.s, backgroundSize: `${eff.s}px ${eff.s}px` }}
          />
        ) : null}
      </div>

      <div className="viewer-badge">
        {badge}
        {processedNote ? <span className="viewer-note num">{processedNote}</span> : null}
      </div>

      <div className="viewer-tools">
        <span className="zoom-readout num" aria-live="polite">
          {pctLabel}
        </span>
        <button type="button" className="btn btn-sm btn-quiet" onClick={() => setView(null)} title="화면에 맞춤 (0)">
          맞춤
        </button>
        <button type="button" className="btn btn-sm btn-quiet" onClick={() => zoomCentre(1)} title="원본 크기 (1)">
          100%
        </button>
        <LayerMenu prefs={layers} onChange={onLayersChange} debugLayers={debugLayers} gtAvailable={gtAvailable} />
        <button
          type="button"
          className="icon-btn"
          aria-pressed={inspect}
          aria-label="픽셀 정보"
          title="픽셀 정보 (x, y, RGB, HSV, Lab)"
          onClick={() => setInspect((v) => !v)}
        >
          <Crosshair size={16} strokeWidth={1.75} />
        </button>
        <button
          type="button"
          className="icon-btn"
          aria-label={fullscreen ? '전체 화면 끝내기' : '전체 화면'}
          title="전체 화면 (F)"
          onClick={toggleFullscreen}
        >
          {fullscreen ? <Minimize size={16} strokeWidth={1.75} /> : <Maximize size={16} strokeWidth={1.75} />}
        </button>
      </div>

      {tool ? (
        <div className="tool-hint" role="status">
          {tool.kind === 'pick' ? (
            <>
              <strong>{tool.label}</strong> 이미지를 클릭하면 5×5 픽셀의 중앙값을 씁니다. <span className="kbd">Esc</span> 취소
            </>
          ) : (
            <>
              <strong>{tool.mode === 'include' ? '포함 영역' : '제외 영역'} 그리기</strong> 클릭으로 점 추가 · 점 {draft.length}개 ·{' '}
              <span className="kbd">Enter</span>/더블클릭 완료 · <span className="kbd">⌫</span> 되돌리기 · <span className="kbd">Esc</span> 취소
            </>
          )}
        </div>
      ) : null}

      {gtAvailable && layers.gt && gt ? <GtCard d={gt} highlight={highlight} /> : null}

      {(inspect || tool?.kind === 'pick') && readout ? (
        <div className="inspector num" aria-live="off">
          <div className="insp-row">
            <span className="faint">x</span> {readout.x} <span className="faint">y</span> {readout.y}
            {readout.rgb ? <i className="insp-swatch" style={{ background: rgbToHex(readout.rgb) }} /> : null}
          </div>
          {readout.rgb ? (
            <>
              <div className="insp-row">
                <span className="insp-k">RGB</span> {readout.rgb.join(' ')}
              </div>
              <div className="insp-row">
                <span className="insp-k">HSV</span> {rgbToHsvCv(readout.rgb).join(' ')}
              </div>
              <div className="insp-row">
                <span className="insp-k">Lab</span>{' '}
                {rgbToLabCv(readout.rgb)
                  .map((v) => v.toFixed(1))
                  .join(' ')}
              </div>
            </>
          ) : (
            <div className="insp-row faint">재생 중 — 멈추면 색을 읽습니다</div>
          )}
        </div>
      ) : null}

      {isVideo && transport ? (
        <Transport
          frame={pendingSeek ?? frame}
          frameCount={frameCount}
          fps={fps}
          paused={paused}
          marks={marks}
          onToggle={togglePlay}
          onStep={step}
          onSeek={(n) => {
            video?.pause()
            seekFrame(n)
          }}
        />
      ) : null}
    </div>
  )
}
