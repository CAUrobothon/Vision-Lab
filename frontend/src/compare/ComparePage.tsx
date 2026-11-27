// 비교 (Compare) — `/compare?a=<run>&b=<run>&mode=split|overlay`.
// Two viewers side by side with linked zoom and pan (in media-normalised units, so different
// media sizes line up) and, when both are videos, a linked playhead; or one viewer with A solid
// and B dashed. The panel lists parameter differences, cubes per frame, colour counts and timing.
import { useCallback, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Link, useSearchParams } from 'react-router'
import { AlertTriangle, ExternalLink } from 'lucide-react'
import { useMedia, useModels, useRun, useRunFrames } from '../api/queries'
import { CUBE_COLORS, type FrameRecord, type Media, type ModelInfo, type Run } from '../api/types'
import { countColors, recordFor, spanOf } from '../lib/cubes'
import { formatMs, formatNumber } from '../lib/format'
import { formatParamValue, sameValue } from '../lib/params'
import { ClassSwatch, StatusMark } from '../ui/controls'
import { useLayerPrefs } from '../viewer/layers'
import { Viewer, type LinkedView, type OverlaySet, type ViewerHandle } from '../viewer/Viewer'
import './compare.css'

const NO_FRAMES: FrameRecord[] = []

function num(v: string | null): number | null {
  const n = v === null ? NaN : Number(v)
  return Number.isInteger(n) && n > 0 ? n : null
}

function delta(a: number | null | undefined, b: number | null | undefined, digits = 1): string {
  if (a === null || a === undefined || b === null || b === undefined) return ''
  const d = b - a
  if (Math.abs(d) < 10 ** -digits / 2) return '='
  return `${d > 0 ? '+' : '−'}${Math.abs(d).toFixed(digits)}`
}

function Row({ label, a, b, d }: { label: ReactNode; a: ReactNode; b: ReactNode; d?: string }) {
  return (
    <tr>
      <th scope="row">{label}</th>
      <td className="r num">{a}</td>
      <td className="r num">{b}</td>
      <td className="r num faint">{d ?? ''}</td>
    </tr>
  )
}

function ParamDiff({ a, b, models }: { a: Run; b: Run; models: ModelInfo[] }) {
  const ma = models.find((m) => m.key === a.model_key)
  const mb = models.find((m) => m.key === b.model_key)
  const sameModel = a.model_key === b.model_key
  const keys = sameModel
    ? [...new Set([...Object.keys(a.params), ...Object.keys(b.params)])]
    : Object.keys(a.params).filter((k) => k in b.params)
  const diffs = keys.filter((k) => !sameValue(a.params[k], b.params[k]))
  const spec = (k: string) => ma?.params.find((s) => s.name === k) ?? mb?.params.find((s) => s.name === k)
  return (
    <section className="cmp-section">
      <h3 className="cmp-title">파라미터 차이</h3>
      {!sameModel ? (
        <p className="cmp-note">
          모델이 다릅니다: A {a.model_name}, B {b.model_name}. 이름이 같은 파라미터만 나란히 봅니다.
        </p>
      ) : null}
      {diffs.length === 0 ? (
        <p className="faint cmp-note">{sameModel ? '파라미터가 같습니다.' : '이름이 같은 파라미터는 값도 같습니다.'}</p>
      ) : (
        <table className="table cmp-table">
          <thead>
            <tr>
              <th>이름</th>
              <th className="r">A</th>
              <th className="r">B</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {diffs.map((k) => (
              <tr key={k}>
                <th scope="row" title={k}>
                  {spec(k)?.label ?? k}
                </th>
                <td className="r num">{k in a.params ? formatParamValue(spec(k), a.params[k]) : '—'}</td>
                <td className="r num">{k in b.params ? formatParamValue(spec(k), b.params[k]) : '—'}</td>
                <td />
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}

function RunCard({ tag, run, media }: { tag: 'A' | 'B'; run: Run; media: Media | undefined }) {
  return (
    <div className="cmp-run">
      <span className={`cmp-tag ${tag === 'A' ? 'solid' : 'dashed'}`}>{tag}</span>
      <div className="cmp-run-text">
        <Link to={`/?media=${run.media_id}&run=${run.id}`} className="cmp-run-title">
          <span className="num">#{run.id}</span> {run.model_name} <ExternalLink size={11} strokeWidth={1.75} />
        </Link>
        <span className="faint">
          {media?.name ?? run.media_name} {run.status !== 'done' ? <StatusMark status={run.status} /> : null}
        </span>
      </div>
    </div>
  )
}

export default function ComparePage() {
  const [sp, setSp] = useSearchParams()
  const aId = num(sp.get('a'))
  const bId = num(sp.get('b'))
  const runA = useRun(aId)
  const runB = useRun(bId)
  const a = runA.data ?? null
  const b = runB.data ?? null
  const framesA = useRunFrames(a)
  const framesB = useRunFrames(b)
  const mediaA = useMedia(a?.media_id ?? null)
  const mediaB = useMedia(b?.media_id ?? null)
  const models = useModels()
  const [layers, setLayers] = useLayerPrefs()
  const sameMedia = !!a && !!b && a.media_id === b.media_id
  const mode = sp.get('mode') === 'overlay' && sameMedia ? 'overlay' : 'split'
  const [view, setView] = useState<LinkedView | null>(null)
  const [frame, setFrame] = useState(0) // frame on screen in A (or in the overlay viewer)
  const [ownFrameB, setOwnFrameB] = useState(0) // frame on screen in B when B has its own playhead
  const refB = useRef<ViewerHandle>(null)
  const frameB = useRef(0)
  const ma = mediaA.data
  const mb = mediaB.data
  // The playhead is linked only when both are videos; otherwise a video keeps its own transport.
  const linkPlayhead = ma?.kind === 'video' && mb?.kind === 'video'

  const fa = framesA.data?.frames ?? NO_FRAMES
  const fb = framesB.data?.frames ?? NO_FRAMES
  const spanA = useMemo(() => (a ? spanOf(a) : null), [a])
  const spanB = useMemo(() => (b ? spanOf(b) : null), [b])
  const setsA: OverlaySet[] = useMemo(() => [{ key: 'a', frames: fa, span: spanA }], [fa, spanA])
  const setsB: OverlaySet[] = useMemo(() => [{ key: 'b', frames: fb, span: spanB, dashed: true }], [fb, spanB])
  const setsOverlay: OverlaySet[] = useMemo(
    () => [
      { key: 'a', frames: fa, span: spanA, tag: 'A' },
      { key: 'b', frames: fb, span: spanB, dashed: true, tag: 'B' },
    ],
    [fa, fb, spanA, spanB],
  )

  // A drives the playhead; B follows: play/pause mirrored, the same frame when paused, and a
  // re-seek while playing only when B drifts more than 2 frames.
  const playing = useRef(false)
  const frameA = useRef(0)
  const link = useRef(linkPlayhead)
  useLayoutEffect(() => {
    link.current = linkPlayhead
  })
  const onFrameA = useCallback((f: number) => {
    frameA.current = f
    setFrame(f)
    if (link.current && (!playing.current || Math.abs(frameB.current - f) > 2)) refB.current?.seekFrame(f)
  }, [])
  const onPausedA = useCallback((paused: boolean) => {
    playing.current = !paused
    if (!link.current) return
    if (paused) {
      refB.current?.pause()
      refB.current?.seekFrame(frameA.current)
    } else refB.current?.play()
  }, [])
  const onFrameB = useCallback((f: number) => {
    frameB.current = f
    if (!link.current) setOwnFrameB(f) // linked: B is described at A's frame, no extra render
  }, [])

  if (aId === null || bId === null)
    return (
      <div className="page">
        <div className="empty">
          <strong>비교할 기록 두 개를 고르세요</strong>
          <span>
            <Link to="/runs">기록</Link>에서 두 개를 체크하거나, 작업대 아래 기록 띠에서 두 개를 체크한 뒤 “비교”를 누르세요.
          </span>
        </div>
      </div>
    )

  if (runA.isError || runB.isError)
    return (
      <div className="page">
        <div className="empty">
          <strong>기록을 불러올 수 없습니다</strong>
          <span className="error-text">{(runA.error ?? runB.error)?.message}</span>
          <Link to="/runs">기록으로 가기</Link>
        </div>
      </div>
    )

  const aVideo = ma?.kind === 'video'
  const bVideo = mb?.kind === 'video'
  const isVideo = aVideo || bVideo
  // In overlay mode (same media) and with a linked playhead both are described at A's frame.
  const shownA = aVideo ? frame : 0
  const shownB = bVideo ? (mode === 'overlay' || linkPlayhead ? frame : ownFrameB) : 0
  const recA = recordFor(fa, shownA, aVideo ? spanA : null)
  const recB = recordFor(fb, shownB, bVideo ? spanB : null)
  const countA = recA ? countColors(recA.cubes) : null
  const countB = recB ? countColors(recB.cubes) : null

  return (
    <div className={`compare mode-${mode}`}>
      <div className="cmp-bar">
        {a ? <RunCard tag="A" run={a} media={ma} /> : <span className="faint">A 불러오는 중…</span>}
        {b ? <RunCard tag="B" run={b} media={mb} /> : <span className="faint">B 불러오는 중…</span>}
        <span className="cmp-spacer" />
        {!sameMedia && a && b ? (
          <span className="cmp-warn">
            <AlertTriangle size={13} strokeWidth={1.75} /> 서로 다른 미디어의 기록입니다 — 나란히만 볼 수 있습니다
          </span>
        ) : null}
        <div className="segmented" role="group" aria-label="보기 방식">
          <button type="button" aria-pressed={mode === 'split'} onClick={() => setSp((p) => (p.delete('mode'), p), { replace: true })}>
            나란히
          </button>
          <button
            type="button"
            aria-pressed={mode === 'overlay'}
            disabled={!sameMedia}
            onClick={() => setSp((p) => (p.set('mode', 'overlay'), p), { replace: true })}
          >
            겹쳐 보기
          </button>
        </div>
      </div>

      <div className="cmp-viewers">
        {mode === 'overlay' && ma ? (
          <Viewer
            key={`o-${ma.id}`}
            media={ma}
            sets={setsOverlay}
            layers={layers}
            onLayersChange={setLayers}
            onFrameChange={setFrame}
            badge={
              <div className="mode-badge cmp-badge">
                <span className="cmp-legend solid">A #{a?.id}</span>
                <span className="cmp-legend dashed">B #{b?.id}</span>
              </div>
            }
          />
        ) : (
          <>
            <div className="cmp-pane">
              {ma ? (
                <Viewer
                  key={`a-${ma.id}`}
                  media={ma}
                  sets={setsA}
                  layers={layers}
                  onLayersChange={setLayers}
                  view={view}
                  onViewChange={setView}
                  onFrameChange={onFrameA}
                  onPausedChange={onPausedA}
                  badge={<div className="mode-badge cmp-badge">A · #{a?.id} {a?.model_name}</div>}
                />
              ) : (
                <div className="viewer-empty" />
              )}
            </div>
            <div className="cmp-pane">
              {mb ? (
                <Viewer
                  key={`b-${mb.id}`}
                  ref={refB}
                  media={mb}
                  sets={setsB}
                  layers={layers}
                  onLayersChange={setLayers}
                  view={view}
                  onViewChange={setView}
                  onFrameChange={onFrameB}
                  transport={!linkPlayhead}
                  keyboard={false}
                  badge={<div className="mode-badge cmp-badge">B · #{b?.id} {b?.model_name}</div>}
                />
              ) : (
                <div className="viewer-empty" />
              )}
            </div>
          </>
        )}
      </div>

      <aside className="cmp-panel" aria-label="차이">
        {a && b ? (
          <>
            <section className="cmp-section">
              <h3 className="cmp-title">
                {isVideo ? (
                  <>
                    현재 프레임{' '}
                    <span className="num faint">
                      {aVideo && bVideo && shownA !== shownB ? `A f ${shownA} · B f ${shownB}` : `f ${aVideo ? shownA : shownB}`}
                    </span>
                  </>
                ) : (
                  '검출'
                )}
              </h3>
              <table className="table cmp-table">
                <thead>
                  <tr>
                    <th />
                    <th className="r">A</th>
                    <th className="r">B</th>
                    <th className="r">B−A</th>
                  </tr>
                </thead>
                <tbody>
                  <Row label="큐브" a={recA?.cubes.length ?? '—'} b={recB?.cubes.length ?? '—'} d={delta(recA?.cubes.length, recB?.cubes.length, 0)} />
                  {CUBE_COLORS.map((c) => (
                    <Row key={c} label={<ClassSwatch color={c} withName />} a={countA?.[c] ?? '—'} b={countB?.[c] ?? '—'} d={delta(countA?.[c], countB?.[c], 0)} />
                  ))}
                  {isVideo ? (
                    <Row label="처리 프레임" a={recA ? `f ${recA.frame}` : '—'} b={recB ? `f ${recB.frame}` : '—'} />
                  ) : null}
                </tbody>
              </table>
            </section>

            <section className="cmp-section">
              <h3 className="cmp-title">프레임당 평균</h3>
              <table className="table cmp-table">
                <tbody>
                  <Row label="큐브" a={formatNumber(a.summary?.cubes_mean, 2)} b={formatNumber(b.summary?.cubes_mean, 2)} d={delta(a.summary?.cubes_mean, b.summary?.cubes_mean, 2)} />
                  <Row
                    label="최소–최대"
                    a={a.summary ? `${a.summary.cubes_min}–${a.summary.cubes_max}` : '—'}
                    b={b.summary ? `${b.summary.cubes_min}–${b.summary.cubes_max}` : '—'}
                  />
                  {CUBE_COLORS.map((c) => (
                    <Row
                      key={c}
                      label={<ClassSwatch color={c} withName />}
                      a={formatNumber(a.summary?.color_mean[c], 2)}
                      b={formatNumber(b.summary?.color_mean[c], 2)}
                      d={delta(a.summary?.color_mean[c], b.summary?.color_mean[c], 2)}
                    />
                  ))}
                  <Row label="처리한 프레임" a={a.frames_done} b={b.frames_done} />
                </tbody>
              </table>
            </section>

            <section className="cmp-section">
              <h3 className="cmp-title">시간</h3>
              <table className="table cmp-table">
                <tbody>
                  <Row label="p50" a={formatMs(a.timing?.p50_ms)} b={formatMs(b.timing?.p50_ms)} d={delta(a.timing?.p50_ms, b.timing?.p50_ms)} />
                  <Row label="p95" a={formatMs(a.timing?.p95_ms)} b={formatMs(b.timing?.p95_ms)} d={delta(a.timing?.p95_ms, b.timing?.p95_ms)} />
                  <Row label="첫 프레임" a={formatMs(a.timing?.first_ms)} b={formatMs(b.timing?.first_ms)} d={delta(a.timing?.first_ms, b.timing?.first_ms)} />
                  <Row label="장치" a={a.device ?? '—'} b={b.device ?? '—'} />
                </tbody>
              </table>
            </section>

            <ParamDiff a={a} b={b} models={models.data ?? []} />
          </>
        ) : (
          <p className="faint cmp-note">불러오는 중…</p>
        )}
      </aside>
    </div>
  )
}
