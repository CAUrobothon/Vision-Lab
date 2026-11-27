// 작업대 (Workbench) — `/?media=<id>&run=<id>`; `&dataset=<id>&gt=1` (the 채점 page opening a run)
// also puts the rail on that dataset and turns the 정답 layer on.
// Left: media rail. Centre: the viewer and this media's runs. Right: model, presets, params,
// actions and the result of what the viewer shows. Pipelines preview live on every change;
// "기록으로 저장" / "실행" create runs.
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useSearchParams } from 'react-router'
import { FolderOpen, PanelLeft, Sparkles, Upload, X } from 'lucide-react'
import { api, ApiError } from '../api/client'
import { useFeature } from '../api/features'
import {
  invalidateGt,
  invalidateRuns,
  isActive,
  keys,
  queryClient,
  useMedia,
  useMediaGt,
  useMediaList,
  useModels,
  useRun,
  useRunFrames,
  useScoreFrame,
  useScoreImage,
} from '../api/queries'
import type { DebugLayer, FrameRecord, Media, MediaUploadResult, ModelInfo, ParamValue, Params, PreviewRequest, RoiPolygon, Run } from '../api/types'
import { outsideSpan, recordFor, spanOf } from '../lib/cubes'
import { joinFolder } from '../lib/folders'
import { idParam } from '../lib/format'
import { mergeParams, paramOverrides } from '../lib/params'
import { loadJson, saveJson } from '../lib/storage'
import { InlineText, StatusMark, TagEditor } from '../ui/controls'
import { useToast } from '../ui/toast'
import { useLayerPrefs } from '../viewer/layers'
import { Viewer, type OverlaySet, type ViewerTool } from '../viewer/Viewer'
import { railDataset, railFolder, rememberRail } from './DatasetPanel'
import { ACCEPT, MediaRail, mediaMeta } from './MediaRail'
import { ModelPicker } from './ModelPicker'
import { ParamControls, type ParamTool } from './ParamControls'
import { ParamJsonButton } from './ParamJson'
import { PresetBar } from './PresetBar'
import { ResultPanel, type LiveScore, type ResultTiming } from './ResultPanel'
import { useGtPick, useSettledCriteria } from '../score/criteria'
import { RunActions, type RunRequest } from './RunActions'
import { RunStrip } from './RunStrip'
import { placeFiles, plainFiles, usePasteFiles, useUploads, useWindowFileDrop, type PickedFile } from './uploads'
import { usePreview } from './usePreview'
import './workbench.css'
import { josa, named } from '../lib/josa'

const NO_FRAMES: FrameRecord[] = []
const NO_DEBUG: DebugLayer[] = []

/** The ROI value inside a params object (the first list of {mode, points}). */
function roiOf(params: Params | undefined, model: ModelInfo | undefined): RoiPolygon[] | null {
  if (!params) return null
  const spec = model?.params.find((s) => s.type === 'roi')
  if (spec) return (params[spec.name] as RoiPolygon[] | undefined) ?? null
  const v = Object.values(params).find((x) => Array.isArray(x) && x.length > 0 && typeof x[0] === 'object' && x[0] !== null && 'points' in x[0])
  return (v as RoiPolygon[] | undefined) ?? null
}

function EmptyViewer({ libraryEmpty, onFiles, onSamples, samplesPending }: { libraryEmpty: boolean; onFiles: (f: File[]) => void; onSamples: () => void; samplesPending: boolean }) {
  const fileRef = useRef<HTMLInputElement>(null)
  if (!libraryEmpty)
    return (
      <div className="viewer-empty">
        <p className="faint">왼쪽 목록에서 미디어를 고르세요.</p>
      </div>
    )
  return (
    <div className="viewer-empty">
      <div className="dropzone reg">
        <Upload size={28} strokeWidth={1.5} />
        <strong>이미지나 영상을 여기에 끌어다 놓으세요</strong>
        <p className="muted">
          파일 고르기, 또는 <span className="kbd">⌘V</span>로 클립보드 이미지를 붙여넣어도 됩니다. 영상은 프레임 번호를 맞추려고 H.264로 다시 인코딩합니다.
        </p>
        <div className="dropzone-actions">
          <button type="button" className="btn btn-primary" onClick={() => fileRef.current?.click()}>
            <FolderOpen size={14} strokeWidth={1.75} /> 파일 고르기
          </button>
          <button type="button" className="btn" onClick={onSamples} disabled={samplesPending}>
            <Sparkles size={14} strokeWidth={1.75} /> {samplesPending ? '만드는 중…' : '합성 샘플 넣기'}
          </button>
        </div>
        <input
          ref={fileRef}
          type="file"
          multiple
          accept={ACCEPT}
          hidden
          onChange={(e) => {
            const files = Array.from(e.target.files ?? [])
            e.target.value = ''
            if (files.length) onFiles(files)
          }}
        />
      </div>
    </div>
  )
}

function MediaBar({ media, onToggleRail }: { media: Media; onToggleRail: () => void }) {
  const toast = useToast()
  const patch = useMutation({
    mutationFn: (p: { name?: string; tags?: string[] }) => api.patchMedia(media.id, p),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: keys.mediaAll }),
    onError: (e: Error) => toast.error(e.message),
  })
  return (
    <div className="media-bar">
      <button type="button" className="icon-btn rail-toggle" aria-label="미디어 목록" title="미디어 목록" onClick={onToggleRail}>
        <PanelLeft size={16} strokeWidth={1.75} />
      </button>
      <InlineText value={media.name} placeholder="이름 없음" label="미디어 이름" onSave={(name) => name && patch.mutate({ name })} />
      <span className="faint num media-bar-meta">{mediaMeta(media)}</span>
      <TagEditor tags={media.tags} onChange={(tags) => patch.mutate({ tags })} />
    </div>
  )
}

export default function WorkbenchPage() {
  const toast = useToast()
  const [search, setSearch] = useSearchParams()
  const [railOpen, setRailOpen] = useState(false)
  const mediaId = idParam(search.get('media'))
  const runId = idParam(search.get('run'))
  const rerunId = idParam(search.get('rerun'))
  // "&dataset=<id>&gt=1": shown from the first render; the effect under "datasets" keeps them and
  // takes them out of the URL.
  const intentDataset = idParam(search.get('dataset'))
  const intentGt = search.get('gt') === '1'

  const models = useModels()
  const allMedia = useMediaList({})
  const mediaQ = useMedia(mediaId)
  const media = mediaQ.data ?? null
  const runQ = useRun(runId)
  const run = runQ.data && runQ.data.media_id === mediaId ? runQ.data : null
  const framesQ = useRunFrames(run)
  const [storedLayers, setLayers] = useLayerPrefs()
  const layers = useMemo(() => (intentGt && !storedLayers.gt ? { ...storedLayers, gt: true } : storedLayers), [intentGt, storedLayers])

  const selectMedia = useCallback(
    (id: number | null) => setSearch(id === null ? {} : { media: String(id) }),
    [setSearch],
  )
  const pickMedia = useCallback(
    (id: number | null) => {
      selectMedia(id)
      setRailOpen(false)
    },
    [selectMedia],
  )
  const selectRun = useCallback(
    (id: number | null) =>
      setSearch((prev) => {
        const next = new URLSearchParams(prev)
        if (id === null) next.delete('run')
        else next.set('run', String(id))
        return next
      }),
    [setSearch],
  )

  // Open the newest media when nothing is selected; drop a selection that no longer exists. Only a
  // 404 means "gone": a failed request while the server restarts must keep the selection.
  useEffect(() => {
    if (mediaId === null && rerunId === null && allMedia.data && allMedia.data.length > 0)
      setSearch({ media: String(allMedia.data[0].id) }, { replace: true })
  }, [mediaId, rerunId, allMedia.data, setSearch])
  // A run deleted elsewhere (a batch, the runs page, another tab): back to the live preview
  // instead of drawing its stale cubes (found by exploratory QA).
  const runGone = runQ.error instanceof ApiError && runQ.error.status === 404
  useEffect(() => {
    if (runId !== null && runGone) {
      selectRun(null)
      toast.ok(`기록 #${runId}${josa(String(runId), '이/가')} 지워져 미리보기로 돌아왔습니다`)
    }
  }, [runId, runGone, selectRun, toast])
  const mediaGone = mediaQ.error instanceof ApiError && mediaQ.error.status === 404
  useEffect(() => {
    if (mediaId !== null && mediaGone) setSearch({}, { replace: true })
  }, [mediaId, mediaGone, setSearch])

  // ----- model and params (last model and per-model params remembered locally) -----
  const [modelKey, setModelKey] = useState<string | null>(() => loadJson<string | null>('model', null))
  const modelList = useMemo(() => models.data ?? [], [models.data])
  const model =
    modelList.find((m) => m.key === modelKey && m.state === 'ready') ??
    modelList.find((m) => m.kind === 'pipeline' && m.state === 'ready') ??
    null
  const [paramStore, setParamStore] = useState<Record<string, Params>>({})
  const params = useMemo(
    () => (model ? mergeParams(model.params, paramStore[model.key] ?? loadJson<Params | null>(`params.${model.key}`, null)) : {}),
    [model, paramStore],
  )
  const [parentRunId, setParentRunIdRaw] = useState<number | null>(null)
  const [parentFor, setParentFor] = useState<number | null>(null) // the media the parent run belongs to
  const setParentRunId = useCallback(
    (id: number | null, media: number | null = null) => {
      setParentRunIdRaw(id)
      setParentFor(id === null ? null : media)
    },
    [],
  )

  const latest = useRef({ model, params, runId })
  useLayoutEffect(() => {
    latest.current = { model, params, runId }
  })

  // Only the values that differ from the defaults are stored, so a later change of a default
  // reaches every param the user never touched.
  const writeParams = useCallback((m: ModelInfo, next: Params) => {
    setParamStore((prev) => ({ ...prev, [m.key]: next }))
    saveJson(`params.${m.key}`, paramOverrides(m.params, next))
  }, [])

  const leaveRun = useCallback(() => {
    if (latest.current.runId !== null) selectRun(null)
  }, [selectRun])

  const changeParam = useCallback(
    (name: string, value: ParamValue) => {
      const { model: m, params: p } = latest.current
      if (!m) return
      writeParams(m, { ...p, [name]: value })
      leaveRun()
    },
    [writeParams, leaveRun],
  )
  const resetGroup = useCallback(
    (names: string[]) => {
      const { model: m, params: p } = latest.current
      if (!m) return
      const next = { ...p }
      for (const s of m.params) if (names.includes(s.name)) next[s.name] = s.default
      writeParams(m, next)
      leaveRun()
    },
    [writeParams, leaveRun],
  )
  const applyPreset = useCallback(
    (p: Params) => {
      const m = latest.current.model
      if (!m) return
      writeParams(m, mergeParams(m.params, p))
      leaveRun()
    },
    [writeParams, leaveRun],
  )
  // Current values of any model (the JSON dialog may be given JSON for another one).
  const paramsFor = useCallback(
    (m: ModelInfo) => mergeParams(m.params, paramStore[m.key] ?? loadJson<Params | null>(`params.${m.key}`, null)),
    [paramStore],
  )
  // JSON from a chat assistant, resolved by the server and merged by the dialog: switch model if
  // it names another, then set exactly those values.
  const applyJson = useCallback(
    (m: ModelInfo, next: Params) => {
      if (m.key !== latest.current.model?.key) {
        setModelKey(m.key)
        saveJson('model', m.key)
        setParamTool(null)
        setParentRunId(null)
      }
      writeParams(m, next)
      leaveRun()
    },
    [writeParams, leaveRun],
  )
  const chooseModel = (key: string) => {
    setModelKey(key)
    saveJson('model', key)
    setParamTool(null)
    setParentRunId(null) // a re-run keeps the run's model
    leaveRun()
  }

  const rerunFrom = useCallback(
    (r: Run) => {
      const m = modelList.find((x) => x.key === r.model_key)
      if (!m) {
        toast.error(`모델 ${named(r.model_name, '이/가')} 지워져서 같은 파라미터로 다시 실행할 수 없습니다`)
        return
      }
      setModelKey(m.key)
      saveJson('model', m.key)
      writeParams(m, mergeParams(m.params, r.params))
      setParentRunId(r.id, r.media_id)
    },
    [modelList, toast, writeParams],
  )

  // "/?media=..&rerun=<id>" from the history page.
  const rerunQ = useRun(rerunId)
  useEffect(() => {
    if (!rerunQ.data || modelList.length === 0) return
    rerunFrom(rerunQ.data)
    setSearch({ media: String(rerunQ.data.media_id) }, { replace: true })
  }, [rerunQ.data, modelList.length, rerunFrom, setSearch])

  // ----- per-media view state -----
  const [frame, setFrame] = useState(0)
  const [paused, setPaused] = useState(true)
  const [highlight, setHighlight] = useState<number | null>(null)
  const [paramTool, setParamTool] = useState<ParamTool>(null)
  const [checked, setChecked] = useState<number[]>([])
  const [stateFor, setStateFor] = useState(mediaId)
  if (stateFor !== mediaId) {
    setStateFor(mediaId)
    setFrame(0)
    setPaused(true)
    setHighlight(null)
    setParamTool(null)
    setChecked([])
    // "Re-run with changes" belongs to the run's own media (found by exploratory QA: the link
    // survived a media switch and the new run was saved with a foreign parent).
    if (parentRunId !== null && parentFor !== mediaId) setParentRunId(null)
  }

  // ----- preview -----
  const isVideo = media?.kind === 'video'
  const previewReq: PreviewRequest | null =
    media && model?.kind === 'pipeline' && runId === null && (!isVideo || paused)
      ? { media_id: media.id, frame: isVideo ? frame : 0, model_key: model.key, params, debug: layers.debug }
      : null
  const preview = usePreview(previewReq)
  const previewResult =
    preview.result && media && model && preview.result.media_id === media.id && preview.result.model_key === model.key ? preview.result : null

  const runSpan = useMemo(() => (run ? spanOf(run) : null), [run])
  const sets: OverlaySet[] = useMemo(() => {
    if (runId !== null) return run ? [{ key: `run-${run.id}`, frames: framesQ.data?.frames ?? NO_FRAMES, span: runSpan }] : []
    if (previewResult)
      return [{ key: 'preview', exact: true, frames: [{ frame: previewResult.frame, ms: previewResult.ms, cubes: previewResult.cubes }] }]
    return []
  }, [runId, run, runSpan, framesQ.data, previewResult])

  // The same record the viewer draws (see Viewer: recordFor with the run's span).
  const shownFrame = isVideo ? frame : 0
  const shownRec = useMemo(() => {
    const set = sets[0]
    if (!set) return null
    const rec = recordFor(set.frames, shownFrame, isVideo ? set.span : null)
    return rec && (!set.exact || rec.frame === shownFrame) ? rec : null
  }, [sets, shownFrame, isVideo])

  // Debug images belong to the preview's frame only: not drawn while the video plays or seeks.
  const previewDebug = previewResult && previewResult.frame === shownFrame ? previewResult.debug_layers : NO_DEBUG

  const roi = runId !== null ? roiOf(run?.params, modelList.find((m) => m.key === run?.model_key)) : roiOf(params, model ?? undefined)

  // ----- viewer tools (colour pick, ROI drawing) -----
  const onTool = useCallback(
    (t: ParamTool) => {
      setParamTool(t)
      if (t) leaveRun()
    },
    [leaveRun],
  )
  const viewerTool: ViewerTool | null = useMemo(() => {
    if (!paramTool) return null
    if (paramTool.kind === 'pick') {
      const spec = model?.params.find((s) => s.name === paramTool.param)
      return {
        kind: 'pick',
        label: spec?.label ?? '색',
        onPick: (hex) => {
          changeParam(paramTool.param, hex)
          setParamTool(null)
        },
        onCancel: () => setParamTool(null),
      }
    }
    return {
      kind: 'roi',
      mode: paramTool.mode,
      onComplete: (points) => {
        const current = (latest.current.params[paramTool.param] as RoiPolygon[] | undefined) ?? []
        changeParam(paramTool.param, [...current, { mode: paramTool.mode, points }])
        setParamTool(null)
      },
      onCancel: () => setParamTool(null),
    }
  }, [paramTool, model, changeParam])

  // ----- datasets (the selected one, and the selected folder in it, filter the rail and receive uploads) -----
  const [storedDataset, setStoredDataset] = useState<number | null>(railDataset)
  const [storedFolder, setStoredFolder] = useState<string>(() => railFolder(storedDataset))
  // The URL's dataset wins while it is there (on another dataset, the rail starts at its top).
  const datasetId = intentDataset ?? storedDataset
  const folder = intentDataset !== null && intentDataset !== storedDataset ? '' : storedFolder
  const chooseDataset = useCallback((id: number | null) => {
    setStoredDataset(id)
    setStoredFolder('')
    rememberRail(id, null)
  }, [])
  const chooseFolder = useCallback(
    (path: string) => {
      setStoredFolder(path)
      rememberRail(datasetId, path)
    },
    [datasetId],
  )
  // Also when the workbench is already open: keep what the URL asked for, then the URL keeps
  // media and run only.
  useEffect(() => {
    if (intentDataset === null && !intentGt) return
    if (intentDataset !== null && intentDataset !== storedDataset) chooseDataset(intentDataset)
    if (intentGt) setLayers({ gt: true })
    setSearch(
      (prev) => {
        const next = new URLSearchParams(prev)
        next.delete('dataset')
        next.delete('gt')
        return next
      },
      { replace: true },
    )
  }, [intentDataset, intentGt, storedDataset, chooseDataset, setLayers, setSearch])
  const openRun = useCallback(
    (mid: number, rid: number) => {
      setSearch({ media: String(mid), run: String(rid) })
      setRailOpen(false)
    },
    [setSearch],
  )

  // ----- uploads -----
  const onUploaded = useCallback(
    (res: MediaUploadResult) => {
      void queryClient.invalidateQueries({ queryKey: keys.mediaAll })
      void queryClient.invalidateQueries({ queryKey: keys.datasets })
      selectMedia(res.media.id)
    },
    [selectMedia],
  )
  // Once per upload (a file, a selection, a whole directory), not once per file: GT links and
  // scores, and what is reported.
  const onUploadGroup = useCallback(
    (results: MediaUploadResult[]) => {
      if (results.some((r) => r.folder !== null)) invalidateGt() // a new dataset image may match a GT image by sha256
      const kept = results.filter((r) => r.already_in_dataset)
      if (kept.length) {
        const where = new Map<string, number>()
        for (const r of kept) where.set(r.folder ?? '', (where.get(r.folder ?? '') ?? 0) + 1)
        const label = (f: string) => (f ? `"${f}"` : '맨 위')
        if (where.size === 1) {
          const [f] = where.keys()
          toast.ok(`이미 데이터셋${f ? `의 ${label(f)}` : ' 맨 위'}에 있음, ${kept.length}개 (그대로 둠)`)
        } else {
          const parts = [...where].map(([f, n]) => `${label(f)} ${n}`).join(' · ')
          toast.ok(`이미 데이터셋에 있음, ${kept.length}개 (그대로 둠): ${parts}`)
        }
      }
      const known = results.filter((r) => r.duplicate && !r.already_in_dataset)
      if (known.length === 1) toast.ok(`${named(known[0].media.name, '은/는')} 이미 있는 파일이라 기존 미디어를 열었습니다`)
      else if (known.length > 1) toast.ok(`이미 있던 파일 ${known.length}개는 기존 미디어를 썼습니다`)
    },
    [toast],
  )
  const uploads = useUploads(onUploaded, onUploadGroup)
  const { add: addUploads } = uploads
  const addFiles = useCallback(
    (picked: PickedFile[]) => {
      if (datasetId === null && picked.some((p) => p.dir)) toast.ok('데이터셋을 고르지 않아 폴더 구조 없이 올립니다 (폴더는 데이터셋 안에만 있습니다)')
      const placed = datasetId === null ? picked : placeFiles(picked, folder)
      addUploads(placed.map((p) => ({ file: p.file, folder: joinFolder(folder, p.dir) })), datasetId)
    },
    [addUploads, datasetId, folder, toast],
  )
  const addPlainFiles = useCallback((files: File[]) => addFiles(plainFiles(files)), [addFiles])
  const dragging = useWindowFileDrop(addFiles)
  usePasteFiles(addPlainFiles)

  const samples = useMutation({
    mutationFn: api.addSamples,
    onSuccess: (list) => {
      void queryClient.invalidateQueries({ queryKey: keys.mediaAll })
      if (list.length) selectMedia(list[0].id)
      toast.ok(`합성 샘플 ${list.length}개를 넣었습니다`)
    },
    onError: (e: Error) => toast.error(e.message),
  })

  // ----- runs -----
  const createRun = useMutation({
    mutationFn: (req: RunRequest) =>
      api.createRun({ media_id: media!.id, model_key: model!.key, params, parent_run_id: parentRunId, ...req }),
    onSuccess: (r, req) => {
      invalidateRuns()
      setParentRunId(null)
      const oneFrame = r.media_kind === 'image' || (req.frame_end !== null && req.frame_end - req.frame_start === 1)
      if (model?.kind === 'pipeline' && oneFrame) toast.ok(`기록 #${r.id}을 저장했습니다`)
      else selectRun(r.id)
    },
    onError: (e: Error) => toast.error(e.message),
  })

  // ----- badge, result panel -----
  const runActive = isActive(run)
  const badge =
    runId !== null ? (
      <div className="mode-badge is-run">
        <span className="mb-main">
          기록 <span className="num">#{runId}</span>
          {run ? ` · ${run.model_name}` : ''}
        </span>
        {run && run.status !== 'done' ? <StatusMark status={run.status} /> : null}
        {run && runActive ? (
          <span className="num faint">
            {run.frames_done}/{run.frames_total ?? '?'}
          </span>
        ) : null}
        {run ? (
          <button type="button" className="btn btn-sm" onClick={() => rerunFrom(run)} title="이 기록의 모델과 파라미터를 오른쪽에 불러옵니다">
            파라미터 바꿔 다시 실행
          </button>
        ) : null}
        <button type="button" className="icon-btn sm" aria-label="미리보기로 돌아가기" title="미리보기로" onClick={() => selectRun(null)}>
          <X size={13} strokeWidth={1.75} />
        </button>
      </div>
    ) : model?.kind === 'pipeline' ? (
      <div className={`mode-badge is-preview${preview.pending ? ' pending' : ''}`}>
        <span className="mb-main">미리보기 · 저장 안 됨</span>
        {isVideo && !paused ? <span className="faint">재생 중에는 미리보기를 멈춥니다</span> : null}
      </div>
    ) : model ? (
      <div className="mode-badge">
        <span className="mb-main">가중치 모델</span>
        <span className="faint">미리보기가 없습니다 · “실행”을 누르면 결과가 여기에 그려집니다</span>
      </div>
    ) : null

  const resultTitle =
    runId !== null
      ? `기록 #${runId}${isVideo && shownRec ? ` · f ${shownRec.frame}` : ''}`
      : previewResult
        ? `미리보기${isVideo ? ` · f ${previewResult.frame}` : ''}`
        : ''
  const timing: ResultTiming | null =
    runId !== null
      ? run
        ? { ms: shownRec?.ms ?? null, stages: null, rtt: null, run: run.timing }
        : null
      : previewResult
        ? { ms: previewResult.ms, stages: previewResult.stages_ms, rtt: preview.rtt, run: null }
        : null
  const resultEmpty =
    !media
      ? '미디어를 고르면 결과가 여기에 나옵니다.'
      : runId !== null
        ? run?.status === 'failed'
          ? '실행이 실패했습니다.'
          : isVideo && runSpan && outsideSpan(runSpan, shownFrame)
            ? '이 구간은 처리하지 않았습니다.'
            : runActive
              ? '처리 중입니다…'
              : '이 프레임의 처리 결과가 없습니다.'
        : model?.kind === 'weights'
          ? '“실행”을 누르면 결과가 나옵니다.'
          : isVideo && !paused
            ? '멈추면 현재 프레임을 미리 봅니다.'
            : '미리보기를 계산하는 중…'
  const resultError = runId !== null ? (run?.error ?? (runQ.error?.message || null)) : preview.error

  // ----- ground truth: the 정답 layer and the live score (images with a GT label only) -----
  const hasScore = useFeature('score')
  const mediaGt = useMediaGt(media?.kind === 'image' ? media.id : null)
  const [pickedGt] = useGtPick(datasetId)
  // The set the rail shows for its dataset if it labels this image, else the newest that does
  // (none on a server that cannot score).
  const gtLink =
    (hasScore && media?.kind === 'image' && (mediaGt.links.find((l) => l.dataset_id === datasetId && l.gt_set_id === pickedGt) ?? mediaGt.links[0])) ||
    null
  const { thresholds: th } = useSettledCriteria()
  // A shown run has its own score (GET /api/score/image), for the overlay and the result panel
  // alike; only an unsaved preview is scored live (POST /api/score/frame).
  const runScore = useScoreImage(gtLink && runId !== null ? run : null, gtLink?.gt_set_id ?? null, th)
  const frameScore = useScoreFrame(
    gtLink && media && runId === null && shownRec ? { media_id: media.id, cubes: shownRec.cubes, gt_set_id: gtLink.gt_set_id, criteria: th } : null,
  )
  const runDetail = run && runScore.data?.run_id === run.id ? runScore.data : null
  const frameDetail = media && frameScore.data?.media_id === media.id ? frameScore.data : null
  const shown = runId !== null ? runScore : frameScore
  const overlayGt = runId !== null ? runDetail : frameDetail
  const liveScore: LiveScore | null =
    gtLink && (shownRec || (runId !== null && runDetail))
      ? { detail: overlayGt, pending: shown.isFetching, error: shown.error?.message ?? null }
      : null

  const highlightSafe = highlight !== null && shownRec && highlight < shownRec.cubes.length ? highlight : null

  const libraryEmpty = allMedia.data?.length === 0

  return (
    <div className={`workbench${railOpen ? ' rail-open' : ''}`}>
      <aside className="wb-rail" aria-label="미디어 목록">
        <MediaRail
          selectedId={mediaId}
          onSelect={pickMedia}
          uploads={uploads.items}
          onDismissUpload={uploads.dismiss}
          onFiles={addFiles}
          datasetId={datasetId}
          onDataset={chooseDataset}
          folder={datasetId === null ? '' : folder}
          onFolder={chooseFolder}
          model={model}
          params={params}
          onOpenRun={openRun}
        />
      </aside>
      {railOpen ? <div className="rail-scrim" onClick={() => setRailOpen(false)} /> : null}

      <section className="wb-centre" aria-label="뷰어">
        {media ? <MediaBar media={media} onToggleRail={() => setRailOpen((v) => !v)} /> : null}
        <div className="wb-viewer">
          {media ? (
            <Viewer
              key={media.id}
              media={media}
              sets={sets}
              layers={layers}
              onLayersChange={setLayers}
              debugLayers={runId === null && model?.kind === 'pipeline' ? previewDebug : null}
              roi={roi}
              highlight={highlightSafe}
              onHighlight={setHighlight}
              onFrameChange={setFrame}
              onPausedChange={setPaused}
              badge={badge}
              tool={viewerTool}
              gtAvailable={!!gtLink}
              gt={overlayGt}
            />
          ) : (
            <EmptyViewer
              libraryEmpty={!!libraryEmpty}
              onFiles={addPlainFiles}
              onSamples={() => samples.mutate()}
              samplesPending={samples.isPending}
            />
          )}
        </div>
        {media ? (
          <RunStrip
            mediaId={media.id}
            mediaThumb={media.thumb_url}
            selectedId={runId}
            onSelect={selectRun}
            checked={checked}
            onCheck={setChecked}
          />
        ) : null}
      </section>

      <aside className="wb-panel" aria-label="모델과 파라미터">
        <div className="panel-top">
          <h2 className="panel-title">모델</h2>
          {models.isError ? <p className="error-text">{models.error.message}</p> : null}
          <ModelPicker models={modelList} value={model?.key ?? null} onChange={chooseModel} />
          {model ? (
            // The JSON dialog sits outside the keyed preset bar: applying JSON may switch the model,
            // and the dialog must stay open to show its report.
            <div className="preset-row">
              <PresetBar key={model.key} modelKey={model.key} specs={model.params} params={params} onApply={applyPreset} />
              <ParamJsonButton
                model={model}
                models={modelList}
                params={params}
                paramsFor={paramsFor}
                media={media ?? null}
                onApply={applyJson}
              />
            </div>
          ) : null}
        </div>
        <div className="panel-params">
          {model ? (
            <ParamControls
              specs={model.params}
              values={params}
              onChange={changeParam}
              onResetGroup={resetGroup}
              tool={paramTool}
              onTool={onTool}
              canUseViewer={!!media}
            />
          ) : models.isPending ? (
            <p className="faint pc-empty">모델 목록을 불러오는 중…</p>
          ) : (
            <p className="faint pc-empty">쓸 수 있는 모델이 없습니다.</p>
          )}
        </div>
        <RunActions
          key={`${media?.id ?? 0}-${model?.kind ?? ''}`}
          media={media}
          model={model}
          frame={frame}
          pending={createRun.isPending}
          parentRunId={parentRunId}
          onClearParent={() => setParentRunId(null)}
          onCreate={(req) => createRun.mutate(req)}
        />
        <ResultPanel
          title={resultTitle}
          cubes={shownRec?.cubes ?? null}
          highlight={highlightSafe}
          onHighlight={setHighlight}
          timing={timing}
          notes={runId === null ? (previewResult?.notes ?? []) : []}
          error={resultError}
          empty={resultEmpty}
          score={liveScore}
        />
      </aside>

      {dragging ? (
        <div className="drop-overlay" aria-hidden="true">
          <div className="drop-target reg">
            <Upload size={32} strokeWidth={1.5} />
            <strong>놓으면 올립니다</strong>
            <span className="muted">이미지 · 영상 · 폴더, 여러 개도 순서대로</span>
          </div>
        </div>
      ) : null}
    </div>
  )
}
