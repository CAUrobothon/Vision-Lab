// react-query keys and hooks, one per resource. Runs poll every second only while one of the
// visible runs is queued or running; everything else refreshes on mutation.
import { useEffect } from 'react'
import { hashKey, keepPreviousData, QueryClient, useQuery, type QueryKey } from '@tanstack/react-query'
import { api, ApiError, type MediaFilter, type RunFilter } from './client'
import { useFeature } from './features'
import type { Run, RunFrames, ScoreCriteriaIn, ScoreFrameRequest, ScoreImageRow, ScoreRequest, ScoreResult } from './types'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 2_000,
      retry: (count, err) => (err as ApiError).status >= 500 && count < 1,
    },
  },
})

export const keys = {
  health: ['health'] as const,
  system: ['system'] as const,
  storage: ['storage'] as const,
  mediaAll: ['media'] as const,
  mediaList: (f: MediaFilter) => ['media', 'list', f] as const,
  media: (id: number) => ['media', 'one', id] as const,
  models: ['models'] as const,
  datasets: ['datasets'] as const,
  batches: (datasetId: number) => ['batches', 'list', datasetId] as const,
  batchesAll: ['batches'] as const,
  batch: (id: number) => ['batches', 'one', id] as const,
  presets: (modelKey: string) => ['presets', modelKey] as const,
  runsAll: ['runs'] as const,
  runList: (f: RunFilter) => ['runs', 'list', f] as const,
  run: (id: number) => ['runs', 'one', id] as const,
  runFrames: (id: number) => ['runFrames', id] as const,
  gtAll: ['gt'] as const,
  gtSets: (datasetId: number) => ['gt', 'sets', datasetId] as const,
  gtSet: (id: number) => ['gt', 'one', id] as const,
  mediaGt: (mediaId: number) => ['gt', 'media', mediaId] as const,
  scoreAll: ['score'] as const,
  scoreCriteria: ['scoreCriteria'] as const,
  score: (req: ScoreRequest, stamp: string) => ['score', 'batches', req, stamp] as const,
  scoreImages: (gtSetId: number, batchId: number, c: ScoreCriteriaIn, stamp: string) =>
    ['score', 'images', gtSetId, batchId, c, stamp] as const,
  scoreImage: (runId: number, gtSetId: number | null, c: ScoreCriteriaIn, stamp: string) =>
    ['score', 'image', runId, gtSetId, c, stamp] as const,
  scoreFrame: (req: ScoreFrameRequest) => ['score', 'frame', req] as const,
}

export function isActive(run: Pick<Run, 'status'> | null | undefined): boolean {
  return run?.status === 'queued' || run?.status === 'running'
}

/** Refresh the lists that show runs or run counts after any run change. */
export function invalidateRuns() {
  void queryClient.invalidateQueries({ queryKey: keys.runsAll })
  void queryClient.invalidateQueries({ queryKey: keys.mediaAll })
  void queryClient.invalidateQueries({ queryKey: keys.batchesAll })
}

/** Ground-truth links or sets changed: every link count and score is stale. `fresh` = a GT query
 *  whose answer was just put into the cache, so it is not fetched again. */
export function invalidateGt(fresh?: QueryKey) {
  const skip = fresh ? hashKey(fresh) : null
  void queryClient.invalidateQueries({ queryKey: keys.gtAll, predicate: (q) => q.queryHash !== skip })
  void queryClient.invalidateQueries({ queryKey: keys.scoreAll })
}

// GT and score requests are not retried: a 404 is a missing record (a server without these routes
// is never asked, see api/features.ts) and a scoring error repeats.
const noRetry = { retry: false } as const

export function useHealth() {
  return useQuery({
    queryKey: keys.health,
    queryFn: api.health,
    refetchInterval: 4_000,
    retry: false,
  })
}

export function useSystem(poll = false) {
  return useQuery({ queryKey: keys.system, queryFn: api.system, refetchInterval: poll ? 3_000 : 15_000 })
}

export function useStorage() {
  return useQuery({ queryKey: keys.storage, queryFn: api.storage })
}

export function useMediaList(filter: MediaFilter) {
  return useQuery({ queryKey: keys.mediaList(filter), queryFn: () => api.listMedia(filter), placeholderData: keepPreviousData })
}

export function useMedia(id: number | null) {
  return useQuery({
    queryKey: keys.media(id ?? -1),
    queryFn: () => api.getMedia(id!),
    enabled: id !== null,
  })
}

export function useDatasets() {
  return useQuery({ queryKey: keys.datasets, queryFn: api.listDatasets })
}

/** Batches of a dataset; polls every second while one of them is still running. */
export function useBatches(datasetId: number | null) {
  return useQuery({
    queryKey: keys.batches(datasetId ?? -1),
    queryFn: () => api.listBatches(datasetId!),
    enabled: datasetId !== null,
    refetchInterval: (q) => (q.state.data?.some((b) => !b.finished) ? 1_000 : false),
  })
}

export function useBatch(id: number | null) {
  return useQuery({
    queryKey: keys.batch(id ?? -1),
    queryFn: () => api.getBatch(id!),
    enabled: id !== null,
    refetchInterval: (q) => (q.state.data && !q.state.data.finished ? 1_000 : false),
  })
}

export function useModels() {
  return useQuery({ queryKey: keys.models, queryFn: api.listModels, staleTime: 30_000 })
}

export function usePresets(modelKey: string | null) {
  return useQuery({
    queryKey: keys.presets(modelKey ?? ''),
    queryFn: () => api.listPresets(modelKey!),
    enabled: !!modelKey,
  })
}

export function useRuns(filter: RunFilter, enabled = true) {
  return useQuery({
    queryKey: keys.runList(filter),
    queryFn: () => api.listRuns(filter),
    enabled,
    placeholderData: keepPreviousData,
    refetchInterval: (q) => (q.state.data?.items.some(isActive) ? 1_000 : false),
  })
}

export function useRun(id: number | null) {
  return useQuery({
    queryKey: keys.run(id ?? -1),
    queryFn: () => api.getRun(id!),
    enabled: id !== null,
    refetchInterval: (q) => (isActive(q.state.data) ? 1_000 : false),
  })
}

/** A run's frames plus the run's `frames_done` when they were fetched. */
type RunFramesFor = RunFrames & { framesDone: number }

/**
 * Frames of a run: one cached copy per run (a long video run is megabytes of JSON, so copies per
 * progress step must not pile up). While the run makes progress the copy is fetched again whenever
 * `frames_done` has moved past what it covers, one request at a time; the API has no "frames
 * since", so each fetch replaces the whole list.
 */
export function useRunFrames(run: Run | null | undefined) {
  const id = run?.id ?? -1
  const done = run?.frames_done ?? 0
  const q = useQuery({
    queryKey: keys.runFrames(id),
    queryFn: async (): Promise<RunFramesFor> => ({ ...(await api.runFrames(id)), framesDone: done }),
    enabled: !!run && done > 0,
    staleTime: Infinity,
    gcTime: 60_000,
  })
  const { data, isFetching, refetch } = q
  useEffect(() => {
    if (data && !isFetching && done > 0 && data.framesDone !== done) void refetch()
  }, [data, isFetching, done, refetch])
  return q
}

// The GT and score hooks ask only a server that has those routes (health `features`).

/** The dataset's ground-truth sets, newest first. */
export function useGtSets(datasetId: number | null) {
  const gt = useFeature('gt')
  return useQuery({
    queryKey: keys.gtSets(datasetId ?? -1),
    queryFn: () => api.listGtSets(datasetId!),
    enabled: gt && datasetId !== null,
    ...noRetry,
  })
}

export function useGtSet(id: number | null) {
  const gt = useFeature('gt')
  return useQuery({ queryKey: keys.gtSet(id ?? -1), queryFn: () => api.getGtSet(id!), enabled: gt && id !== null, ...noRetry })
}

/** The sets that label this media (images only); none on a server without ground truth. */
export function useMediaGt(mediaId: number | null) {
  const gt = useFeature('gt')
  const q = useQuery({
    queryKey: keys.mediaGt(mediaId ?? -1),
    queryFn: () => api.mediaGt(mediaId!),
    enabled: gt && mediaId !== null,
    staleTime: 10_000,
    ...noRetry,
  })
  return { ...q, links: (gt && q.data) || [] }
}

/** The criteria defaults and limits (GET /api/score/criteria); fixed while the server runs. */
export function useScoreCriteriaInfo() {
  const score = useFeature('score')
  return useQuery({ queryKey: keys.scoreCriteria, queryFn: api.scoreCriteria, enabled: score, staleTime: Infinity, ...noRetry })
}

// A score is polled while some image it covers waits for or is in a run, as its last answer says
// (`fallback`, a batch of the selection not finished, until there is one): cancelled or missing
// images never change it.

function scoreRunning(data: ScoreResult | undefined, fallback: boolean): boolean {
  return data ? data.batches.some((b) => b.pending.queued + b.pending.running > 0) : fallback
}

function imagesRunning(rows: ScoreImageRow[] | undefined, fallback: boolean): boolean {
  return rows ? rows.some((r) => r.status === 'queued' || r.status === 'running') : fallback
}

/**
 * POST /api/score. `stamp` names the state of the batches it covers (their ids and which are
 * finished), so a new or finished batch is scored at once; while one runs the result is polled
 * (`poll`: whether one of them is unfinished, used until an answer says). The previous result
 * stays on screen while a new one is computed (the first scoring after a server start or a τ / g
 * change takes seconds). A request superseded by another is aborted.
 */
export function useScore(req: ScoreRequest | null, stamp: string, poll: boolean) {
  const score = useFeature('score')
  return useQuery({
    queryKey: keys.score(req ?? { gt_set_id: -1, batch_ids: [], criteria: {}, reference_batch_id: null }, stamp),
    queryFn: ({ signal }) => api.score(req!, signal),
    enabled: score && req !== null,
    placeholderData: keepPreviousData,
    staleTime: 30_000,
    refetchInterval: (q) => (scoreRunning(q.state.data, poll) ? 2_000 : false),
    ...noRetry,
  })
}

export function useScoreImages(gtSetId: number | null, batchId: number | null, c: ScoreCriteriaIn, stamp: string, poll: boolean) {
  const score = useFeature('score')
  return useQuery({
    queryKey: keys.scoreImages(gtSetId ?? -1, batchId ?? -1, c, stamp),
    queryFn: ({ signal }) => api.scoreImages(gtSetId!, batchId!, c, signal),
    enabled: score && gtSetId !== null && batchId !== null,
    placeholderData: keepPreviousData,
    staleTime: 30_000,
    refetchInterval: (q) => (imagesRunning(q.state.data, poll) ? 2_000 : false),
    ...noRetry,
  })
}

/** A run's frame 0 against its media's GT image (the viewer's 정답 layer and the live score of a
 *  shown run). Asked again when the run's status changes (it is scored once it is done or failed). */
export function useScoreImage(run: Pick<Run, 'id' | 'status'> | null, gtSetId: number | null, c: ScoreCriteriaIn) {
  const score = useFeature('score')
  return useQuery({
    queryKey: keys.scoreImage(run?.id ?? -1, gtSetId, c, run?.status ?? ''),
    queryFn: ({ signal }) => api.scoreImage(run!.id, gtSetId, c, signal),
    enabled: score && run !== null,
    placeholderData: keepPreviousData,
    ...noRetry,
  })
}

/** Cubes the workbench has right now (an unsaved preview) against the media's GT image. */
export function useScoreFrame(req: ScoreFrameRequest | null) {
  const score = useFeature('score')
  return useQuery({
    queryKey: keys.scoreFrame(req ?? { media_id: -1, cubes: [], gt_set_id: null, criteria: {} }),
    queryFn: ({ signal }) => api.scoreFrame(req!, signal),
    enabled: score && req !== null,
    placeholderData: (prev) => (prev && req && prev.media_id === req.media_id ? prev : undefined),
    staleTime: Infinity,
    gcTime: 30_000,
    ...noRetry,
  })
}
