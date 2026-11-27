// Small typed client over fetch. Errors carry the backend's Korean `detail`; a network failure
// becomes ApiError with status 0 so the shell can show "server unreachable".
// Uploads use XMLHttpRequest because fetch has no upload progress.
import type {
  Batch,
  BatchDetail,
  CleanupRequest,
  Dataset,
  Health,
  CleanupResult,
  ExportInfo,
  GtLinkIn,
  GtSetDetail,
  GtSetInfo,
  Media,
  MediaGtLink,
  MediaKind,
  MediaUploadResult,
  ModelInfo,
  Ok,
  ParamCheck,
  Params,
  Preset,
  PreviewRequest,
  PreviewResult,
  Run,
  RunCreate,
  RunFrames,
  RunList,
  RunStatus,
  ScoreCriteriaIn,
  ScoreCriteriaInfo,
  ScoreFrameRequest,
  ScoreImageDetail,
  ScoreImageRow,
  ScoreRequest,
  ScoreResult,
  StorageInfo,
  SystemInfo,
} from './types'

export class ApiError extends Error {
  status: number
  constructor(status: number, detail: string) {
    super(detail)
    this.status = status
  }
}

export const UNREACHABLE = '서버에 연결할 수 없습니다'

async function readDetail(res: Response): Promise<string> {
  try {
    const body = await res.json()
    if (typeof body?.detail === 'string') return body.detail
    if (Array.isArray(body?.detail)) return '요청 형식이 올바르지 않습니다'
  } catch {
    // Not JSON. The backend always answers with {detail}, so a bare 5xx comes from the dev proxy
    // (or nothing at all) when the backend is down.
  }
  return res.status >= 500 ? UNREACHABLE : `요청이 실패했습니다 (HTTP ${res.status})`
}

async function request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  let res: Response
  try {
    res = await fetch('/api' + path, {
      method,
      signal,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch (err) {
    if ((err as Error).name === 'AbortError') throw err
    throw new ApiError(0, UNREACHABLE)
  }
  if (!res.ok) throw new ApiError(res.status, await readDetail(res))
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export function query(params: Record<string, string | number | boolean | null | undefined>): string {
  const q = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') q.set(k, String(v))
  }
  const s = q.toString()
  return s ? '?' + s : ''
}

export interface UploadProgress {
  loaded: number
  total: number
}

function upload<T>(path: string, form: FormData, onProgress?: (p: UploadProgress) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api' + path)
    xhr.responseType = 'json'
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress?.({ loaded: e.loaded, total: e.total })
    }
    xhr.onload = () => {
      const body = xhr.response
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as T)
      else {
        const detail = typeof body?.detail === 'string' ? body.detail : `올리기가 실패했습니다 (HTTP ${xhr.status})`
        reject(new ApiError(xhr.status, detail))
      }
    }
    xhr.onerror = () => reject(new ApiError(0, UNREACHABLE))
    xhr.send(form)
  })
}

export interface MediaFilter {
  dataset_id?: number
  folder?: string // with dataset_id: one folder and everything below it
  kind?: MediaKind
  q?: string
  starred?: boolean
  tag?: string
}

export interface RunFilter {
  media_id?: number
  model_key?: string
  status?: RunStatus
  starred?: boolean
  tag?: string
  q?: string
  limit?: number
  offset?: number
}

export interface MediaPatch {
  name?: string
  tags?: string[]
  note?: string
  starred?: boolean
}

export interface RunPatch {
  note?: string
  tags?: string[]
  starred?: boolean
}

export const api = {
  health: () => request<Health>('GET', '/health'),
  system: () => request<SystemInfo>('GET', '/system'),

  listMedia: (f: MediaFilter = {}) => request<Media[]>('GET', '/media' + query({ ...f })),
  getMedia: (id: number) => request<Media>('GET', `/media/${id}`),
  patchMedia: (id: number, patch: MediaPatch) => request<Media>('PATCH', `/media/${id}`, patch),
  deleteMedia: (id: number) => request<Ok>('DELETE', `/media/${id}`),
  /** With a dataset, `folder` ('' = its top level) is where the file goes in it. */
  uploadMedia: (file: File, onProgress?: (p: UploadProgress) => void, datasetId?: number | null, folder = '') => {
    const form = new FormData()
    if (datasetId != null && folder) form.append('folder', folder)
    form.append('file', file, file.name)
    return upload<MediaUploadResult>('/media' + query({ dataset_id: datasetId }), form, onProgress)
  },

  listDatasets: () => request<Dataset[]>('GET', '/datasets'),
  createDataset: (body: { name: string; note?: string; media_ids?: number[] }) => request<Dataset>('POST', '/datasets', body),
  patchDataset: (id: number, patch: { name?: string; note?: string }) => request<Dataset>('PATCH', `/datasets/${id}`, patch),
  deleteDataset: (id: number) => request<Ok>('DELETE', `/datasets/${id}`),
  addToDataset: (id: number, mediaIds: number[]) => request<Dataset>('POST', `/datasets/${id}/media`, { media_ids: mediaIds }),
  removeFromDataset: (id: number, mediaIds: number[]) =>
    request<Dataset>('POST', `/datasets/${id}/media/remove`, { media_ids: mediaIds }),
  moveMedia: (id: number, mediaIds: number[], folder: string) =>
    request<Dataset>('POST', `/datasets/${id}/media/move`, { media_ids: mediaIds, folder }),
  createFolder: (id: number, path: string) => request<Dataset>('POST', `/datasets/${id}/folders`, { path }),
  renameFolder: (id: number, path: string, newPath: string) =>
    request<Dataset>('POST', `/datasets/${id}/folders/rename`, { path, new_path: newPath }),
  deleteFolder: (id: number, path: string) => request<Dataset>('POST', `/datasets/${id}/folders/delete`, { path }),
  listBatches: (datasetId: number) => request<Batch[]>('GET', `/datasets/${datasetId}/batches`),
  createBatch: (datasetId: number, body: { model_key: string; params: Params; stride: number; folder: string | null }) =>
    request<BatchDetail>('POST', `/datasets/${datasetId}/batches`, body),
  getBatch: (id: number) => request<BatchDetail>('GET', `/batches/${id}`),
  cancelBatch: (id: number) => request<BatchDetail>('POST', `/batches/${id}/cancel`),
  deleteBatch: (id: number) => request<Ok>('DELETE', `/batches/${id}`),
  addSamples: () => request<Media[]>('POST', '/samples'),

  listModels: () => request<ModelInfo[]>('GET', '/models'),
  uploadWeights: (file: File, name: string, onProgress?: (p: UploadProgress) => void) => {
    const form = new FormData()
    form.append('file', file, file.name)
    if (name) form.append('name', name)
    return upload<ModelInfo>('/models/weights', form, onProgress)
  },
  patchModel: (key: string, patch: { name?: string; note?: string }) =>
    request<ModelInfo>('PATCH', `/models/${encodeURIComponent(key)}`, patch),
  deleteModel: (key: string) => request<Ok>('DELETE', `/models/${encodeURIComponent(key)}`),

  checkParams: (body: { model_key: string; params: Params }) => request<ParamCheck>('POST', '/params/check', body),

  listPresets: (modelKey: string) => request<Preset[]>('GET', '/presets' + query({ model_key: modelKey })),
  createPreset: (body: { model_key: string; name: string; params: Params }) =>
    request<Preset>('POST', '/presets', body),
  deletePreset: (id: number) => request<Ok>('DELETE', `/presets/${id}`),

  preview: (body: PreviewRequest, signal?: AbortSignal) => request<PreviewResult>('POST', '/preview', body, signal),

  listRuns: (f: RunFilter = {}) => request<RunList>('GET', '/runs' + query({ ...f })),
  getRun: (id: number) => request<Run>('GET', `/runs/${id}`),
  createRun: (body: RunCreate) => request<Run>('POST', '/runs', body),
  patchRun: (id: number, patch: RunPatch) => request<Run>('PATCH', `/runs/${id}`, patch),
  deleteRun: (id: number) => request<Ok>('DELETE', `/runs/${id}`),
  cancelRun: (id: number) => request<Run>('POST', `/runs/${id}/cancel`),
  runFrames: (id: number) => request<RunFrames>('GET', `/runs/${id}/frames`),
  exportRun: (id: number) => request<ExportInfo>('POST', `/runs/${id}/export`),

  listGtSets: (datasetId: number) => request<GtSetInfo[]>('GET', `/datasets/${datasetId}/gt`),
  uploadGt: (datasetId: number, file: File) => {
    const form = new FormData()
    form.append('file', file, file.name)
    return upload<GtSetInfo>(`/datasets/${datasetId}/gt`, form)
  },
  getGtSet: (id: number) => request<GtSetDetail>('GET', `/gt/${id}`),
  renameGtSet: (id: number, name: string) => request<GtSetInfo>('PATCH', `/gt/${id}`, { name }),
  deleteGtSet: (id: number) => request<Ok>('DELETE', `/gt/${id}`),
  confirmGtLinks: (id: number, links: GtLinkIn[]) => request<GtSetDetail>('POST', `/gt/${id}/links`, { links }),
  removeGtLink: (id: number, gtIndex: number) => request<GtSetDetail>('DELETE', `/gt/${id}/links/${gtIndex}`),
  organizeGt: (id: number, root: string) => request<Dataset>('POST', `/gt/${id}/organize`, { root }),
  mediaGt: (mediaId: number) => request<MediaGtLink[]>('GET', `/media/${mediaId}/gt`),

  score: (body: ScoreRequest, signal?: AbortSignal) => request<ScoreResult>('POST', '/score', body, signal),
  scoreCriteria: () => request<ScoreCriteriaInfo>('GET', '/score/criteria'),
  scoreImage: (runId: number, gtSetId: number | null, c: ScoreCriteriaIn, signal?: AbortSignal) =>
    request<ScoreImageDetail>('GET', '/score/image' + query({ run_id: runId, gt_set_id: gtSetId, ...c }), undefined, signal),
  scoreFrame: (body: ScoreFrameRequest, signal?: AbortSignal) => request<ScoreImageDetail>('POST', '/score/frame', body, signal),
  scoreImages: (gtSetId: number, batchId: number, c: ScoreCriteriaIn, signal?: AbortSignal) =>
    request<ScoreImageRow[]>('GET', '/score/images' + query({ gt_set_id: gtSetId, batch_id: batchId, ...c }), undefined, signal),
  /** cubes.csv / summary.csv for the selection (no batch ids = every batch of the set's dataset). */
  scoreCsvUrl: (kind: 'cubes' | 'summary', gtSetId: number, batchIds: number[], c: ScoreCriteriaIn) =>
    `/api/score/${kind}.csv` + query({ gt_set_id: gtSetId, batch_ids: batchIds.join(','), ...c }),

  storage: () => request<StorageInfo>('GET', '/storage'),
  cleanup: (body: CleanupRequest) => request<CleanupResult>('POST', '/storage/cleanup', body),
  restartWorker: () => request<unknown>('POST', '/worker/restart'),
}
