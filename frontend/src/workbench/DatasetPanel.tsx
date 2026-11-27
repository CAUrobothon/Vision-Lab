// Datasets in the left rail: pick one (the media list then shows only its media, and uploads go
// into it), create / rename / delete, download its originals; its folder tree (pick a folder to
// list and upload into it, create / rename / delete folders); run the current model on all of it
// or on the selected folder and download the drawn results, all at once (zip) or one by one.
import { Fragment, useEffect, useMemo, useState, type CSSProperties } from 'react'
import { useMutation } from '@tanstack/react-query'
import { ChevronDown, ChevronRight, Download, Folder, FolderOpen, FolderPlus, FolderTree as FolderTreeIcon, Pencil, Play, Square } from 'lucide-react'
import { api } from '../api/client'
import { useFeature } from '../api/features'
import { invalidateGt, invalidateRuns, keys, queryClient, useBatch, useBatches, useDatasets, useGtSets, useScore } from '../api/queries'
import type { Batch, BatchItem, Dataset, DatasetFolder, GtSetInfo, ModelInfo, Params, ScoreBatch } from '../api/types'
import { byName, joinFolder, parentOf, relativeTo, within } from '../lib/folders'
import { changedParams } from '../lib/params'
import { loadJson, saveJson } from '../lib/storage'
import { batchStamp, chooseGt, useGtPick, useSettledCriteria } from '../score/criteria'
import { coverageNd, int, pct, ratioNd } from '../score/format'
import { BatchScope } from '../score/parts'
import { ConfirmButton, NameForm, NumberField, STATUS_LABEL } from '../ui/controls'
import { useToast } from '../ui/toast'
import { named } from '../lib/josa'
import { GtRow } from './GtRow'

// The rail's dataset and folder as this browser remembers them (each open tab keeps its own).

/** The dataset the rail last showed (null = all media). */
export function railDataset(): number | null {
  return loadJson<number | null>('dataset', null)
}

/** The folder the rail last showed in `datasetId` ('' = all of it). */
export function railFolder(datasetId: number | null): string {
  const saved = loadJson<{ dataset: number | null; path: string } | null>('folder', null)
  return saved && saved.dataset === datasetId ? saved.path : ''
}

/** Remember the rail's dataset and its folder (null = none picked in it yet). */
export function rememberRail(datasetId: number | null, folder: string | null) {
  saveJson('dataset', datasetId)
  saveJson('folder', folder === null ? null : { dataset: datasetId, path: folder })
}

/** Media joined, left or moved: dataset counts, lists, GT links (sha256 links follow the media)
 *  and scores (a batch's scope follows the folders) are stale. */
export function invalidateDatasets() {
  void queryClient.invalidateQueries({ queryKey: keys.datasets })
  void queryClient.invalidateQueries({ queryKey: keys.mediaAll })
  invalidateGt()
}

/** A dataset as a change returned it, straight into the cached list (so a renamed folder is never
 *  briefly missing; the list is not fetched again). */
function putDataset(d: Dataset) {
  queryClient.setQueryData<Dataset[]>(keys.datasets, (old) => old?.map((x) => (x.id === d.id ? d : x)))
}

/** A dataset after its media moved or left (folders renamed or deleted, media moved): it goes into
 *  the cache, and the media lists, GT links and scores that follow its folders are refetched. */
export function applyDataset(d: Dataset) {
  putDataset(d)
  void queryClient.invalidateQueries({ queryKey: keys.mediaAll })
  invalidateGt()
}

/** The dataset picker, the selected dataset's actions and its folder tree. */
export function DatasetBar({
  datasetId,
  onDataset,
  folder,
  onFolder,
}: {
  datasetId: number | null
  onDataset: (id: number | null) => void
  folder: string
  onFolder: (path: string) => void
}) {
  const toast = useToast()
  const hasFolders = useFeature('folders')
  const datasets = useDatasets()
  const list = datasets.data ?? []
  const current = list.find((d) => d.id === datasetId) ?? null
  const [naming, setNaming] = useState<'new' | 'rename' | null>(null)

  // A remembered dataset that was deleted (here or in another tab): back to all media.
  useEffect(() => {
    if (datasetId !== null && datasets.data && !datasets.isFetching && !current) onDataset(null)
  }, [datasetId, datasets.data, datasets.isFetching, current, onDataset])

  const create = useMutation({
    mutationFn: (name: string) => api.createDataset({ name }),
    onSuccess: (d) => {
      // In the cache before it is selected, so the "selected dataset is gone" check never sees it missing.
      queryClient.setQueryData<Dataset[]>(keys.datasets, (old) => [...(old ?? []), d])
      invalidateDatasets()
      onDataset(d.id)
      setNaming(null)
      toast.ok(`데이터셋 ${named(d.name, '을/를')} 만들었습니다. 지금부터 올리는 파일은 여기에 들어갑니다`)
    },
    onError: (e: Error) => toast.error(e.message),
  })
  const rename = useMutation({
    mutationFn: (name: string) => api.patchDataset(current!.id, { name }),
    onSuccess: () => {
      invalidateDatasets()
      setNaming(null)
    },
    onError: (e: Error) => toast.error(e.message),
  })
  const remove = useMutation({
    mutationFn: () => api.deleteDataset(current!.id),
    onSuccess: () => {
      toast.ok(`데이터셋 ${named(current?.name ?? '', '을/를')} 지웠습니다. 미디어와 실행 기록은 남아 있고, 일괄 실행 묶음(zip)은 함께 지웠습니다`)
      onDataset(null)
      invalidateDatasets()
    },
    onError: (e: Error) => toast.error(e.message),
  })

  return (
    <div className="ds-bar">
      {naming ? (
        <NameForm
          className="ds-row"
          initial={naming === 'rename' ? (current?.name ?? '') : ''}
          label="데이터셋 이름"
          placeholder="데이터셋 이름"
          submitLabel={naming === 'rename' ? '바꾸기' : '만들기'}
          pending={create.isPending || rename.isPending}
          onSubmit={(name) => (naming === 'rename' ? rename.mutate(name) : create.mutate(name))}
          onCancel={() => setNaming(null)}
        />
      ) : (
        <div className="ds-row">
          <select
            className="select"
            aria-label="데이터셋"
            value={datasetId ?? ''}
            onChange={(e) => onDataset(e.target.value ? Number(e.target.value) : null)}
          >
            <option value="">전체 미디어</option>
            {list.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} · {d.media_count}
              </option>
            ))}
          </select>
          <button type="button" className="btn btn-sm" title="새 데이터셋" onClick={() => setNaming('new')}>
            <FolderPlus size={13} strokeWidth={1.75} /> 새로
          </button>
        </div>
      )}
      {current && !naming ? (
        <div className="ds-row ds-meta">
          <span className="faint num">
            이미지 {current.image_count} · 영상 {current.video_count}
          </span>
          <button type="button" className="icon-btn sm" aria-label="데이터셋 이름 바꾸기" title="이름 바꾸기" onClick={() => setNaming('rename')}>
            <Pencil size={12} strokeWidth={1.75} />
          </button>
          {current.media_count > 0 ? (
            <a className="btn btn-sm btn-quiet" href={current.originals_url} download title="데이터셋의 원본 파일 전부 (zip)">
              <Download size={12} strokeWidth={1.75} /> 원본 zip
            </a>
          ) : null}
          <ConfirmButton
            label={`데이터셋 ${current.name} 삭제 (미디어와 실행 기록은 남음)`}
            confirmLabel="삭제 (미디어는 남음)"
            onConfirm={() => remove.mutate()}
            quiet
          />
        </div>
      ) : null}
      {/* Keyed by dataset (state starts over for another one); the prefixes keep the siblings apart. */}
      {current && !naming ? <GtRow key={`gt-${current.id}`} dataset={current} /> : null}
      {/* A server started before folders existed (restart it): no tree then. */}
      {current && hasFolders ? (
        <FolderTree key={`ft-${current.id}`} dataset={current} folder={folder} onFolder={onFolder} settled={!datasets.isFetching} />
      ) : null}
    </div>
  )
}

// ----- folders -----

interface FolderNode extends DatasetFolder {
  name: string
  depth: number // 1 = a folder at the dataset's top level
  children: FolderNode[]
}

/** The flat, path-sorted list (parents first) as a tree; siblings in natural order (증강2 before 증강10). */
function buildTree(folders: DatasetFolder[]): FolderNode[] {
  const nodes = new Map<string, FolderNode>()
  const top: FolderNode[] = []
  for (const f of folders) {
    const cut = f.path.lastIndexOf('/')
    const node: FolderNode = { ...f, name: f.path.slice(cut + 1), depth: f.path.split('/').length, children: [] }
    nodes.set(f.path, node)
    const parent = cut < 0 ? undefined : nodes.get(f.path.slice(0, cut))
    ;(parent ? parent.children : top).push(node)
  }
  const sort = (list: FolderNode[]) => {
    list.sort((a, b) => byName.compare(a.name, b.name))
    for (const n of list) sort(n.children)
  }
  sort(top)
  return top
}

/** A folder name typed in the tree, indented to `depth`. */
function FolderNameForm({ depth, ...form }: {
  initial: string
  label: string
  submitLabel: string
  depth: number
  pending: boolean
  onSubmit: (name: string) => void
  onCancel: () => void
}) {
  return (
    <NameForm {...form} className="ft-form" style={{ '--depth': depth } as CSSProperties} placeholder="폴더 이름" cancel="icon" selectOnFocus />
  )
}

/** The selected dataset's folders: pick one (the media list and uploads follow), make one inside
 *  it, rename or delete it. Counts include every folder below. */
function FolderTree({
  dataset,
  folder,
  onFolder,
  settled,
}: {
  dataset: Dataset
  folder: string
  onFolder: (path: string) => void
  settled: boolean // the dataset list is not being refetched
}) {
  const toast = useToast()
  const tree = useMemo(() => buildTree(dataset.folders), [dataset.folders])
  const [collapsed, setCollapsed] = useState<string[]>([])
  const [creating, setCreating] = useState(false)
  const [renaming, setRenaming] = useState<string | null>(null)

  // A remembered folder that is gone (deleted or renamed elsewhere): its nearest parent that is left.
  useEffect(() => {
    if (!folder || !settled) return
    const paths = new Set(dataset.folders.map((f) => f.path))
    if (paths.has(folder)) return
    let up = parentOf(folder)
    while (up && !paths.has(up)) up = parentOf(up)
    onFolder(up)
  }, [folder, settled, dataset.folders, onFolder])

  const create = useMutation({
    mutationFn: (name: string) => api.createFolder(dataset.id, joinFolder(folder, name)),
    onSuccess: (d) => {
      putDataset(d) // an empty folder: no media, GT link or score changed
      setCreating(false)
      setCollapsed((c) => c.filter((p) => p !== folder))
    },
    onError: (e: Error) => toast.error(e.message),
  })
  const rename = useMutation({
    mutationFn: ({ path, to }: { path: string; to: string }) => api.renameFolder(dataset.id, path, to),
    onSuccess: (d, { path, to }) => {
      applyDataset(d)
      void queryClient.invalidateQueries({ queryKey: keys.batchesAll }) // their scope is renamed too
      setRenaming(null)
      if (within(folder, path)) onFolder(to + folder.slice(path.length))
    },
    onError: (e: Error) => toast.error(e.message),
  })
  const remove = useMutation({
    mutationFn: (node: FolderNode) => api.deleteFolder(dataset.id, node.path),
    onSuccess: (d, node) => {
      applyDataset(d)
      if (within(folder, node.path)) onFolder(parentOf(node.path))
      toast.ok(
        `폴더 ${named(node.name, '을/를')} 지웠습니다` +
          (node.total ? `. 그 안의 미디어 ${node.total}개는 이 데이터셋에서만 빠졌고, 파일과 실행 기록은 남아 있습니다` : ''),
      )
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const rows: FolderNode[] = []
  const walk = (list: FolderNode[]) => {
    for (const n of list) {
      rows.push(n)
      if (!collapsed.includes(n.path)) walk(n.children)
    }
  }
  walk(tree)

  const newForm = (depth: number) => (
    <li>
      <FolderNameForm
        initial=""
        label="새 폴더 이름"
        submitLabel="만들기"
        depth={depth}
        pending={create.isPending}
        onSubmit={(name) => create.mutate(name)}
        onCancel={() => setCreating(false)}
      />
    </li>
  )

  return (
    <div className="ft">
      <div className="ft-head">
        <span className="ft-title">폴더</span>
        <button
          type="button"
          className="btn btn-sm btn-quiet"
          title={folder ? `"${folder}" 안에 새 폴더` : '데이터셋 맨 위에 새 폴더'}
          onClick={() => {
            setRenaming(null)
            setCreating(true)
          }}
        >
          <FolderPlus size={13} strokeWidth={1.75} /> 새 폴더
        </button>
      </div>
      <ul className="ft-list" aria-label={`데이터셋 ${dataset.name}의 폴더`}>
        <li className={`ft-row${folder === '' ? ' selected' : ''}`} data-path="">
          <span className="ft-twisty" aria-hidden="true" />
          <button
            type="button"
            className="ft-main"
            aria-current={folder === '' ? 'true' : undefined}
            title="데이터셋 전체 (모든 폴더)"
            onClick={() => onFolder('')}
          >
            <FolderTreeIcon size={13} strokeWidth={1.75} aria-hidden="true" />
            <span className="ft-name">전체</span>
            <span className="ft-count num">{dataset.media_count}</span>
          </button>
        </li>
        {creating && folder === '' ? newForm(1) : null}
        {rows.map((n) => {
          const selected = n.path === folder
          const open = !collapsed.includes(n.path)
          return (
            <Fragment key={n.path}>
              <li className={`ft-row${selected ? ' selected' : ''}`} data-path={n.path} style={{ '--depth': n.depth } as CSSProperties}>
                {n.children.length ? (
                  <button
                    type="button"
                    className="ft-twisty"
                    aria-expanded={open}
                    aria-label={`${n.name} 하위 폴더 ${open ? '접기' : '펼치기'}`}
                    onClick={() => setCollapsed((c) => (open ? [...c, n.path] : c.filter((p) => p !== n.path)))}
                  >
                    {open ? <ChevronDown size={12} strokeWidth={1.75} /> : <ChevronRight size={12} strokeWidth={1.75} />}
                  </button>
                ) : (
                  <span className="ft-twisty" aria-hidden="true" />
                )}
                {renaming === n.path ? (
                  <FolderNameForm
                    initial={n.name}
                    label="폴더 이름"
                    submitLabel="바꾸기"
                    depth={0}
                    pending={rename.isPending}
                    onSubmit={(name) => (name === n.name ? setRenaming(null) : rename.mutate({ path: n.path, to: joinFolder(parentOf(n.path), name) }))}
                    onCancel={() => setRenaming(null)}
                  />
                ) : (
                  <>
                    <button
                      type="button"
                      className="ft-main"
                      aria-current={selected ? 'true' : undefined}
                      title={`${n.path} · 하위 폴더 포함 ${n.total}개`}
                      onClick={() => onFolder(n.path)}
                    >
                      {selected ? (
                        <FolderOpen size={13} strokeWidth={1.75} aria-hidden="true" />
                      ) : (
                        <Folder size={13} strokeWidth={1.75} aria-hidden="true" />
                      )}
                      <span className="ft-name">{n.name}</span>
                      <span className="ft-count num">{n.total}</span>
                    </button>
                    {selected ? (
                      <span className="ft-actions">
                        <button
                          type="button"
                          className="icon-btn sm"
                          aria-label={`폴더 ${n.name} 이름 바꾸기`}
                          title="이름 바꾸기"
                          onClick={() => {
                            setCreating(false)
                            setRenaming(n.path)
                          }}
                        >
                          <Pencil size={12} strokeWidth={1.75} />
                        </button>
                        <ConfirmButton
                          label={`폴더 ${n.name} 삭제 (하위 폴더와 함께, 미디어는 이 데이터셋에서만 빠짐)`}
                          confirmLabel="폴더 삭제"
                          onConfirm={() => remove.mutate(n)}
                          quiet
                        />
                      </span>
                    ) : null}
                  </>
                )}
              </li>
              {creating && selected ? newForm(n.depth + 1) : null}
            </Fragment>
          )
        })}
      </ul>
    </div>
  )
}

function cubesText(it: Pick<BatchItem, 'status' | 'media_kind' | 'summary'>): string {
  if (it.status !== 'done') return STATUS_LABEL[it.status]
  if (!it.summary) return '완료'
  return it.media_kind === 'image' ? `큐브 ${Math.round(it.summary.cubes_mean)}` : `평균 ${it.summary.cubes_mean.toFixed(1)}`
}

/** The batch's score for the dataset's GT set with the shared criteria (docs/SCORING.md §3–4). */
function ScoreChip({ s, gt, loading }: { s: ScoreBatch | null; gt: GtSetInfo; loading: boolean }) {
  if (!s)
    return loading ? (
      <div className="ds-batch-score faint" aria-live="polite">
        채점 중…
      </div>
    ) : null
  const m = s.metrics
  const interim = s.status === 'interim'
  const title =
    `정답셋 ${gt.name} · 허용 오차 충족률 ${pct(m.tolerance_rate.value)} (${ratioNd(m.tolerance_rate)}) · ` +
    `이미지 전체 기준 충족률 ${pct(m.image_pass_rate?.value)} (${ratioNd(m.image_pass_rate)})` +
    (interim ? ` · 중간 점수: 끝난 이미지 ${int(m.n_images ?? 0)}/${int(s.n_eval_images)}` : '') +
    (s.full_coverage ? '' : ` · 부분 범위 ${coverageNd(s, gt.n_images)}`) +
    ' · 잠정 기준 (채점 화면에서 바꿈)'
  return (
    <div className="ds-batch-score num" title={title}>
      <span className="dbs-k">정답</span>
      <span className="dbs-v">
        {pct(m.tolerance_rate.value)} · {pct(m.image_pass_rate?.value)}
      </span>
      <span className={`dbs-state${interim ? ' interim' : ''}`}>{interim ? '중간' : '완료'}</span>
      {s.full_coverage ? null : (
        <span className="dbs-part">
          부분 {coverageNd(s, gt.n_images)}
        </span>
      )}
    </div>
  )
}

function BatchRow({
  batch,
  expanded,
  onToggle,
  onOpenRun,
  onDeleted,
  score,
}: {
  batch: Batch
  expanded: boolean
  onToggle: () => void
  onOpenRun: (mediaId: number, runId: number) => void
  onDeleted: () => void
  score: { s: ScoreBatch | null; gt: GtSetInfo; loading: boolean } | null // null = no GT set
}) {
  const toast = useToast()
  const detail = useBatch(expanded ? batch.id : null)
  const c = batch.counts
  const ended = c.done + c.failed + c.cancelled
  const after = () => {
    invalidateRuns()
    void queryClient.invalidateQueries({ queryKey: keys.datasets })
  }
  const cancel = useMutation({ mutationFn: () => api.cancelBatch(batch.id), onSuccess: after, onError: (e: Error) => toast.error(e.message) })
  const remove = useMutation({
    mutationFn: () => api.deleteBatch(batch.id),
    onSuccess: () => {
      // Drop its detail before the lists refetch, or the expanded row asks for a deleted batch.
      queryClient.removeQueries({ queryKey: keys.batch(batch.id) })
      onDeleted()
      after()
    },
    onError: (e: Error) => toast.error(e.message),
  })
  // After 멈추기 the running item still finishes its current frame: say so until it does.
  const stopping = cancel.isSuccess && !batch.finished
  const state = !batch.finished
    ? stopping
      ? '멈추는 중 · 지금 항목이 끝나면 멈춥니다'
      : c.running > 0
        ? '실행 중'
        : '대기'
    : c.failed + c.cancelled > 0
      ? `완료 ${c.done} · 실패 ${c.failed} · 취소 ${c.cancelled}`
      : '완료'
  return (
    <li className="ds-batch-row">
      <div className="ds-batch-line">
        <button type="button" className="ds-toggle" aria-expanded={expanded} onClick={onToggle}>
          {expanded ? <ChevronDown size={13} strokeWidth={1.75} /> : <ChevronRight size={13} strokeWidth={1.75} />}
          <span className="ds-batch-name" title={`${batch.model_name} · 영상 ${batch.stride}프레임마다`}>
            #{batch.id} {batch.model_name}
          </span>
        </button>
        <BatchScope folder={batch.folder} />
        <span className="num faint">
          {ended}/{c.total}
        </span>
      </div>
      <div className="progress" aria-hidden="true">
        <span style={{ width: `${c.total ? (ended / c.total) * 100 : 0}%` }} />
      </div>
      {score ? <ScoreChip {...score} /> : null}
      <div className="ds-batch-actions">
        <span className="faint ds-state">{state}</span>
        {c.done > 0 ? (
          <a className="btn btn-sm" href={batch.zip_url} download title="결과가 그려진 이미지·영상 + results.json + results.csv">
            <Download size={12} strokeWidth={1.75} /> {batch.finished ? '전체 zip' : '끝난 것 zip'}
          </a>
        ) : null}
        {!batch.finished ? (
          <button type="button" className="btn btn-sm btn-quiet" onClick={() => cancel.mutate()} disabled={cancel.isPending || stopping}>
            <Square size={11} strokeWidth={2} /> 멈추기
          </button>
        ) : null}
        <ConfirmButton label={`일괄 실행 #${batch.id} 삭제`} confirmLabel="기록까지 삭제" onConfirm={() => remove.mutate()} quiet />
      </div>
      {expanded ? (
        <ul className="ds-items" aria-label={`일괄 실행 #${batch.id} 항목`}>
          {detail.isPending ? <li className="faint">불러오는 중…</li> : null}
          {detail.data?.items.map((it) => {
            const below = it.folder ? relativeTo(it.folder, batch.folder ?? '') : '' // where it is in the batch's scope
            return (
              <li key={it.run_id} className="ds-item">
                <button type="button" className="ds-item-name" title={`${it.media_name}: 작업대에서 결과 보기`} onClick={() => onOpenRun(it.media_id, it.run_id)}>
                  {below ? <span className="media-folder">{below}/</span> : null}
                  {it.media_name}
                </button>
                <span className={`num ds-item-state ${it.status}`} title={it.error ?? undefined}>
                  {cubesText(it)}
                </span>
                {it.download_url ? (
                  <a className="icon-btn sm" href={it.download_url} download aria-label={`${it.media_name} 결과 받기`} title="결과 받기">
                    <Download size={12} strokeWidth={1.75} />
                  </a>
                ) : (
                  <span className="icon-btn sm" aria-hidden="true" />
                )}
              </li>
            )
          })}
        </ul>
      ) : null}
    </li>
  )
}

/** Run the current model on the whole dataset or on the selected folder (with its subfolders);
 *  the batches of this dataset with their scope and downloads. */
export function BatchPanel({
  dataset,
  folder,
  model,
  params,
  onOpenRun,
}: {
  dataset: Dataset
  folder: string // '' = the whole dataset
  model: ModelInfo | null
  params: Params
  onOpenRun: (mediaId: number, runId: number) => void
}) {
  const toast = useToast()
  const batches = useBatches(dataset.id)
  const [open, setOpen] = useState(true)
  const [stride, setStride] = useState(1)
  const [expanded, setExpanded] = useState<number | null>(null)
  const changed = model ? changedParams(model.params, params).length : 0
  const count = folder ? (dataset.folders.find((f) => f.path === folder)?.total ?? 0) : dataset.media_count
  const start = useMutation({
    mutationFn: () => api.createBatch(dataset.id, { model_key: model!.key, params, stride, folder: folder || null }),
    onSuccess: (b) => {
      invalidateRuns()
      void queryClient.invalidateQueries({ queryKey: keys.datasets })
      setExpanded(b.id)
      toast.ok(`${b.folder ? `폴더 "${b.folder}"의 ` : ''}${b.counts.total}개를 ${named(b.model_name, '으로/로')} 실행합니다`)
    },
    onError: (e: Error) => toast.error(e.message),
  })
  const list = batches.data ?? []

  // Score chips: one POST /api/score (detail "summary": metrics only, no slices or bootstrap) for
  // every batch of the dataset, only while the list is open; never in the way of the rail.
  const hasScore = useFeature('score')
  const gtSets = useGtSets(dataset.id)
  const [pickedGt] = useGtPick(dataset.id)
  const gt = chooseGt(gtSets.data, pickedGt)
  const { criteria } = useSettledCriteria()
  const scoreReq =
    open && gt && list.length > 0 ? { gt_set_id: gt.id, batch_ids: [], criteria, reference_batch_id: null, detail: 'summary' as const } : null
  const score = useScore(scoreReq, batchStamp(batches.data), list.some((b) => !b.finished))
  const chip = (id: number) =>
    hasScore && gt && !score.isError
      ? {
          s: score.data?.gt_set.id === gt.id ? (score.data.batches.find((x) => x.batch_id === id) ?? null) : null,
          gt,
          loading: score.isFetching,
        }
      : null

  return (
    <section className="ds-batch">
      <button type="button" className="ds-toggle ds-batch-head" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
        {open ? <ChevronDown size={13} strokeWidth={1.75} /> : <ChevronRight size={13} strokeWidth={1.75} />}
        <span>일괄 실행 · 결과 받기</span>
        {list.length ? <span className="num faint">{list.length}</span> : null}
      </button>
      {open ? (
        <>
          <div className="ds-run">
            <button
              type="button"
              className="btn btn-sm btn-primary"
              disabled={!model || count === 0 || start.isPending}
              title={
                folder
                  ? `폴더 "${folder}"와 그 아래 폴더의 ${count}개를 실행합니다`
                  : `데이터셋 "${dataset.name}"의 ${count}개를 모두 실행합니다`
              }
              onClick={() => start.mutate()}
            >
              <Play size={12} strokeWidth={2} /> {folder ? '지금 모델로 이 폴더 실행' : '지금 모델로 전체 실행'}
            </button>
            <p className="faint ds-run-meta">
              {model ? model.name : '모델 없음'} · {folder ? `"${folder}" ` : ''}
              {count}개{changed ? ` · 바꾼 값 ${changed}개` : ' · 기본값'}
            </p>
            {count === 0 ? (
              <p className="ds-run-why">
                {folder
                  ? '이 폴더는 비어 있어 실행할 수 없습니다. 여기서 올리거나 "선택"으로 미디어를 옮겨 넣으세요.'
                  : '데이터셋이 비어 있어 실행할 수 없습니다. 여기서 올리거나 "선택"으로 미디어를 넣으세요.'}
              </p>
            ) : !model ? (
              <p className="ds-run-why">오른쪽에서 모델을 고르면 실행할 수 있습니다.</p>
            ) : null}
            {dataset.video_count > 0 ? (
              <label className="ds-stride faint">
                영상은
                <NumberField integer label="영상 프레임 간격" value={stride} min={1} max={1000} onCommit={setStride} />
                프레임마다
              </label>
            ) : null}
          </div>
          {list.length > 0 ? (
            <ul className="ds-batches">
              {list.map((b) => (
                <BatchRow
                  key={b.id}
                  batch={b}
                  expanded={expanded === b.id}
                  onToggle={() => setExpanded((cur) => (cur === b.id ? null : b.id))}
                  onOpenRun={onOpenRun}
                  onDeleted={() => setExpanded((cur) => (cur === b.id ? null : cur))}
                  score={chip(b.id)}
                />
              ))}
            </ul>
          ) : (
            <p className="faint ds-empty">아직 일괄 실행이 없습니다. 오른쪽에서 모델과 파라미터를 고른 뒤 실행하세요.</p>
          )}
        </>
      ) : null}
    </section>
  )
}
