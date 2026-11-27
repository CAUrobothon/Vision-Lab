// Left rail: datasets and their folders, upload (files or a whole directory), filters, search and
// the media list (with a selection mode to put media into a dataset, move them between its folders
// or take them out).
import { memo, useCallback, useDeferredValue, useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { CheckSquare, Download, Film, FolderUp, Image as ImageIcon, Search, Upload, X } from 'lucide-react'
import { api } from '../api/client'
import { useFeature } from '../api/features'
import { keys, queryClient, useDatasets, useMediaList } from '../api/queries'
import type { Media, MediaKind, ModelInfo, Params } from '../api/types'
import { joinFolder, relativeTo } from '../lib/folders'
import { formatDuration } from '../lib/format'
import { ConfirmButton, NameForm, StarButton } from '../ui/controls'
import { useToast } from '../ui/toast'
import { BatchPanel, DatasetBar, applyDataset, invalidateDatasets } from './DatasetPanel'
import { directoryFiles, plainFiles, type PickedFile, type UploadItem } from './uploads'
import { named } from '../lib/josa'

type Filter = 'all' | 'image' | 'video' | 'starred'

const FILTERS: [Filter, string][] = [
  ['all', '전체'],
  ['image', '이미지'],
  ['video', '영상'],
  ['starred', '별표'],
]

export const ACCEPT = 'image/*,video/*,.heic,.mkv,.avi,.mov,.m4v,.webm'

export function mediaMeta(m: Media): string {
  const size = `${m.width}×${m.height}`
  if (m.kind === 'image') return `이미지 · ${size}`
  return `영상 ${formatDuration(m.duration)} · ${size} · ${m.fps ? Math.round(m.fps) : '?'}fps`
}

function railMeta(m: Media): [string, string] {
  const runs = m.run_count > 0 ? `기록 ${m.run_count}` : '기록 없음'
  if (m.kind === 'image') return [`이미지 · ${m.width}×${m.height}`, runs]
  return [`영상 ${formatDuration(m.duration)} · ${m.width}×${m.height}`, `${m.frame_count ?? '?'}프레임 · ${runs}`]
}

interface MediaRailProps {
  selectedId: number | null
  onSelect: (id: number | null) => void
  uploads: UploadItem[]
  onDismissUpload: (id: number) => void
  onFiles: (files: PickedFile[]) => void // goes into the selected dataset and folder, if any
  datasetId: number | null
  onDataset: (id: number | null) => void
  folder: string // selected folder of the dataset ('' = all of it)
  onFolder: (path: string) => void
  model: ModelInfo | null // what "run on the whole dataset" uses
  params: Params
  onOpenRun: (mediaId: number, runId: number) => void
}

function UploadRow({ item, onDismiss }: { item: UploadItem; onDismiss: () => void }) {
  const pct = item.total > 0 ? Math.round((item.loaded / item.total) * 100) : 0
  const label =
    item.state === 'waiting'
      ? '대기'
      : item.state === 'uploading'
        ? `${pct}%`
        : item.state === 'processing'
          ? '변환 중'
          : item.state === 'done'
            ? '완료'
            : '실패'
  return (
    <li className={`upload-row ${item.state}`}>
      <div className="upload-line">
        <span className="upload-name" title={item.name}>
          {item.name}
        </span>
        <span className="num faint">{label}</span>
        {item.state === 'error' ? (
          <button type="button" className="icon-btn sm" aria-label="닫기" onClick={onDismiss}>
            <X size={12} strokeWidth={1.75} />
          </button>
        ) : null}
      </div>
      {item.state === 'error' ? (
        <p className="error-text">{item.error}</p>
      ) : (
        <div className="progress" aria-hidden="true">
          <span style={{ width: `${item.state === 'processing' || item.state === 'done' ? 100 : pct}%` }} />
        </div>
      )}
    </li>
  )
}

const MediaRow = memo(function MediaRow({
  m,
  folder,
  selected,
  onSelect,
  onDeleted,
  checking,
  checked,
  onCheck,
}: {
  m: Media
  folder: string // where it is below the selected folder ('' = in it)
  selected: boolean
  onSelect: (id: number) => void
  onDeleted: (id: number) => void
  checking: boolean // selection mode: a click ticks the row instead of opening it
  checked: boolean
  onCheck: (id: number) => void
}) {
  const toast = useToast()
  const patch = useMutation({
    mutationFn: (starred: boolean) => api.patchMedia(m.id, { starred }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: keys.mediaAll }),
    onError: (e: Error) => toast.error(e.message),
  })
  const remove = useMutation({
    mutationFn: () => api.deleteMedia(m.id),
    onSuccess: () => {
      onDeleted(m.id)
      toast.ok(`${named(m.name, '과/와')} 그 기록을 지웠습니다`)
      invalidateDatasets()
      void queryClient.invalidateQueries({ queryKey: keys.runsAll })
    },
    onError: (e: Error) => toast.error(e.message),
  })
  return (
    <li className={`media-row${selected ? ' selected reg' : ''}${checking && checked ? ' checked' : ''}`}>
      <button
        type="button"
        className="media-main"
        aria-current={selected ? 'true' : undefined}
        aria-pressed={checking ? checked : undefined}
        onClick={() => (checking ? onCheck(m.id) : onSelect(m.id))}
      >
        {checking ? <span className={`media-check${checked ? ' on' : ''}`} aria-hidden="true" /> : null}
        <span className="media-thumb">
          <img src={m.thumb_url} alt="" loading="lazy" />
          <span className="media-kind" aria-hidden="true">
            {m.kind === 'video' ? <Film size={11} strokeWidth={2} /> : <ImageIcon size={11} strokeWidth={2} />}
          </span>
        </span>
        <span className="media-text">
          <span className="media-name" title={joinFolder(folder, m.name)}>
            {folder ? <span className="media-folder">{folder}/</span> : null}
            {m.name}
          </span>
          <span className="media-meta num">{railMeta(m)[0]}</span>
          <span className="media-meta num">{railMeta(m)[1]}</span>
        </span>
      </button>
      <StarButton on={m.starred} label={m.name} onToggle={() => patch.mutate(!m.starred)} />
      {selected && !checking ? (
        <div className="media-actions">
          <a className="btn btn-sm btn-quiet" href={m.original_url} download>
            <Download size={12} strokeWidth={1.75} /> 원본
          </a>
          <ConfirmButton label={`${m.name} 삭제`} confirmLabel="기록까지 삭제" onConfirm={() => remove.mutate()} quiet />
        </div>
      ) : null}
    </li>
  )
})

function BulkBar({
  ids,
  items,
  datasetId,
  onAll,
  onDone,
  onClear,
}: {
  ids: number[]
  items: Media[]
  datasetId: number | null
  onAll: () => void
  onDone: () => void
  onClear: () => void
}) {
  const toast = useToast()
  const hasFolders = useFeature('folders')
  const datasets = useDatasets()
  const [naming, setNaming] = useState(false)
  const others = (datasets.data ?? []).filter((d) => d.id !== datasetId)
  const current = (datasets.data ?? []).find((d) => d.id === datasetId) ?? null
  // Selection mode stays on after adding, so the same files can go into a second dataset; after
  // "여기서 빼기" the removed files leave the list, so the selection is cleared.
  const finish = (text: string, clear: boolean) => {
    toast.ok(text)
    setNaming(false)
    if (clear) onClear()
  }
  const add = useMutation({
    mutationFn: (target: number) => api.addToDataset(target, ids),
    onSuccess: (d) => {
      invalidateDatasets()
      finish(`${ids.length}개를 "${d.name}"에 넣었습니다`, false)
    },
    onError: (e: Error) => toast.error(e.message),
  })
  const create = useMutation({
    mutationFn: (name: string) => api.createDataset({ name, media_ids: ids }),
    onSuccess: (d) => {
      invalidateDatasets()
      finish(`${ids.length}개로 데이터셋 ${named(d.name, '을/를')} 만들었습니다`, false)
    },
    onError: (e: Error) => toast.error(e.message),
  })
  const remove = useMutation({
    mutationFn: () => api.removeFromDataset(datasetId!, ids),
    onSuccess: (d) => {
      invalidateDatasets()
      finish(`${ids.length}개를 "${d.name}"에서 뺐습니다 (미디어는 남아 있습니다)`, true)
    },
    onError: (e: Error) => toast.error(e.message),
  })
  // Moved files may leave the folder on screen, so the selection is cleared.
  const move = useMutation({
    mutationFn: (folder: string) => api.moveMedia(datasetId!, ids, folder),
    onSuccess: (d, folder) => {
      applyDataset(d) // the dataset as it came back; lists, GT links and scores refetched
      finish(`${ids.length}개를 ${folder ? named(folder, '으로/로') : '맨 위로'} 옮겼습니다`, true)
    },
    onError: (e: Error) => toast.error(e.message),
  })
  const none = ids.length === 0
  return (
    <div className="bulk-bar" role="region" aria-label="선택한 미디어">
      <div className="ds-row">
        <span className="num">{ids.length}개 선택</span>
        <button type="button" className="btn btn-sm btn-quiet" onClick={onAll} disabled={items.length === 0}>
          {ids.length === items.length && items.length > 0 ? '모두 해제' : '모두 선택'}
        </button>
        <button type="button" className="btn btn-sm btn-quiet bulk-done" onClick={onDone}>
          완료
        </button>
      </div>
      {naming ? (
        <NameForm
          className="ds-row"
          label="새 데이터셋 이름"
          placeholder="새 데이터셋 이름"
          submitLabel="만들기"
          pending={create.isPending}
          cancel="none"
          onSubmit={(name) => create.mutate(name)}
          onCancel={() => setNaming(false)}
        />
      ) : (
        <div className="ds-row">
          <select
            className="select"
            aria-label="선택한 미디어를 넣을 데이터셋"
            value=""
            disabled={none}
            onChange={(e) => {
              const v = e.target.value
              if (v === 'new') setNaming(true)
              else if (v) add.mutate(Number(v))
            }}
          >
            <option value="">데이터셋에 넣기…</option>
            {others.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
            <option value="new">＋ 새 데이터셋으로 묶기</option>
          </select>
          {datasetId !== null ? (
            <button type="button" className="btn btn-sm" disabled={none || remove.isPending} onClick={() => remove.mutate()}>
              여기서 빼기
            </button>
          ) : null}
        </div>
      )}
      {current && hasFolders && !naming ? (
        <div className="ds-row">
          <select
            className="select"
            aria-label="선택한 미디어를 옮길 폴더"
            value=""
            disabled={none || move.isPending}
            onChange={(e) => {
              // Values carry a prefix so the top level ('') differs from the placeholder.
              if (e.target.value) move.mutate(e.target.value.slice(1))
            }}
          >
            <option value="">폴더로 옮기기…</option>
            <option value="/">맨 위 (폴더 밖)</option>
            {current.folders.map((f) => (
              <option key={f.path} value={`/${f.path}`}>
                {f.path}
              </option>
            ))}
          </select>
        </div>
      ) : null}
    </div>
  )
}

export function MediaRail({
  selectedId,
  onSelect,
  uploads,
  onDismissUpload,
  onFiles,
  datasetId,
  onDataset,
  folder,
  onFolder,
  model,
  params,
  onOpenRun,
}: MediaRailProps) {
  const [filter, setFilter] = useState<Filter>('all')
  const [q, setQ] = useState('')
  const deferredQ = useDeferredValue(q.trim())
  const fileRef = useRef<HTMLInputElement>(null)
  const dirRef = useRef<HTMLInputElement | null>(null)
  const kind: MediaKind | undefined = filter === 'image' || filter === 'video' ? filter : undefined
  const list = useMediaList({
    dataset_id: datasetId ?? undefined,
    folder: datasetId !== null && folder ? folder : undefined,
    kind,
    starred: filter === 'starred' ? true : undefined,
    q: deferredQ || undefined,
  })
  const items = list.data ?? []
  const onDeleted = useCallback((id: number) => id === selectedId && onSelect(null), [selectedId, onSelect])
  const datasets = useDatasets()
  const dataset = datasets.data?.find((d) => d.id === datasetId) ?? null
  const target = dataset ? `"${joinFolder(dataset.name, folder)}"` : ''
  // Where each listed media is below the selected folder ('' = in it).
  const below = (m: Media) => (m.folder ? relativeTo(m.folder, folder) : '')

  const [checking, setChecking] = useState(false)
  const [checked, setChecked] = useState<number[]>([])
  const onCheck = useCallback((id: number) => setChecked((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id])), [])
  const visibleChecked = checked.filter((id) => items.some((m) => m.id === id))
  const stopChecking = () => {
    setChecking(false)
    setChecked([])
  }

  return (
    <div className="rail">
      <div className="rail-head">
        <h2 className="panel-title">미디어</h2>
        <span className="rail-head-actions">
          <button
            type="button"
            className="btn btn-sm btn-quiet"
            aria-pressed={checking}
            title="여러 개를 골라 데이터셋에 넣거나 빼기"
            onClick={() => (checking ? stopChecking() : setChecking(true))}
          >
            <CheckSquare size={13} strokeWidth={1.75} /> 선택
          </button>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => fileRef.current?.click()}
            title={dataset ? `올린 파일은 데이터셋 ${target}에도 들어갑니다` : undefined}
          >
            <Upload size={13} strokeWidth={1.75} /> 올리기
          </button>
          <button
            type="button"
            className="icon-btn"
            aria-label="폴더 올리기"
            title={
              dataset
                ? `폴더를 통째로 올립니다. 그 안의 폴더 구조가 ${target} 아래에 그대로 생깁니다`
                : '폴더 안의 파일을 모두 올립니다 (폴더 구조는 데이터셋을 고르면 남습니다)'
            }
            onClick={() => dirRef.current?.click()}
          >
            <FolderUp size={15} strokeWidth={1.75} />
          </button>
        </span>
        <input
          ref={fileRef}
          type="file"
          multiple
          accept={ACCEPT}
          hidden
          onChange={(e) => {
            const files = Array.from(e.target.files ?? [])
            e.target.value = ''
            if (files.length) onFiles(plainFiles(files))
          }}
        />
        <input
          ref={(el) => {
            // React has no prop for it: the attribute makes the picker choose a directory.
            el?.setAttribute('webkitdirectory', '')
            dirRef.current = el
          }}
          type="file"
          multiple
          hidden
          onChange={(e) => {
            const files = Array.from(e.target.files ?? [])
            e.target.value = ''
            const picked = directoryFiles(files)
            if (picked.length) onFiles(picked)
          }}
        />
      </div>
      <DatasetBar datasetId={datasetId} onDataset={onDataset} folder={folder} onFolder={onFolder} />
      {dataset ? <BatchPanel key={dataset.id} dataset={dataset} folder={folder} model={model} params={params} onOpenRun={onOpenRun} /> : null}
      <div className="rail-filters">
        <label className="search">
          <Search size={13} strokeWidth={1.75} aria-hidden="true" />
          <input className="input" type="search" value={q} placeholder="이름으로 찾기" aria-label="미디어 이름 검색" onChange={(e) => setQ(e.target.value)} />
        </label>
        <div className="chips" role="group" aria-label="미디어 종류">
          {FILTERS.map(([f, label]) => (
            <button key={f} type="button" className="chip" aria-pressed={filter === f} onClick={() => setFilter(f)}>
              {label}
            </button>
          ))}
        </div>
      </div>
      {uploads.length > 0 ? (
        <ul className="upload-list" aria-label="올리는 중">
          {uploads.map((it) => (
            <UploadRow key={it.id} item={it} onDismiss={() => onDismissUpload(it.id)} />
          ))}
        </ul>
      ) : null}
      <div className="rail-list">
        {list.isPending ? (
          <ul className="media-list" aria-hidden="true">
            {[0, 1, 2, 3].map((i) => (
              <li key={i} className="media-row skeleton-row">
                <span className="media-thumb skeleton" />
                <span className="media-text">
                  <span className="skeleton sk-line" />
                  <span className="skeleton sk-line short" />
                </span>
              </li>
            ))}
          </ul>
        ) : list.isError ? (
          <p className="rail-empty error-text">{list.error.message}</p>
        ) : items.length === 0 ? (
          <p className="rail-empty faint">
            {filter !== 'all' || deferredQ
              ? '조건에 맞는 미디어가 없습니다.'
              : dataset && folder
                ? '이 폴더는 비어 있습니다. 여기서 올리거나, "선택"으로 다른 폴더의 미디어를 옮겨 넣으세요.'
                : dataset
                  ? '이 데이터셋은 비어 있습니다. 여기서 올리거나, "선택"으로 다른 미디어를 골라 넣으세요.'
                  : '아직 올린 미디어가 없습니다. 파일을 끌어다 놓거나 붙여넣으세요.'}
          </p>
        ) : (
          <ul className="media-list">
            {items.map((m) => (
              <MediaRow
                key={m.id}
                m={m}
                folder={below(m)}
                selected={m.id === selectedId}
                onSelect={onSelect}
                onDeleted={onDeleted}
                checking={checking}
                checked={checked.includes(m.id)}
                onCheck={onCheck}
              />
            ))}
          </ul>
        )}
      </div>
      {checking ? (
        <BulkBar
          ids={visibleChecked}
          items={items}
          datasetId={datasetId}
          onAll={() => setChecked(visibleChecked.length === items.length ? [] : items.map((m) => m.id))}
          onDone={stopChecking}
          onClear={() => setChecked([])}
        />
      ) : (
        <p className="rail-foot faint">
          끌어다 놓기 · <span className="kbd">⌘V</span> 붙여넣기{dataset ? ` → ${target}` : ''}
        </p>
      )}
    </div>
  )
}
