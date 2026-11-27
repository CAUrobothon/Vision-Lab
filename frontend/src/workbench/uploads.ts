// Media uploads: a sequential queue with per-file progress, a page-wide drop target (files and
// whole directories, read recursively) and ⌘V paste. Files keep the directory they came from, so
// an upload into a dataset reproduces the folder tree.
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { MediaUploadResult } from '../api/types'
import { byName, joinFolder } from '../lib/folders'

/** A file to upload and the directory it came from inside what was picked or dropped ('' = none). */
export interface PickedFile {
  file: File
  dir: string
}

/** Files an OS leaves in folders: Finder's .DS_Store and ._ resource forks, Windows' Thumbs.db. */
function isJunk(name: string): boolean {
  const n = name.toLowerCase()
  return n === '.ds_store' || n === 'thumbs.db' || name.startsWith('._')
}

export function plainFiles(files: File[]): PickedFile[] {
  return files.filter((f) => !isJunk(f.name)).map((file) => ({ file, dir: '' }))
}

/** Files of an <input webkitdirectory>: webkitRelativePath "picked/원본/a.png" -> dir "picked/원본". */
export function directoryFiles(files: File[]): PickedFile[] {
  return files
    .filter((f) => !isJunk(f.name))
    .map((file) => ({ file, dir: (file.webkitRelativePath || file.name).split('/').slice(0, -1).join('/') }))
}

function readBatch(reader: FileSystemDirectoryReader): Promise<FileSystemEntry[]> {
  return new Promise((ok, fail) => reader.readEntries(ok, fail))
}

function entryFile(entry: FileSystemFileEntry): Promise<File> {
  return new Promise((ok, fail) => entry.file(ok, fail))
}

async function readDirectory(dir: FileSystemDirectoryEntry, path: string, out: PickedFile[]): Promise<void> {
  // readEntries hands the entries out in batches (100 in Chromium) and an empty batch at the end.
  const reader = dir.createReader()
  const entries: FileSystemEntry[] = []
  for (let batch = await readBatch(reader); batch.length > 0; batch = await readBatch(reader)) entries.push(...batch)
  entries.sort((a, b) => byName.compare(a.name, b.name))
  for (const entry of entries) {
    if (entry.isDirectory) await readDirectory(entry as FileSystemDirectoryEntry, `${path}/${entry.name}`, out)
    else if (entry.isFile && !isJunk(entry.name)) out.push({ file: await entryFile(entry as FileSystemFileEntry), dir: path })
  }
}

/** Everything dropped: files as they are, directories read recursively. The items are readable only
 *  during the drop event, so this must be called inside it (the reading itself goes on after). */
function droppedFiles(dt: DataTransfer): Promise<PickedFile[]> {
  const taken = Array.from(dt.items ?? [])
    .filter((it) => it.kind === 'file')
    .map((it) => ({ entry: it.webkitGetAsEntry?.() ?? null, file: it.getAsFile() }))
  if (!taken.some((t) => t.entry?.isDirectory)) return Promise.resolve(plainFiles(Array.from(dt.files)))
  return (async () => {
    const out: PickedFile[] = []
    for (const { entry, file } of taken) {
      if (entry?.isDirectory) await readDirectory(entry as FileSystemDirectoryEntry, entry.name, out)
      else if (entry?.isFile) out.push({ file: await entryFile(entry as FileSystemFileEntry), dir: '' })
      else if (file) out.push({ file, dir: '' })
    }
    return out.filter((p) => !isJunk(p.file.name))
  })()
}

/**
 * The directories of picked or dropped files below the folder they go into. One directory on its
 * own adds no level when it only wraps: if it holds nothing but folders (a parent of 원본/ 증강1/),
 * or has the target folder's own name, its contents go straight into the target. Anything else
 * keeps its directories as they are (several dropped folders keep their names).
 */
export function placeFiles(picked: PickedFile[], target: string): PickedFile[] {
  const tops = new Set(picked.map((p) => p.dir.split('/')[0]))
  if (tops.size !== 1 || tops.has('')) return picked
  const [top] = tops
  const onlyFolders = !picked.some((p) => p.dir === top)
  const sameName = top.normalize('NFC') === (target.split('/').pop() ?? '').normalize('NFC')
  if (!onlyFolders && !sameName) return picked
  return picked.map((p) => ({ ...p, dir: p.dir.slice(top.length + 1) }))
}

export interface UploadItem {
  id: number
  name: string
  loaded: number
  total: number
  state: 'waiting' | 'uploading' | 'processing' | 'done' | 'error'
  error?: string
}

interface UploadJob {
  file: File
  folder: string // in the dataset; '' = its top level
}

let nextId = 1
let nextGroup = 1

/** `onUploaded` runs after each file; `onGroupDone` once every file of one `add` call has ended,
 *  with the results of those that went up. */
export function useUploads(
  onUploaded: (res: MediaUploadResult) => void,
  onGroupDone: (results: MediaUploadResult[]) => void,
) {
  const [items, setItems] = useState<UploadItem[]>([])
  const queue = useRef<{ id: number; file: File; datasetId: number | null; folder: string; group: number }[]>([])
  const groups = useRef(new Map<number, { left: number; results: MediaUploadResult[] }>())
  const busy = useRef(false)
  const cb = useRef(onUploaded)
  const groupCb = useRef(onGroupDone)
  useLayoutEffect(() => {
    cb.current = onUploaded
    groupCb.current = onGroupDone
  })

  const update = useCallback((id: number, patch: Partial<UploadItem>) => {
    setItems((prev) => prev.map((it) => (it.id === id ? { ...it, ...patch } : it)))
  }, [])

  const pump = useCallback(async () => {
    if (busy.current) return
    busy.current = true
    while (queue.current.length > 0) {
      const { id, file, datasetId, folder, group } = queue.current.shift()!
      const g = groups.current.get(group)!
      update(id, { state: 'uploading' })
      try {
        const res = await api.uploadMedia(
          file,
          (p) => update(id, { loaded: p.loaded, total: p.total, state: p.loaded >= p.total ? 'processing' : 'uploading' }),
          datasetId,
          folder,
        )
        update(id, { state: 'done', loaded: file.size, total: file.size })
        g.results.push(res)
        cb.current(res)
        window.setTimeout(() => setItems((prev) => prev.filter((it) => it.id !== id)), 2500)
      } catch (err) {
        update(id, { state: 'error', error: (err as Error).message })
      }
      if (--g.left === 0) {
        groups.current.delete(group)
        groupCb.current(g.results)
      }
    }
    busy.current = false
  }, [update])

  /** Queue files; with a dataset id they also go into that dataset, each into its folder. */
  const add = useCallback(
    (jobs: UploadJob[], datasetId: number | null = null) => {
      if (jobs.length === 0) return
      const group = nextGroup++
      groups.current.set(group, { left: jobs.length, results: [] })
      const fresh = jobs.map(({ file, folder }) => ({ id: nextId++, file, datasetId, folder: datasetId === null ? '' : folder, group }))
      queue.current.push(...fresh)
      setItems((prev) => [
        ...prev,
        ...fresh.map(({ id, file, folder }) => ({
          id,
          name: joinFolder(folder, file.name),
          loaded: 0,
          total: file.size,
          state: 'waiting' as const,
        })),
      ])
      void pump()
    },
    [pump],
  )

  const dismiss = useCallback((id: number) => setItems((prev) => prev.filter((it) => it.id !== id)), [])

  return { items, add, dismiss }
}

/** True while files are dragged over the window; dropped files and directories go to `onFiles`. */
export function useWindowFileDrop(onFiles: (files: PickedFile[]) => void): boolean {
  const [dragging, setDragging] = useState(false)
  const cb = useRef(onFiles)
  useLayoutEffect(() => {
    cb.current = onFiles
  })
  useEffect(() => {
    let depth = 0
    const hasFiles = (e: DragEvent) => !!e.dataTransfer && Array.from(e.dataTransfer.types).includes('Files')
    const onEnter = (e: DragEvent) => {
      if (!hasFiles(e)) return
      e.preventDefault()
      depth++
      setDragging(true)
    }
    const onOver = (e: DragEvent) => {
      if (!hasFiles(e)) return
      e.preventDefault()
      e.dataTransfer!.dropEffect = 'copy'
    }
    const onLeave = (e: DragEvent) => {
      if (!hasFiles(e)) return
      depth = Math.max(0, depth - 1)
      if (depth === 0) setDragging(false)
    }
    const onDrop = (e: DragEvent) => {
      if (!hasFiles(e)) return
      e.preventDefault()
      depth = 0
      setDragging(false)
      void droppedFiles(e.dataTransfer!).then((files) => {
        if (files.length) cb.current(files)
      })
    }
    window.addEventListener('dragenter', onEnter)
    window.addEventListener('dragover', onOver)
    window.addEventListener('dragleave', onLeave)
    window.addEventListener('drop', onDrop)
    return () => {
      window.removeEventListener('dragenter', onEnter)
      window.removeEventListener('dragover', onOver)
      window.removeEventListener('dragleave', onLeave)
      window.removeEventListener('drop', onDrop)
    }
  }, [])
  return dragging
}

/** ⌘V / Ctrl+V with an image on the clipboard uploads it (text pastes are left alone). */
export function usePasteFiles(onFiles: (files: File[]) => void) {
  const cb = useRef(onFiles)
  useLayoutEffect(() => {
    cb.current = onFiles
  })
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const files = Array.from(e.clipboardData?.files ?? []).filter((f) => f.type.startsWith('image/') || f.type.startsWith('video/'))
      if (files.length === 0) return
      e.preventDefault()
      const stamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '')
      cb.current(
        files.map((f, i) => {
          const ext = f.type.split('/')[1]?.replace('jpeg', 'jpg') ?? 'png'
          return new File([f], `붙여넣기-${stamp}${files.length > 1 ? `-${i + 1}` : ''}.${ext}`, { type: f.type })
        }),
      )
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [])
}
