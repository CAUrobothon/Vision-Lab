// The selected dataset's ground-truth sets in the rail (docs/SCORING.md §1): upload a GT JSON, pick
// / rename / delete a set, its link count, the link-check dialog (GT images that no dataset image
// matches by sha256, with the same-name images as candidates: nothing links until confirmed), and
// moving the linked media into one folder per augmentation. A server without these routes (started
// before they existed) shows nothing here.
import { useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { AlertTriangle, Link2, Pencil, Unlink, Upload } from 'lucide-react'
import { api } from '../api/client'
import { useFeature } from '../api/features'
import { invalidateGt, keys, queryClient, useGtSet, useGtSets } from '../api/queries'
import type { Dataset, GtSetDetail, GtSetInfo, GtUnlinkedImage } from '../api/types'
import { joinFolder } from '../lib/folders'
import { named } from '../lib/josa'
import { chooseGt, useGtPick } from '../score/criteria'
import { ConfirmButton, Modal, NameForm } from '../ui/controls'
import { useToast } from '../ui/toast'
import { applyDataset } from './DatasetPanel'

function linkedCount(s: GtSetInfo): number {
  return s.links.linked_sha256 + s.links.linked_confirmed
}

function place(m: { name: string; folder: string }): string {
  return joinFolder(m.folder, m.name)
}

/** The set as the link change returned it, in the cache; every other GT query and score refetched. */
function afterLinks(gid: number, d: GtSetDetail) {
  queryClient.setQueryData(keys.gtSet(gid), d)
  invalidateGt(keys.gtSet(gid))
}

function LinkDialog({ set, onClose }: { set: GtSetInfo; onClose: () => void }) {
  const toast = useToast()
  const detail = useGtSet(set.id)
  const d = detail.data
  // The media chosen for each unlinked GT image (default: its first candidate).
  const [chosen, setChosen] = useState<Record<number, number | ''>>({})
  const pick = (u: GtUnlinkedImage): number | '' => chosen[u.gt_index] ?? u.candidates[0]?.media_id ?? ''

  const confirm = useMutation({
    mutationFn: (links: { gt_index: number; media_id: number }[]) => api.confirmGtLinks(set.id, links),
    onSuccess: (res, links) => {
      afterLinks(set.id, res)
      setChosen({})
      toast.ok(`정답 이미지 ${links.length}장을 연결했습니다`)
    },
    onError: (e: Error) => toast.error(e.message),
  })
  const unlink = useMutation({
    mutationFn: (gtIndex: number) => api.removeGtLink(set.id, gtIndex),
    onSuccess: (res) => afterLinks(set.id, res),
    onError: (e: Error) => toast.error(e.message),
  })

  const unlinked = d?.unlinked ?? []
  const confirmed = d?.linked.filter((l) => l.method === 'confirmed') ?? []
  const free = d?.images_without_gt ?? []
  // "후보 모두 연결": every row with a choice, each media once (the server refuses the lot otherwise).
  const all: { gt_index: number; media_id: number }[] = []
  const used = new Set<number>()
  for (const u of unlinked) {
    const m = pick(u)
    if (m !== '' && !used.has(m)) {
      used.add(m)
      all.push({ gt_index: u.gt_index, media_id: m })
    }
  }
  const l = set.links
  return (
    <Modal
      top
      className="gl-dialog"
      onClose={onClose}
      title={
        <>
          정답 연결 확인 <span className="pj-model">· {set.name}</span>
        </>
      }
    >
      <p className="gl-summary num">
        연결 <strong>{linkedCount(set)}/{l.gt_images}</strong> — sha256 {l.linked_sha256} · 확인 {l.linked_confirmed} · 연결 안 됨 {l.unlinked} · 정답 없는
        이미지 {l.dataset_images_without_gt}
        {l.dataset_videos ? ` · 영상 ${l.dataset_videos} (채점 안 함)` : ''}
      </p>
      <p className="notice">
        <Link2 size={14} strokeWidth={1.75} />
        <span>
          sha256이 같은 이미지는 저절로 연결됩니다. 아래 후보는 파일 이름만 같은 이미지라 확인하기 전에는 연결되지 않습니다 — 같은 사진일 때만
          연결하세요. 이름을 바꾼 파일은 목록에서 정답 없는 이미지를 직접 고를 수 있습니다.
        </span>
      </p>

      {detail.isPending ? <p className="faint">불러오는 중…</p> : detail.isError ? <p className="error-text">{detail.error.message}</p> : null}

      {d ? (
        <>
          <div className="gl-head">
            <h3>연결 안 된 정답 이미지 {unlinked.length}</h3>
            <button
              type="button"
              className="btn btn-sm btn-primary"
              disabled={all.length === 0 || confirm.isPending}
              onClick={() => confirm.mutate(all)}
              title={all.length ? `고른 후보 ${all.length}개를 한꺼번에 연결합니다` : '고른 후보가 없습니다'}
            >
              후보 모두 연결
            </button>
          </div>
          {unlinked.length === 0 ? (
            <p className="faint gl-empty">정답 이미지가 모두 연결되어 있습니다.</p>
          ) : (
            <ul className="gl-list" aria-label="연결 안 된 정답 이미지">
              {unlinked.map((u) => {
                const m = pick(u)
                // The dataset's images without GT that are not already among its candidates.
                const others = free.filter((f) => !u.candidates.some((cand) => cand.media_id === f.media_id))
                return (
                  <li key={u.gt_index} className="gl-row">
                    <span className="gl-name" title={`정답 이미지 ${u.gt_index}번 · ${u.filename}`}>
                      {u.filename}
                      <span className="faint"> · {u.variant}</span>
                    </span>
                    <span className="gl-cand-label faint">{u.candidates.length ? `후보 ${u.candidates.length}` : '후보 없음'}</span>
                    <select
                      className="select gl-cand"
                      aria-label={`${u.filename} 후보`}
                      value={m}
                      onChange={(e) => setChosen((c) => ({ ...c, [u.gt_index]: e.target.value ? Number(e.target.value) : '' }))}
                    >
                      <option value="">{u.candidates.length ? '연결하지 않음' : '고르기…'}</option>
                      {u.candidates.length ? (
                        <optgroup label="파일 이름이 같은 이미지">
                          {u.candidates.map((cand) => (
                            <option key={cand.media_id} value={cand.media_id}>
                              {place(cand)}
                            </option>
                          ))}
                        </optgroup>
                      ) : null}
                      {others.length ? (
                        <optgroup label="정답 없는 다른 이미지">
                          {others.map((f) => (
                            <option key={f.media_id} value={f.media_id}>
                              {place(f)}
                            </option>
                          ))}
                        </optgroup>
                      ) : null}
                    </select>
                    <button
                      type="button"
                      className="btn btn-sm"
                      disabled={m === '' || confirm.isPending}
                      onClick={() => m !== '' && confirm.mutate([{ gt_index: u.gt_index, media_id: m }])}
                    >
                      연결
                    </button>
                  </li>
                )
              })}
            </ul>
          )}

          <div className="gl-head">
            <h3>확인한 연결 {confirmed.length}</h3>
          </div>
          {confirmed.length === 0 ? (
            <p className="faint gl-empty">확인해서 연결한 이미지가 없습니다 (sha256 연결은 끊을 수 없습니다).</p>
          ) : (
            <ul className="gl-list gl-confirmed" aria-label="확인한 연결">
              {confirmed.map((c) => (
                <li key={c.gt_index} className="gl-row">
                  <span className="gl-name">
                    {c.filename}
                    <span className="faint"> · {c.variant}</span>
                  </span>
                  <span className="gl-arrow faint">↔</span>
                  <span className="gl-media" title={place(c.media)}>
                    {place(c.media)}
                  </span>
                  <button
                    type="button"
                    className="btn btn-sm btn-quiet"
                    disabled={unlink.isPending}
                    aria-label={`${c.filename} 연결 끊기`}
                    onClick={() => unlink.mutate(c.gt_index)}
                  >
                    <Unlink size={12} strokeWidth={1.75} /> 연결 끊기
                  </button>
                </li>
              ))}
            </ul>
          )}
          {l.ignored.length ? (
            <>
              <div className="gl-head">
                <h3>쓰이지 않는 확인 연결 {l.ignored.length}</h3>
              </div>
              <ul className="gl-list" aria-label="쓰이지 않는 확인 연결">
                {l.ignored.map((g) => (
                  <li key={`${g.gt_index}-${g.media_id}`} className="gl-row gl-ignored">
                    <span className="gl-name">정답 이미지 {g.gt_index}번 · 미디어 #{g.media_id}</span>
                    <span className="faint">{g.reason}</span>
                  </li>
                ))}
              </ul>
            </>
          ) : null}
        </>
      ) : null}
    </Modal>
  )
}

function RenameForm({ set, onDone }: { set: GtSetInfo; onDone: () => void }) {
  const toast = useToast()
  const rename = useMutation({
    mutationFn: (n: string) => api.renameGtSet(set.id, n),
    onSuccess: () => {
      invalidateGt()
      onDone()
    },
    onError: (e: Error) => toast.error(e.message),
  })
  return (
    <NameForm
      className="ds-row"
      initial={set.name}
      label="정답셋 이름"
      maxLength={200}
      submitLabel="바꾸기"
      pending={rename.isPending}
      cancel="icon"
      selectOnFocus
      onSubmit={(name) => (name !== set.name ? rename.mutate(name) : onDone())}
      onCancel={onDone}
    />
  )
}

export function GtRow({ dataset }: { dataset: Dataset }) {
  const toast = useToast()
  const hasGt = useFeature('gt')
  const sets = useGtSets(dataset.id)
  const [picked, pickGt] = useGtPick(dataset.id)
  const set = chooseGt(sets.data, picked)
  const fileRef = useRef<HTMLInputElement>(null)
  const [renaming, setRenaming] = useState(false)
  const [dialog, setDialog] = useState(false)

  const upload = useMutation({
    mutationFn: (file: File) => api.uploadGt(dataset.id, file),
    onSuccess: (s) => {
      queryClient.setQueryData<GtSetInfo[]>(keys.gtSets(dataset.id), (old) => [s, ...(old ?? [])])
      pickGt(s.id)
      invalidateGt()
      const n = linkedCount(s)
      toast.ok(
        `정답셋 ${named(s.name, '을/를')} 올렸습니다: 이미지 ${s.n_images}장 중 ${n}장 연결` +
          (s.links.unlinked ? ` · 연결 안 됨 ${s.links.unlinked}장은 "연결 안 됨"을 눌러 확인하세요` : ''),
      )
    },
    onError: (e: Error) => toast.error(e.message),
  })
  const remove = useMutation({
    mutationFn: (s: GtSetInfo) => api.deleteGtSet(s.id),
    onSuccess: (_, s) => {
      queryClient.setQueryData<GtSetInfo[]>(keys.gtSets(dataset.id), (old) => old?.filter((x) => x.id !== s.id))
      invalidateGt()
      toast.ok(`정답셋 ${named(s.name, '을/를')} 지웠습니다. 미디어와 실행 기록은 그대로입니다`)
    },
    onError: (e: Error) => toast.error(e.message),
  })
  const organize = useMutation({
    mutationFn: (s: GtSetInfo) => api.organizeGt(s.id, ''),
    onSuccess: (d, s) => {
      applyDataset(d) // GT links and scores follow the moved media
      toast.ok(`연결된 이미지 ${linkedCount(s)}장을 증강 종류별 폴더 ${s.variants.length}개로 옮겼습니다`)
    },
    onError: (e: Error) => toast.error(e.message),
  })

  // A server without ground truth (or one that has not said yet): nothing to show.
  if (!hasGt) return null

  const input = (
    <input
      ref={fileRef}
      type="file"
      accept=".json,application/json"
      hidden
      onChange={(e) => {
        const f = e.target.files?.[0]
        e.target.value = ''
        if (f) upload.mutate(f)
      }}
    />
  )

  return (
    <section className="gt-row" aria-label="정답셋">
      {input}
      {sets.isPending ? (
        <div className="ds-row">
          <span className="gt-title">정답셋</span>
          <span className="faint gt-hint">불러오는 중…</span>
        </div>
      ) : sets.isError ? (
        <p className="error-text">{sets.error.message}</p>
      ) : !set ? (
        <div className="ds-row">
          <span className="gt-title">정답셋</span>
          <span className="faint gt-hint">정답셋 없음</span>
          <button
            type="button"
            className="btn btn-sm"
            disabled={upload.isPending}
            onClick={() => fileRef.current?.click()}
            title="이 데이터셋의 정답 JSON (robothon_top_face_gt_v1.1 · _augmented_v1.0)을 올리면 일괄 실행을 채점합니다"
          >
            <Upload size={12} strokeWidth={1.75} /> {upload.isPending ? '올리는 중…' : '정답 JSON 올리기'}
          </button>
        </div>
      ) : renaming ? (
        <RenameForm set={set} onDone={() => setRenaming(false)} />
      ) : (
        <>
          <div className="ds-row">
            <span className="gt-title">정답셋</span>
            <select className="select" aria-label="정답셋 선택" value={set.id} title={set.name} onChange={(e) => pickGt(Number(e.target.value))}>
              {sets.data!.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name} · {s.n_images}장
                </option>
              ))}
            </select>
          </div>
          <div className="ds-row gt-links num">
            <span className={set.links.unlinked ? '' : 'faint'} title={`sha256 ${set.links.linked_sha256} · 확인 ${set.links.linked_confirmed} · 정답 없는 이미지 ${set.links.dataset_images_without_gt}`}>
              연결 {linkedCount(set)}/{set.links.gt_images}
            </span>
            {set.links.unlinked ? (
              <button type="button" className="btn btn-sm gt-warn" onClick={() => setDialog(true)}>
                <AlertTriangle size={12} strokeWidth={1.75} /> 연결 안 됨 {set.links.unlinked}
              </button>
            ) : (
              <button type="button" className="btn btn-sm btn-quiet" onClick={() => setDialog(true)}>
                연결 확인
              </button>
            )}
            {set.links.ignored.length ? (
              <span className="gt-ignored" title={set.links.ignored.map((g) => `${g.gt_index}번: ${g.reason}`).join('\n')}>
                무시 {set.links.ignored.length}
              </span>
            ) : null}
            <span className="gt-actions">
              <button
                type="button"
                className="icon-btn sm"
                aria-label="정답셋 더 올리기"
                title="다른 정답 JSON 올리기"
                disabled={upload.isPending}
                onClick={() => fileRef.current?.click()}
              >
                <Upload size={12} strokeWidth={1.75} />
              </button>
              <button type="button" className="icon-btn sm" aria-label="정답셋 이름 바꾸기" title="이름 바꾸기" onClick={() => setRenaming(true)}>
                <Pencil size={12} strokeWidth={1.75} />
              </button>
              <ConfirmButton label={`정답셋 ${set.name} 삭제 (미디어 · 실행 기록은 남음)`} confirmLabel="정답셋 삭제" onConfirm={() => remove.mutate(set)} quiet />
            </span>
          </div>
          {linkedCount(set) > 0 ? (
            <div className="ds-row gt-organize">
              <ConfirmButton
                label="증강 종류별 폴더로 정리"
                confirmLabel={`폴더로 옮기기 (${linkedCount(set)}장)`}
                icon={false}
                disabled={organize.isPending}
                onConfirm={() => organize.mutate(set)}
              />
              <span className="faint gt-hint" title={`데이터셋 맨 위에 증강 이름 폴더를 만들어 옮깁니다: ${set.variants.map((v) => `${v.name} ${v.count}`).join(' · ')}`}>
                {set.variants.length}종 → 맨 위 폴더
              </span>
            </div>
          ) : null}
        </>
      )}
      {dialog && set ? <LinkDialog set={set} onClose={() => setDialog(false)} /> : null}
    </section>
  )
}
