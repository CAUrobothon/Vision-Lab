// Presets for the current model: select to apply, save the current values as a new preset, or
// (after they moved away from the shown preset) restore it or overwrite it, delete.
//
// The select always shows where the current values stand: the preset they equal, else a separate
// "A (수정됨)" entry. A native select fires no change when the option already shown is picked
// again, so A's own option must not stay selected once the values differ from it: picking A then
// always restores A. Overwriting is labelled and takes a second click.
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { FilePlus2, Undo2 } from 'lucide-react'
import { api } from '../api/client'
import { keys, queryClient, usePresets } from '../api/queries'
import type { ParamSpec, Params } from '../api/types'
import { sameParams } from '../lib/params'
import { loadJson, saveJson } from '../lib/storage'
import { ConfirmButton } from '../ui/controls'
import { useToast } from '../ui/toast'
import { named } from '../lib/josa'

export function PresetBar({
  modelKey,
  specs,
  params,
  onApply,
}: {
  modelKey: string
  specs: ParamSpec[]
  params: Params
  onApply: (p: Params) => void
}) {
  const toast = useToast()
  const presets = usePresets(modelKey)
  // The preset last picked for this model is remembered, so "A (수정됨)" with its 되돌리기 and
  // 덮어쓰기 survives a model switch or a reload (found by exploratory QA).
  const [selected, setSelectedRaw] = useState<number | null>(() => loadJson<number | null>(`preset.${modelKey}`, null))
  const setSelected = (id: number | null) => {
    setSelectedRaw(id)
    saveJson(`preset.${modelKey}`, id)
  }
  const [naming, setNaming] = useState(false)
  const [name, setName] = useState('')
  const list = presets.data ?? []
  const picked = list.find((p) => p.id === selected) ?? null
  // Two presets can hold the same values: the one picked wins over the first equal one.
  const match = picked && sameParams(specs, picked.params, params) ? picked : (list.find((p) => sameParams(specs, p.params, params)) ?? null)
  const current = match ?? picked
  const modified = current !== null && match === null
  // The server overwrites a preset saved under an existing name.
  const overwrites = list.some((p) => p.name === name.trim())

  const create = useMutation({
    mutationFn: () => api.createPreset({ model_key: modelKey, name: name.trim(), params }),
    onSuccess: (p) => {
      setSelected(p.id)
      setNaming(false)
      setName('')
      toast.ok(`프리셋 ${named(p.name, '을/를')} 저장했습니다`)
      void queryClient.invalidateQueries({ queryKey: keys.presets(modelKey) })
    },
    onError: (e: Error) => toast.error(e.message),
  })
  const update = useMutation({
    mutationFn: () => api.createPreset({ model_key: modelKey, name: current!.name, params }),
    onSuccess: (p) => {
      setSelected(p.id)
      toast.ok(`프리셋 ${named(p.name, '을/를')} 지금 값으로 덮어썼습니다`)
      void queryClient.invalidateQueries({ queryKey: keys.presets(modelKey) })
    },
    onError: (e: Error) => toast.error(e.message),
  })
  const remove = useMutation({
    mutationFn: (id: number) => api.deletePreset(id),
    onSuccess: () => {
      setSelected(null)
      void queryClient.invalidateQueries({ queryKey: keys.presets(modelKey) })
    },
    onError: (e: Error) => toast.error(e.message),
  })

  if (naming)
    return (
      <form
        className="preset-bar"
        onSubmit={(e) => {
          e.preventDefault()
          if (name.trim()) create.mutate()
        }}
      >
        <div className="preset-line">
          <input
            className="input"
            autoFocus
            value={name}
            maxLength={80}
            placeholder="새 프리셋 이름"
            aria-label="새 프리셋 이름"
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === 'Escape' && setNaming(false)}
          />
          <button type="submit" className={`btn btn-sm${overwrites ? ' btn-danger' : ''}`} disabled={!name.trim() || create.isPending}>
            {overwrites ? '덮어쓰기' : '저장'}
          </button>
          <button type="button" className="btn btn-sm btn-quiet" onClick={() => setNaming(false)}>
            취소
          </button>
        </div>
        {overwrites ? <p className="preset-hint">같은 이름의 프리셋이 있어 그 값을 지금 값으로 덮어씁니다.</p> : null}
      </form>
    )

  return (
    <div className="preset-bar">
      <div className="preset-line">
        <select
          className="select"
          aria-label="프리셋"
          value={modified ? 'modified' : (current?.id ?? '')}
          title={current ? (modified ? `"${current.name}"에서 바뀐 값이 있습니다` : `지금 값 = 프리셋 "${current.name}"`) : undefined}
          onChange={(e) => {
            const id = Number(e.target.value)
            const p = list.find((x) => x.id === id)
            setSelected(p ? p.id : null)
            if (p) onApply(p.params)
          }}
        >
          {modified ? (
            <option value="modified" disabled>
              {current!.name} (수정됨)
            </option>
          ) : (
            <option value="">{list.length ? (current ? '프리셋 선택' : '프리셋과 다름') : '프리셋 없음'}</option>
          )}
          {list.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </div>
      <div className="preset-line preset-actions">
        <button type="button" className="btn btn-sm btn-quiet" title="지금 값을 새 프리셋으로 저장" onClick={() => setNaming(true)}>
          <FilePlus2 size={13} strokeWidth={1.75} /> 새 프리셋으로 저장
        </button>
        {modified ? (
          <>
            <button type="button" className="btn btn-sm btn-quiet" title={`"${current!.name}"의 값으로 되돌리기`} onClick={() => onApply(current!.params)}>
              <Undo2 size={13} strokeWidth={1.75} /> 되돌리기
            </button>
            <ConfirmButton
              label={`"${current!.name}"에 덮어쓰기`}
              confirmLabel="정말 덮어쓰기"
              icon={false}
              disabled={update.isPending}
              onConfirm={() => update.mutate()}
            />
          </>
        ) : null}
        {current ? <ConfirmButton label={`프리셋 ${current.name} 삭제`} onConfirm={() => remove.mutate(current.id)} quiet /> : null}
      </div>
    </div>
  )
}
