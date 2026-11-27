// "JSON" next to the presets: the selected model's parameters as JSON (current values, or the
// defaults) to hand to a chat assistant, plus a fuller description; paste the JSON it answers with
// and apply it.
//
// The box is checked as you type: the server resolves the JSON exactly as a run would
// (POST /api/params/check), and the dialog shows what applying would change ("지금 → 새 값").
// "적용" is only enabled when something would change; once applied it reads "적용됨" and stays
// disabled until the box or the current values differ again, so the state is never ambiguous.
import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Braces, Check, Copy } from 'lucide-react'
import { api, ApiError } from '../api/client'
import { usePresets } from '../api/queries'
import type { Media, ModelInfo, ParamCheck, Params } from '../api/types'
import { currentAsJson, describeForChat, parseParamJson, type ParsedParamJson } from '../lib/paramJson'
import { defaultParams, diffParams, formatParamValue, sameParams } from '../lib/params'
import { useSettled } from '../lib/settled'
import { closeDialog, Modal } from '../ui/controls'
import { useToast } from '../ui/toast'
import { josa, named } from '../lib/josa'

const SHOWN_CHANGES = 12

const short = (v: unknown) => {
  const s = JSON.stringify(v)
  return s.length > 48 ? s.slice(0, 45) + '…' : s
}

type Parsed = { ok: true; value: ParsedParamJson } | { ok: false; error: string }

function parse(text: string): Parsed {
  try {
    return { ok: true, value: parseParamJson(text) }
  } catch (e) {
    return { ok: false, error: (e as Error).message }
  }
}

export function ParamJsonButton({
  model,
  models,
  params,
  paramsFor,
  media,
  onApply,
}: {
  model: ModelInfo
  models: ModelInfo[]
  params: Params
  paramsFor: (m: ModelInfo) => Params // current values of any model (the JSON may name another one)
  media: Media | null
  onApply: (target: ModelInfo, next: Params) => void
}) {
  const toast = useToast()
  const [open, setOpen] = useState(false)
  const [text, setText] = useState('')
  const [keepRest, setKeepRest] = useState(true)
  const presets = usePresets(model.key)
  // Every preset holding exactly the current values (two presets can be equal).
  const matching = (presets.data ?? []).filter((p) => sameParams(model.params, p.params, params))

  // The box starts with the selected model's current values (what an assistant asks for).
  // `synced` is the last text the lab put there or that was applied: while the box still holds
  // it, opening the dialog refreshes it with the current values (a preset or a slider may have
  // changed them). A draft that was typed or pasted and not applied is left alone.
  const latest = useRef({ model, params })
  useEffect(() => {
    latest.current = { model, params }
  })
  const synced = useRef('')
  const appliedText = useRef<string | null>(null)
  const fill = (kind: 'current' | 'default') => {
    const { model: m, params: p } = latest.current
    const v = currentAsJson(m, kind === 'current' ? p : defaultParams(m.params))
    synced.current = v
    setText(v)
  }

  // When the model changes in the picker, a JSON written for the old one would switch it back,
  // so the box gets the new model's values. A switch made by applying JSON keeps the box.
  const textModel = useRef(model.key)
  useEffect(() => {
    if (model.key === textModel.current) return
    textModel.current = model.key
    fill('current')
  }, [model.key]) // eslint-disable-line react-hooks/exhaustive-deps

  // ----- live check of the box -----
  const settled = useSettled(text, 250)
  const parsed = useMemo(() => parse(settled), [settled])
  const targetKey = parsed.ok ? (parsed.value.modelKey ?? model.key) : null
  const target = targetKey ? (models.find((m) => m.key === targetKey) ?? null) : null
  const check = useQuery({
    queryKey: ['paramCheck', targetKey, parsed.ok ? JSON.stringify(parsed.value.params) : ''],
    queryFn: () => api.checkParams({ model_key: targetKey!, params: (parsed as { value: ParsedParamJson }).value.params as Params }),
    enabled: open && parsed.ok && !!target && target.state === 'ready',
    staleTime: Infinity,
    gcTime: 60_000,
    retry: false,
  })
  const res: ParamCheck | undefined = check.data
  const current = target ? (target.key === model.key ? params : paramsFor(target)) : params
  const next = useMemo(() => {
    if (!target || !res) return null
    const out: Params = { ...(keepRest ? current : defaultParams(target.params)) }
    for (const n of res.applied) out[n] = res.params[n]
    return out
  }, [target, res, keepRest, current])
  const changes = useMemo(() => (target && next ? diffParams(target.params, current, next) : []), [target, next, current])
  const typing = text !== settled
  const switching = !!target && target.key !== model.key

  type State =
    | { kind: 'empty' | 'checking' }
    | { kind: 'error'; message: string }
    | { kind: 'ready'; count: number }
    | { kind: 'same' }
  let state: State
  if (!text.trim()) state = { kind: 'empty' }
  else if (typing) state = { kind: 'checking' }
  else if (!parsed.ok) state = { kind: 'error', message: parsed.error }
  else if (!target)
    state = { kind: 'error', message: `모델 ${named(targetKey ?? '', '을/를')} 찾을 수 없습니다. 쓸 수 있는 키: ${models.map((m) => m.key).join(', ')}` }
  else if (target.state !== 'ready') state = { kind: 'error', message: `모델 ${named(target.name, '은/는')} 지금 쓸 수 없습니다: ${target.error ?? ''}` }
  else if (check.isError) state = { kind: 'error', message: (check.error as ApiError).message }
  else if (!res || !next) state = { kind: 'checking' }
  else if (changes.length === 0 && !switching) state = { kind: 'same' }
  else state = { kind: 'ready', count: changes.length }

  const apply = () => {
    if (state.kind !== 'ready' || !target || !next) return
    textModel.current = target.key
    synced.current = text
    appliedText.current = text
    onApply(target, next)
    toast.ok(`${switching ? `${named(target.name, '으로/로')} 바꾸고 ` : ''}${changes.length}개 값을 적용했습니다`)
  }
  // "적용" or ⌘Enter while the box is still being checked: apply as soon as the check is done
  // (found by exploratory QA: ⌘Enter right after pasting did nothing, silently).
  const [queued, setQueued] = useState(false)
  const requestApply = () => {
    if (state.kind === 'ready') apply()
    else if (state.kind === 'checking') setQueued(true)
  }
  useEffect(() => {
    if (!queued || state.kind === 'checking') return
    setQueued(false)
    if (state.kind === 'ready') apply()
  })
  // "적용됨" only for text that was really applied; a box that merely equals the current values
  // (on opening, or JSON without any of this model's names) says so instead.
  const wasApplied = state.kind === 'same' && appliedText.current === text
  const noNames = state.kind === 'same' && !!res && res.applied.length === 0

  const copy = async (what: string, value: string) => {
    try {
      await navigator.clipboard.writeText(value)
      toast.ok(`${what}을 복사했습니다`)
    } catch {
      toast.error('클립보드에 복사하지 못했습니다')
    }
  }

  const applyLabel =
    state.kind === 'same'
      ? wasApplied
        ? '적용됨'
        : '바뀔 값 없음'
      : state.kind === 'ready'
        ? `${switching ? '모델 바꾸고 ' : ''}${state.count}개 적용`
        : state.kind === 'checking'
          ? queued
            ? '확인 후 적용…'
            : '확인 중…'
          : '적용'

  return (
    <>
      <button
        type="button"
        className="btn btn-sm"
        title="파라미터를 JSON으로 보기 · 복사 · 붙여넣어 적용"
        onClick={() => {
          if (text === '' || text === synced.current) fill('current')
          setKeepRest(true) // an unticked box from last time is not silently reused
          setOpen(true)
        }}
      >
        <Braces size={13} strokeWidth={1.75} /> JSON
      </button>
      {open ? (
        <Modal
          onClose={() => setOpen(false)}
          title={
            <>
              파라미터 JSON <span className="pj-model">· {model.name}</span>
              {matching.length > 0 ? (
                <span className="pj-preset" title={matching.length > 1 ? '이 프리셋들은 값이 모두 같습니다' : undefined}>
                  프리셋 {matching.map((p) => `"${p.name}"`).join(', ')}
                  {josa(matching[matching.length - 1].name, '과/와')} 같음
                </span>
              ) : null}
            </>
          }
        >
          <ol className="pj-steps">
            <li>
              <span>아래 JSON(이 모델의 지금 값)을 복사해 ChatGPT에 줍니다. 범위·뜻까지 주려면 설명을 복사하세요.</span>
              <span className="pj-copy">
                <button type="button" className="btn btn-sm" disabled={!text.trim()} onClick={() => void copy('JSON', text)}>
                  <Copy size={13} strokeWidth={1.75} /> JSON 복사
                </button>
                <button type="button" className="btn btn-sm btn-quiet" onClick={() => void copy('ChatGPT용 설명', describeForChat(model, params, media))}>
                  ChatGPT용 설명 복사 (범위·뜻 포함)
                </button>
              </span>
            </li>
            <li>
              <span>ChatGPT가 준 JSON으로 아래 내용을 바꿔 붙여넣으면, 무엇이 바뀌는지 바로 아래에 보입니다. 확인하고 적용하세요.</span>
            </li>
          </ol>

          <div className="pj-fill">
            <span className="faint">채우기</span>
            <button type="button" className="btn btn-sm btn-quiet" onClick={() => fill('current')}>
              지금 값으로
            </button>
            <button type="button" className="btn btn-sm btn-quiet" onClick={() => fill('default')}>
              기본값으로
            </button>
          </div>
          <textarea
            className="input pj-text"
            aria-label="파라미터 JSON"
            aria-describedby="pj-state"
            spellCheck={false}
            value={text}
            placeholder={`{"model": "${model.key}", "params": {"이름": 값, ...}}`}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                e.preventDefault()
                requestApply()
              }
            }}
          />
          <label className="pj-keep">
            <input type="checkbox" checked={keepRest} onChange={(e) => setKeepRest(e.target.checked)} />
            JSON에 없는 파라미터는 지금 값 유지 (끄면 기본값으로)
          </label>

          <div id="pj-state" className={`pj-state ${state.kind}`} role="status" aria-live="polite">
            {state.kind === 'empty' ? <p>붙여넣을 JSON이 없습니다. "지금 값으로"를 누르면 이 모델의 지금 값이 채워집니다.</p> : null}
            {state.kind === 'checking' && !queued ? <p className="faint">확인 중…</p> : null}
            {state.kind === 'error' ? <p className="error-text">{state.message}</p> : null}
            {state.kind === 'same' && wasApplied ? (
              <p className="pj-same">
                <Check size={14} strokeWidth={2} /> 적용했습니다 — 지금 값과 같습니다
              </p>
            ) : null}
            {state.kind === 'same' && !wasApplied ? (
              <p>
                {noNames
                  ? `이 JSON에는 "${model.name}"의 파라미터 이름이 없습니다. 바꿀 값이 없습니다.`
                  : '지금 값과 같습니다 — 바뀔 값이 없습니다.'}
              </p>
            ) : null}
            {state.kind === 'checking' && queued ? <p className="faint">확인이 끝나면 바로 적용합니다…</p> : null}
            {state.kind === 'ready' && target ? (
              <>
                <p>
                  <strong>
                    {switching ? `모델이 ${named(target.name, '으로/로')} 바뀌고, ` : ''}적용하면 값 {state.count}개가 바뀝니다
                  </strong>
                  {state.count === 0 ? ' (값은 그대로)' : ''}
                </p>
                {changes.length > 0 ? (
                  <ul className="pj-list pj-changes">
                    {changes.slice(0, SHOWN_CHANGES).map((c) => (
                      <li key={c.spec.name}>
                        <code>{c.spec.name}</code> <span className="faint">{c.spec.label}</span>{' '}
                        <span className="num">
                          {formatParamValue(c.spec, c.from)} → <strong>{formatParamValue(c.spec, c.to)}</strong>
                        </span>
                      </li>
                    ))}
                    {changes.length > SHOWN_CHANGES ? <li className="faint">외 {changes.length - SHOWN_CHANGES}개</li> : null}
                  </ul>
                ) : null}
              </>
            ) : null}
            {res && (state.kind === 'ready' || state.kind === 'same') && res.adjusted.length > 0 ? (
              <>
                <p className="pj-sub">범위·형식에 맞춰 바뀌는 값 {res.adjusted.length}개</p>
                <ul className="pj-list">
                  {res.adjusted.map((a) => (
                    <li key={a.name}>
                      <code>{a.name}</code> {short(a.given)} → {short(a.used)}
                    </li>
                  ))}
                </ul>
              </>
            ) : null}
            {res && (state.kind === 'ready' || state.kind === 'same') && res.ignored.length > 0 ? (
              <>
                <p className="pj-sub">이 모델에 없는 이름이라 무시 {res.ignored.length}개</p>
                <p className="pj-names">
                  {res.ignored.map((n) => (
                    <code key={n}>{n}</code>
                  ))}
                </p>
              </>
            ) : null}
          </div>

          <footer className="pj-foot">
            <span className="faint">⌘Enter 적용 · 적용하면 미리보기가 바로 다시 그려집니다</span>
            <button type="button" className="btn btn-quiet" onClick={closeDialog}>
              닫기
            </button>
            <button
              type="button"
              className="btn btn-primary pj-apply"
              disabled={!(state.kind === 'ready' || (state.kind === 'checking' && !queued))}
              onClick={requestApply}
            >
              {wasApplied ? <Check size={14} strokeWidth={2} /> : null}
              {applyLabel}
            </button>
          </footer>
        </Modal>
      ) : null}
    </>
  )
}
