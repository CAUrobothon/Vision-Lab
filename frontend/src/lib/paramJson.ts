// Parameter settings as JSON, for a chat assistant (ChatGPT etc.): a description to hand it, and a
// forgiving reader for the JSON it answers with.
import type { ModelInfo, Params } from '../api/types'

export interface ParsedParamJson {
  modelKey: string | null // a model the JSON names ("model" / "model_key"), else null = the current one
  params: Record<string, unknown>
}

/** Where JSON.parse stopped, in words: "3번째 줄 14번째 글자 근처" (V8 reports a line and column or
 *  a character position). */
function whereItBroke(message: string, text: string): string {
  const lc = message.match(/line (\d+) column (\d+)/)
  if (lc) return `${lc[1]}번째 줄 ${lc[2]}번째 글자 근처`
  const pos = message.match(/position (\d+)/)
  if (pos) {
    const before = text.slice(0, Number(pos[1]))
    const line = before.split('\n').length
    return `${line}번째 줄 ${before.length - before.lastIndexOf('\n')}번째 글자 근처`
  }
  if (/Unexpected end/i.test(message)) return '끝부분 (닫는 괄호나 따옴표가 빠졌을 수 있음)'
  return ''
}

/** JSON that assistants often write but JSON.parse refuses: // line comments and trailing commas. */
function loosen(s: string): string {
  return s.replace(/^\s*\/\/.*$/gm, '').replace(/,(\s*[}\]])/g, '$1')
}

/** Read `{"model": "...", "params": {...}}` or a bare `{"name": value, ...}`, also inside a
 *  ```json fence or surrounded by prose. Throws an Error with a Korean message. */
export function parseParamJson(text: string): ParsedParamJson {
  const t = text.trim()
  if (!t) throw new Error('JSON을 붙여넣으세요')
  const candidates = [t]
  const fence = t.match(/```[a-zA-Z]*\s*([\s\S]*?)```/)
  if (fence) candidates.push(fence[1])
  const a = t.indexOf('{')
  const b = t.lastIndexOf('}')
  if (a >= 0 && b > a) candidates.push(t.slice(a, b + 1))
  let obj: unknown = undefined
  let firstError: string | null = null
  let firstText = t
  outer: for (const c of candidates) {
    for (const s of [c, loosen(c)]) {
      try {
        obj = JSON.parse(s)
        break outer
      } catch (e) {
        if (firstError === null) {
          firstError = (e as Error).message
          firstText = s
        }
      }
    }
  }
  if (obj === undefined) {
    const where = whereItBroke(firstError ?? '', firstText)
    throw new Error(`JSON으로 읽을 수 없습니다${where ? `: ${where}` : ''}. 쉼표·따옴표·괄호를 확인하세요`)
  }
  if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) throw new Error('JSON 객체 { … } 여야 합니다')
  const o = obj as Record<string, unknown>
  const model = typeof o.model === 'string' ? o.model : typeof o.model_key === 'string' ? o.model_key : null
  const inner = o.params ?? o.parameters
  if (inner !== null && typeof inner === 'object' && !Array.isArray(inner)) {
    return { modelKey: model, params: inner as Record<string, unknown> }
  }
  const rest = { ...o }
  delete rest.model
  delete rest.model_key
  return { modelKey: model, params: rest }
}

/** What to paste into the assistant: the model, the answer format and every parameter with its
 *  current value, range and meaning. */
export function describeForChat(model: ModelInfo, params: Params, media?: { name: string; width: number; height: number } | null): string {
  const items = model.params.map((s) => {
    const item: Record<string, unknown> = { name: s.name, group: s.group, label: s.label, type: s.type }
    item.current = params[s.name] ?? s.default
    item.default = s.default
    if (s.min !== null) item.min = s.min
    if (s.max !== null) item.max = s.max
    if (s.step !== null) item.step = s.step
    if (s.unit) item.unit = s.unit
    if (s.options) item.options = s.options
    if (s.help) item.help = s.help
    return JSON.stringify(item)
  })
  return [
    `Vision Lab 파라미터 — 모델 "${model.name}" (키 ${model.key})`,
    media ? `지금 이미지: ${media.name}, ${media.width}×${media.height} px` : '',
    '',
    '모델 설명:',
    model.description,
    '',
    '답은 아래 형식의 JSON 하나로 주세요. 바꿀 파라미터만 넣으면 되고, 넣지 않은 파라미터는 지금 값을 유지합니다.',
    `{"model": "${model.key}", "params": {"<name>": <값>, ...}}`,
    '',
    '값 형식: int·float = 숫자 (min–max 안), bool = true/false, select = options의 value 하나,',
    'multiselect = value 배열 (빈 배열 = 전체), range = [하한, 상한], color = "#rrggbb",',
    'roi = [{"mode": "include" 또는 "exclude", "points": [[x, y], ...]}] (이미지 픽셀 좌표).',
    '',
    '파라미터 목록 (current = 지금 값, default = 기본값):',
    '[',
    items.join(',\n'),
    ']',
  ]
    .filter((line, i, all) => !(line === '' && all[i - 1] === ''))
    .join('\n')
}

/** Every value of `params` with the model key, one parameter per line, in the format
 *  parseParamJson reads back. */
export function currentAsJson(model: ModelInfo, params: Params): string {
  const lines = model.params.map((s) => `    ${JSON.stringify(s.name)}: ${JSON.stringify(params[s.name] ?? s.default)}`)
  return `{\n  "model": ${JSON.stringify(model.key)},\n  "params": {\n${lines.join(',\n')}\n  }\n}`
}
