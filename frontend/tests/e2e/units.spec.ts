// Unit-level checks of the pure helpers in src/lib (no browser page needed).
import { expect, test } from '@playwright/test'
import type { FrameRecord, ParamSpec } from '../../src/api/types'
import { outsideSpan, recordFor } from '../../src/lib/cubes'
import { formatParamValue, mergeParams, paramDecimals, paramOverrides, stepDecimals } from '../../src/lib/params'

function spec(p: Partial<ParamSpec>): ParamSpec {
  return { name: 'x', label: 'x', type: 'float', default: 0, min: 0, max: 1, step: null, options: null, group: 'g', help: '', unit: '', ...p }
}

test('decimals follow the step, not its order of magnitude', () => {
  expect(stepDecimals(0.125)).toBe(3)
  expect(stepDecimals(0.05)).toBe(2)
  expect(stepDecimals(0.1)).toBe(1)
  expect(stepDecimals(2.5)).toBe(1)
  expect(stepDecimals(5)).toBe(0)
  const s = spec({ name: 'cluster_scale', step: 0.125, min: 0.125, max: 1, default: 0.25 })
  expect(paramDecimals(s)).toBe(3)
  expect(formatParamValue(s, 0.375)).toBe('0.375')
  expect(formatParamValue(spec({ step: 0.001 }), 0.0125)).toBe('0.013')
  expect(formatParamValue(spec({ type: 'int', step: 1 }), 7)).toBe('7')
  // No step: 1/100 of the range.
  expect(paramDecimals(spec({ min: 0, max: 1 }))).toBe(2)
  expect(paramDecimals(spec({ min: 0, max: 255 }))).toBe(0)
})

test('only values that differ from the defaults are remembered', () => {
  const specs = [spec({ name: 'a', default: 0.5 }), spec({ name: 'b', type: 'int', default: 3 })]
  expect(paramOverrides(specs, { a: 0.5, b: 4, gone: 1 })).toEqual({ b: 4 })
  expect(paramOverrides(specs, { a: 0.5, b: 3 })).toEqual({})
  // A changed default reaches a param the user never touched.
  const later = [spec({ name: 'a', default: 0.7 }), spec({ name: 'b', type: 'int', default: 3 })]
  expect(mergeParams(later, { b: 4 })).toEqual({ a: 0.7, b: 4 })
})

test('a run result describes only the frames it covers', () => {
  const frames: FrameRecord[] = [10, 12, 14, 16, 18].map((frame) => ({ frame, ms: 1, cubes: [] }))
  const span = { start: 10, end: 20, stride: 2 }
  expect(recordFor(frames, 13, span)?.frame).toBe(12)
  expect(recordFor(frames, 19, span)?.frame).toBe(18)
  expect(recordFor(frames, 20, span)).toBeNull() // past frame_end
  expect(recordFor(frames, 149, span)).toBeNull()
  expect(recordFor(frames, 5, span)).toBeNull() // before frame_start
  expect(outsideSpan(span, 20)).toBe(true)
  expect(outsideSpan(span, 10)).toBe(false)
  // Open-ended range: only within the stride of the last stored frame (a missing frame or a
  // running run that has not got there yet).
  const open = { start: 10, end: null, stride: 2 }
  expect(recordFor(frames, 19, open)?.frame).toBe(18)
  expect(recordFor(frames, 30, open)).toBeNull()
  // No span (images): the latest record at or before the frame.
  expect(recordFor(frames, 149)?.frame).toBe(18)
})

test('param JSON from a chat assistant is read in every usual wrapping', async () => {
  const { parseParamJson } = await import('../../src/lib/paramJson')
  expect(parseParamJson('{"model": "cv.hsv_ranges", "params": {"red_s": [80, 255]}}')).toEqual({
    modelKey: 'cv.hsv_ranges',
    params: { red_s: [80, 255] },
  })
  // A bare object of params, with a model key mixed in.
  expect(parseParamJson('{"model_key": "cv.edge_quads", "canny": [20, 60]}')).toEqual({
    modelKey: 'cv.edge_quads',
    params: { canny: [20, 60] },
  })
  expect(parseParamJson('{"blur_ksize": 3}')).toEqual({ modelKey: null, params: { blur_ksize: 3 } })
  // Prose around a ```json fence, a // comment and trailing commas.
  const chat = '이렇게 바꿔 보세요.\n```json\n{\n  "model": "cv.board_diff_lab",\n  "params": {\n    // 나무가 판으로 샘\n    "t_chroma_lo": 8,\n  },\n}\n```\n그다음 다시 확인하세요.'
  expect(parseParamJson(chat)).toEqual({ modelKey: 'cv.board_diff_lab', params: { t_chroma_lo: 8 } })
  expect(() => parseParamJson('')).toThrow('붙여넣으세요')
  expect(() => parseParamJson('[1, 2]')).toThrow('객체')
  expect(() => parseParamJson('red_s = 80')).toThrow('읽을 수 없습니다')
  // Where it broke, in Korean, not V8's English message.
  expect(() => parseParamJson('{\n  "a": 1,\n  "b" 2\n}')).toThrow(/3번째 줄 \d+번째 글자 근처/)
  expect(() => parseParamJson('{"a": [1, 2')).toThrow(/끝부분|번째 줄/)
})

test('the current-values JSON is one parameter per line and reads back unchanged', async () => {
  const { currentAsJson, parseParamJson } = await import('../../src/lib/paramJson')
  const model = {
    key: 'cv.x',
    params: [spec({ name: 'a', type: 'range', default: [1, 2] }), spec({ name: 'b', type: 'bool', default: true }), spec({ name: 'c', type: 'color', default: '#ffffff' })],
  } as unknown as import('../../src/api/types').ModelInfo
  const text = currentAsJson(model, { a: [-12, 10], c: '#c8323c' })
  expect(text).toBe('{\n  "model": "cv.x",\n  "params": {\n    "a": [-12,10],\n    "b": true,\n    "c": "#c8323c"\n  }\n}')
  expect(parseParamJson(text)).toEqual({ modelKey: 'cv.x', params: { a: [-12, 10], b: true, c: '#c8323c' } })
})

test('particles follow the last sound of a name', async () => {
  const { named } = await import('../../src/lib/josa')
  expect(named('DS2', '을/를')).toBe('"DS2"를')
  expect(named('옵션A', '과/와')).toBe('"옵션A"와')
  expect(named('HSV 색 범위', '으로/로')).toBe('"HSV 색 범위"로')
  expect(named('판 차이 + Lab 최근접색', '으로/로')).toBe('"판 차이 + Lab 최근접색"으로')
  expect(named('검출률우선', '을/를')).toBe('"검출률우선"을')
  expect(named('모델1', '이/가')).toBe('"모델1"이')
  expect(named('scene.jpg', '은/는')).toBe('"scene.jpg"는')
})
