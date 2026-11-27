// Regression tests for the bugs exploratory QA found on 2026-11-25 (one test per finding).
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { basename } from 'node:path'
import { mediaFile, uploadViaRail } from './helpers'

async function upload(request: APIRequestContext, file: string): Promise<number> {
  const res = await request.post('/api/media', {
    multipart: { file: { name: basename(file), mimeType: 'image/jpeg', buffer: readFileSync(file) } },
  })
  expect(res.status()).toBe(200)
  return (await res.json()).media.id
}

async function finishedRun(request: APIRequestContext, mediaId: number): Promise<number> {
  const run = await (await request.post('/api/runs', { data: { media_id: mediaId, model_key: 'cv.hsv_ranges' } })).json()
  await expect.poll(async () => (await (await request.get(`/api/runs/${run.id}`)).json()).status, { timeout: 60_000 }).toBe('done')
  return run.id
}

async function chooseModel(page: Page, name: RegExp) {
  await page.locator('.model-current').click()
  await page.getByRole('option', { name }).click()
}

test('runs page: a bulk delete only counts the rows the filter shows', async ({ page, request }) => {
  const a = await upload(request, mediaFile('datasets/ds_a.jpg'))
  const b = await upload(request, mediaFile('datasets/ds_b.jpg'))
  const ra = await finishedRun(request, a)
  const rb = await finishedRun(request, b)
  await page.goto('/runs')
  await page.getByLabel(`기록 #${ra} 선택`).check()
  await page.getByLabel(`기록 #${rb} 선택`).check()
  await expect(page.locator('.bulk-bar')).toContainText('2개 선택')
  await page.getByLabel('미디어', { exact: true }).selectOption(String(a))
  await expect(page.getByLabel(`기록 #${rb} 선택`)).toHaveCount(0) // filtered out, still ticked in state
  await expect(page.locator('.bulk-bar')).toContainText('1개 선택')
  await expect(page.getByRole('button', { name: '1개 삭제' })).toBeVisible()
})

test('workbench: the re-run link is dropped when the media or the model changes', async ({ page, request }) => {
  const a = await upload(request, mediaFile('datasets/ds_a.jpg'))
  await upload(request, mediaFile('datasets/ds_b.jpg'))
  const ra = await finishedRun(request, a)
  const note = page.locator('.parent-note')

  await page.goto(`/?media=${a}&rerun=${ra}`)
  await expect(note).toContainText(`#${ra}`)
  await page.locator('.media-row', { hasText: 'ds_b.jpg' }).locator('.media-main').click()
  await expect(note).toHaveCount(0)

  await page.goto(`/?media=${a}&rerun=${ra}`)
  await expect(note).toContainText(`#${ra}`)
  await chooseModel(page, /윤곽으로 닫힌 사각형/)
  await expect(note).toHaveCount(0)
})

test('workbench: a deleted run is left for the live preview, not drawn from stale data', async ({ page, request }) => {
  const a = await upload(request, mediaFile('datasets/ds_a.jpg'))
  const ra = await finishedRun(request, a)
  await page.goto(`/?media=${a}&run=${ra}`)
  await expect(page.locator('.mode-badge.is-run')).toContainText(`기록 #${ra}`)
  expect((await request.delete(`/api/runs/${ra}`)).status()).toBe(200)
  await page.reload()
  await expect(page.locator('.mode-badge.is-preview')).toBeVisible()
  await expect.poll(() => new URL(page.url()).searchParams.get('run')).toBeNull()
})

test('params: an emptied number field goes back to its value', async ({ page }) => {
  await page.goto('/')
  await uploadViaRail(page, mediaFile('scene_a.jpg'))
  await chooseModel(page, /HSV 색 범위/)
  const low = page.locator('[data-param=red_s] .num-input').first()
  await expect(low).toHaveValue('90')
  await low.fill('')
  await low.press('Enter')
  await expect(low).toHaveValue('90')
})

test('viewer: keys pressed in the JSON dialog do not step the video behind it', async ({ page }) => {
  await page.goto('/')
  await uploadViaRail(page, mediaFile('frames.mp4'))
  const shown = page.locator('.transport-frame')
  await expect(shown).toContainText('f 0')
  await page.getByRole('button', { name: 'JSON' }).click()
  await page.getByRole('dialog', { name: /파라미터 JSON/ }).getByRole('button', { name: '닫기' }).last().focus()
  for (let i = 0; i < 3; i++) await page.keyboard.press('ArrowRight')
  await page.keyboard.press('Escape')
  await expect(shown).toContainText('f 0')

  // Frame fields can be cleared and retyped; an end before the start is refused with a reason.
  const strideField = page.getByLabel('프레임 간격')
  await strideField.fill('')
  await strideField.type('5')
  await strideField.press('Enter')
  await expect(strideField).toHaveValue('5')
  await page.getByRole('button', { name: '구간', exact: true }).click()
  await page.getByLabel('시작 프레임 (포함)').fill('20')
  await page.getByLabel('시작 프레임 (포함)').press('Enter')
  await page.getByLabel('끝 프레임 (제외)').fill('10')
  await page.getByLabel('끝 프레임 (제외)').press('Enter')
  await expect(page.locator('.vr-summary')).toContainText('시작(20)보다 커야')
  await expect(page.getByRole('button', { name: '구간 실행' })).toBeDisabled()
})

test('JSON dialog: unchanged vs applied, numbers for options, ⌘Enter right after pasting', async ({ page }) => {
  await page.goto('/')
  await uploadViaRail(page, mediaFile('scene_b.jpg'))
  await chooseModel(page, /윤곽으로 닫힌 사각형/)
  await page.getByRole('button', { name: 'JSON' }).click()
  const d = page.getByRole('dialog', { name: /파라미터 JSON/ })
  const box = d.getByLabel('파라미터 JSON')
  const state = d.locator('#pj-state')
  const btn = d.locator('.pj-apply')
  // Opening is not "applied": nothing was.
  await expect(state).toContainText('지금 값과 같습니다 — 바뀔 값이 없습니다')
  await expect(btn).toHaveText('바뀔 값 없음')
  await expect(btn).toBeDisabled()
  await box.fill('{"params": {}}')
  await expect(state).toContainText('파라미터 이름이 없습니다')

  // A number for a select param is that option, not a reset to the default.
  await box.fill('{"params": {"canny_aperture": 5, "region_connectivity": 8}}')
  await expect(state).toContainText('적용하면 값 2개가 바뀝니다')
  await expect(state).toContainText('3 → 5')

  // ⌘Enter straight after pasting applies once the check is done.
  await box.fill('{"params": {"canny_aperture": 7}}')
  await box.press('Meta+Enter')
  await expect(state).toContainText('적용했습니다 — 지금 값과 같습니다')
  await expect(btn).toHaveText('적용됨')
  await expect(btn).toBeDisabled()
})

test('presets: "(수정됨)" with 되돌리기 survives a reload', async ({ page }) => {
  await page.goto('/')
  await uploadViaRail(page, mediaFile('scene_b.jpg'))
  await chooseModel(page, /Lab 자동 군집/)
  const select = page.getByLabel('프리셋', { exact: true })
  await page.getByRole('button', { name: '새 프리셋으로 저장' }).click()
  await page.getByLabel('새 프리셋 이름').fill('새로고침 시험')
  await page.getByRole('button', { name: '저장', exact: true }).click()
  await expect(select.locator('option:checked')).toHaveText('새로고침 시험')
  const k = page.locator('[data-param=k_clusters] .num-input')
  await k.fill('8')
  await k.press('Enter')
  await expect(select.locator('option:checked')).toHaveText('새로고침 시험 (수정됨)')
  await page.reload()
  await expect(select.locator('option:checked')).toHaveText('새로고침 시험 (수정됨)')
  await page.getByRole('button', { name: '되돌리기' }).click()
  await expect(k).toHaveValue('6')
})
