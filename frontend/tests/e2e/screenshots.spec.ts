// Refreshes docs/screenshots/ (1440x900) from the lab's synthetic samples. Skipped unless
// SCREENSHOTS=1:  SCREENSHOTS=1 npx playwright test screenshots
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'
import { mkdirSync } from 'node:fs'
import { resolve } from 'node:path'
import type { Media, Run } from '../../src/api/types'

const OUT = resolve(import.meta.dirname, '../../../docs/screenshots')

test.skip(!process.env.SCREENSHOTS, 'SCREENSHOTS=1 refreshes docs/screenshots/')
test.describe.configure({ mode: 'serial' })

let media: Record<string, Media>
let runs: Record<string, Run>

async function runAndWait(request: APIRequestContext, body: object, patch?: object): Promise<Run> {
  let run = (await (await request.post('/api/runs', { data: body })).json()) as Run
  await expect.poll(async () => (run = await (await request.get(`/api/runs/${run.id}`)).json()).status, { timeout: 90_000 }).toBe('done')
  if (patch) run = await (await request.patch(`/api/runs/${run.id}`, { data: patch })).json()
  return run
}

async function shot(page: Page, name: string) {
  await page.waitForLoadState('networkidle')
  await page.waitForTimeout(400)
  await page.screenshot({ path: `${OUT}/${name}.png` })
}

/** Screen position of an image point in the first viewer. */
async function screenPoint(page: Page, x: number, y: number): Promise<[number, number]> {
  const box = (await page.locator('.viewer-stage').first().boundingBox())!
  const t = await page.locator('.viewer-content').first().evaluate((e) => (e as HTMLElement).style.transform)
  const [, tx, ty, s] = /translate\(([-\d.]+)px, ([-\d.]+)px\) scale\(([\d.]+)\)/.exec(t)!.map(Number)
  return [box.x + tx + x * s, box.y + ty + y * s]
}

test.beforeAll(async ({ request }) => {
  mkdirSync(OUT, { recursive: true })
  const list = (await (await request.post('/api/samples')).json()) as Media[]
  media = Object.fromEntries(list.map((m) => [m.name, m]))
  const a = media['synth_mission_a.jpg']
  const b = media['synth_mission_b.jpg']
  const v = media['synth_moves.mp4']
  runs = {
    hsvA: await runAndWait(request, { media_id: a.id, model_key: 'cv.hsv_ranges', params: {} }, { note: '기본값 기준', tags: ['기준'] }),
    boardA: await runAndWait(
      request,
      { media_id: a.id, model_key: 'cv.board_diff_lab', params: {} },
      { note: '그림자에 강함', tags: ['후보'], starred: true },
    ),
    hsvB: await runAndWait(request, { media_id: b.id, model_key: 'cv.hsv_ranges', params: { red_s: [120, 255] } }, { note: '바구니 바닥 제외 확인' }),
    video: await runAndWait(request, { media_id: v.id, model_key: 'cv.hsv_ranges', params: {}, stride: 2 }),
  }
})

test('workbench: empty library', async ({ page }) => {
  await page.route('**/api/media', (route) => (route.request().method() === 'GET' ? route.fulfill({ json: [] }) : route.fallback()))
  await page.goto('/')
  await expect(page.getByText('이미지나 영상을 여기에 끌어다 놓으세요')).toBeVisible()
  await shot(page, '01-workbench-empty')
})

test('workbench: image preview with a highlighted cube and the pixel inspector', async ({ page }) => {
  const a = media['synth_mission_a.jpg']
  await page.goto(`/?media=${a.id}`)
  await expect(page.locator('.result-table tbody tr').first()).toBeVisible()
  await page.getByRole('button', { name: '픽셀 정보' }).click()
  const cell = await page.locator('.result-table tbody tr').nth(2).locator('td').nth(2).textContent()
  const [cx, cy] = cell!.split('·').map(parseFloat)
  const [sx, sy] = await screenPoint(page, cx, cy)
  await page.mouse.move(sx, sy)
  await shot(page, '02-workbench-image')
})

test('workbench: ROI drawn over the baskets and a debug layer', async ({ page }) => {
  const b = media['synth_mission_b.jpg']
  await page.goto(`/?media=${b.id}`)
  await expect(page.locator('.result-table tbody tr').first()).toBeVisible()
  await page.locator('[data-param=roi]').scrollIntoViewIfNeeded()
  await page.getByRole('button', { name: '제외 영역 그리기' }).click()
  for (const [x, y] of [
    [690, 150],
    [1180, 150],
    [1250, 640],
    [690, 640],
  ]) {
    const [sx, sy] = await screenPoint(page, x, y)
    await page.mouse.click(sx, sy)
  }
  await page.keyboard.press('Enter')
  await page.getByRole('button', { name: '오버레이 레이어' }).click()
  await page.getByLabel('디버그 레이어', { exact: true }).check()
  await page.locator('.layer-debug select').waitFor()
  await page.getByRole('button', { name: '밝은 부분만' }).click()
  await page.getByRole('button', { name: '오버레이 레이어' }).click()
  await shot(page, '03-workbench-roi-debug')
})

test('workbench: video run on a frame between processed frames', async ({ page }) => {
  const v = media['synth_moves.mp4']
  await page.goto(`/?media=${v.id}&run=${runs.video.id}`)
  await expect(page.locator('.overlay .ov-quad').first()).toBeVisible()
  await page.locator('.transport input[type=range]').fill('75')
  await expect(page.locator('.viewer-note')).toHaveText('처리 프레임 f 74')
  await shot(page, '04-workbench-video')
})

test('workbench at 1180 px: the media list becomes a drawer', async ({ page }) => {
  await page.setViewportSize({ width: 1180, height: 800 })
  await page.goto(`/?media=${media['synth_mission_a.jpg'].id}`)
  await expect(page.locator('.overlay .ov-quad').first()).toBeVisible()
  await shot(page, '05-workbench-1180')
})

test('history', async ({ page }) => {
  await page.goto('/runs')
  await expect(page.locator('.run-row').first()).toBeVisible()
  await shot(page, '06-runs')
})

test('compare: side by side and overlaid', async ({ page }) => {
  await page.goto(`/compare?a=${runs.hsvA.id}&b=${runs.boardA.id}`)
  await expect(page.locator('.cmp-pane .ov-quad').first()).toBeVisible()
  await shot(page, '07-compare')
  await page.getByRole('button', { name: '겹쳐 보기' }).click()
  await expect(page.locator('.overlay .ov-quad[stroke-dasharray]').first()).toBeAttached()
  await shot(page, '08-compare-overlay')
})

test('models and storage', async ({ page }) => {
  await page.goto('/models')
  await expect(page.locator('.pipe').first()).toBeVisible()
  await page.locator('.pipe-toggle').first().click()
  await shot(page, '09-models')
})
