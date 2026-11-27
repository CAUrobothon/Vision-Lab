// Screen regression: (1) layout — at the laptop sizes the product targets, no control of the
// right panel, the left rail or the JSON dialog sticks out of its container, for every model;
// (2) pixels — key components compared with committed baseline images
// (`npm run test:visual:update` rewrites them after an intended change; look at them first);
// (3) the built app that ./run.sh serves loads, without the "new version" banner.
import { expect, test, type Page } from '@playwright/test'
import { GT_IMAGES, apiBatch, apiDataset, apiGt, apiUpload, mediaFile, uploadViaRail } from './helpers'

const SIZES = [
  { width: 1180, height: 800 },
  { width: 1280, height: 800 },
  { width: 1440, height: 900 },
]
const MODELS = ['HSV 색 범위', '판 차이 + Lab 최근접색', '윤곽으로 닫힌 사각형', 'Lab 자동 군집']

/** Controls inside `selector` whose box crosses the container's left or right edge. */
async function overflowing(page: Page, selector: string): Promise<string[]> {
  return page.evaluate((sel) => {
    const root = document.querySelector(sel)
    if (!root) return [`missing ${sel}`]
    const r = root.getBoundingClientRect()
    const out: string[] = []
    if (r.right > window.innerWidth + 0.5) out.push(`${sel} itself ends at ${Math.round(r.right)} > window ${window.innerWidth}`)
    for (const el of root.querySelectorAll<HTMLElement>('button, select, input, textarea, a, label, .pc-head, .pg-head, h2')) {
      const b = el.getBoundingClientRect()
      if (b.width === 0 || b.height === 0) continue
      if (b.right > r.right + 0.5 || b.left < r.left - 0.5) {
        const name = el.getAttribute('aria-label') ?? el.textContent?.trim().slice(0, 24) ?? ''
        out.push(`${el.tagName.toLowerCase()} "${name}" x ${Math.round(b.left)}–${Math.round(b.right)} outside ${Math.round(r.left)}–${Math.round(r.right)}`)
      }
    }
    return out
  }, selector)
}

async function chooseModel(page: Page, name: string) {
  await page.locator('.model-current').click()
  await page.getByRole('option', { name: new RegExp(name.replace(/[+]/g, '\\+')) }).click()
  await expect(page.locator('.model-current')).toContainText(name)
}

test('layout: nothing sticks out of the panels at laptop sizes', async ({ page }) => {
  await page.goto('/')
  await uploadViaRail(page, mediaFile('scene_a.jpg'))
  for (const size of SIZES) {
    await page.setViewportSize(size)
    for (const name of MODELS) {
      await chooseModel(page, name)
      for (const sel of ['.panel-top', '.panel-params', '.rail']) {
        expect(await overflowing(page, sel), `${name} at ${size.width}px: ${sel}`).toEqual([])
      }
    }
    await page.getByRole('button', { name: 'JSON' }).click()
    expect(await overflowing(page, '.pj-body'), `JSON dialog at ${size.width}px`).toEqual([])
    await page.keyboard.press('Escape')
  }
})

test('pixels: JSON dialog states, preset row and dataset panel', async ({ page, request }) => {
  // Other specs share the data folder: their presets of this model would show up in the captures.
  for (const p of await (await request.get('/api/presets?model_key=cv.edge_quads')).json()) {
    await request.delete(`/api/presets/${p.id}`)
  }
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.goto('/')
  await uploadViaRail(page, mediaFile('scene_a.jpg'))
  await chooseModel(page, '윤곽으로 닫힌 사각형')
  const shot = { animations: 'disabled' as const, caret: 'hide' as const, maxDiffPixels: 60 }

  await page.getByRole('button', { name: 'JSON' }).click()
  const dialog = page.getByRole('dialog', { name: /파라미터 JSON/ })
  const box = dialog.getByLabel('파라미터 JSON')
  await box.fill('{"model": "cv.edge_quads", "params": {"canny": [25, 75]}}')
  await expect(dialog.locator('.pj-apply')).toBeEnabled()
  await expect(dialog).toHaveScreenshot('json-ready.png', shot)
  await dialog.locator('.pj-apply').click()
  await expect(dialog.locator('.pj-apply')).toHaveText('적용됨')
  await expect(dialog).toHaveScreenshot('json-applied.png', shot)
  await box.fill('{"canny": [25, 75],,}')
  await expect(dialog.locator('#pj-state .error-text')).toBeVisible()
  await expect(dialog).toHaveScreenshot('json-error.png', shot)
  await page.keyboard.press('Escape')

  await page.getByRole('button', { name: '새 프리셋으로 저장' }).click()
  await page.getByLabel('새 프리셋 이름').fill('시각 회귀')
  await page.getByRole('button', { name: '저장', exact: true }).click()
  await expect(page.locator('.panel-top')).toHaveScreenshot('preset-row.png', { ...shot, mask: [page.locator('.model-picker')] })
  const lo = page.locator('[data-param=canny] .num-input').first()
  await lo.fill('35')
  await lo.press('Enter')
  await expect(page.getByRole('button', { name: '되돌리기' })).toBeVisible()
  await expect(page.locator('.panel-top')).toHaveScreenshot('preset-row-modified.png', { ...shot, mask: [page.locator('.model-picker')] })
  expect(await overflowing(page, '.panel-top')).toEqual([])

  await page.getByRole('button', { name: '새로' }).click()
  await page.getByLabel('데이터셋 이름').fill('시각 회귀 데이터셋')
  await page.getByRole('button', { name: '만들기' }).click()
  await expect(page.locator('.ds-batch')).toContainText('아직 일괄 실행이 없습니다')
  await expect(page.locator('.ds-bar')).toHaveScreenshot('dataset-empty.png', shot)
  await expect(page.locator('.ds-batch')).toHaveScreenshot('batch-panel-empty.png', shot)

  // The rail with a two-level folder tree; the selected folder shows its actions and scopes the run.
  for (const [parent, name] of [['', '원본'], ['', '증강1'], ['증강1', '어둡게']]) {
    await page.locator(`.ft-row[data-path="${parent}"] .ft-main`).click()
    await page.getByRole('button', { name: '새 폴더' }).click()
    await page.getByLabel('새 폴더 이름').fill(name)
    await page.getByLabel('새 폴더 이름').press('Enter')
    await expect(page.locator(`.ft-row[data-path="${parent ? `${parent}/` : ''}${name}"]`)).toBeVisible()
  }
  await page.locator('.ft-row[data-path="증강1"] .ft-main').click()
  await expect(page.getByRole('button', { name: '지금 모델로 이 폴더 실행' })).toBeVisible()
  await expect(page.locator('.rail')).toHaveScreenshot('rail-folder-tree.png', shot)
  expect(await overflowing(page, '.rail')).toEqual([])
})

test('pixels: 채점 page, the GT row with its link dialog, the viewer with the 정답 layer', async ({ page, request }) => {
  // Labels of tests/e2e/make_media.py write_gt_set plus gt_extra.jpg, which only a file name links.
  const did = await apiDataset(request, '시각 회귀 채점')
  for (const f of [...GT_IMAGES, 'gt_extra.jpg']) await apiUpload(request, mediaFile(`gt/${f}`), did)
  const gid = await apiGt(request, did, mediaFile('gt/labels_link.json'))
  const shot = { animations: 'disabled' as const, caret: 'hide' as const, maxDiffPixels: 60 }

  await page.setViewportSize({ width: 1440, height: 900 })
  await page.goto('/')
  await page.getByLabel('데이터셋', { exact: true }).selectOption(String(did))
  const row = page.getByRole('region', { name: '정답셋' })
  await expect(row.locator('.gt-links')).toContainText('연결 9/10')
  await expect(page.locator('.ds-bar')).toHaveScreenshot('gt-row.png', shot)
  expect(await overflowing(page, '.rail')).toEqual([])
  await row.getByRole('button', { name: '연결 안 됨 1' }).click()
  const dialog = page.getByRole('dialog', { name: /정답 연결 확인/ })
  await expect(dialog.locator('.gl-row', { hasText: 'gt_extra.jpg' })).toBeVisible()
  // The body, not the dialog: the dialog's fractional bottom edge blends with the page behind it.
  await expect(dialog.locator('.pj-body')).toHaveScreenshot('gt-link-dialog.png', shot)
  await dialog.getByRole('button', { name: '후보 모두 연결' }).click()
  await expect(row.locator('.gt-links')).toContainText('연결 10/10')
  await dialog.getByRole('button', { name: '닫기' }).click()

  // Three batches: the reference, another pipeline (compared), one folder only (excluded).
  await request.post(`/api/gt/${gid}/organize`, { data: { root: '' } })
  await apiBatch(request, did)
  await apiBatch(request, did, { model_key: 'cv.board_diff_lab' })
  await apiBatch(request, did, { folder: 'cool' })

  // Tall enough that the whole page is one capture; ids, times and the clock vary between runs.
  await page.setViewportSize({ width: 1440, height: 3000 })
  await page.goto(`/score?dataset=${did}&gt=${gid}`)
  await expect(page.getByRole('region', { name: '최종 순위' }).locator('tbody tr')).toHaveCount(2, { timeout: 30_000 })
  await expect(page.getByRole('region', { name: '이미지별 결과' }).locator('tbody tr')).toHaveCount(10)
  await expect(page.locator('.sc-section.stale')).toHaveCount(0)
  // Also masked: the free disk space in the top bar and the batch picker, whose "#id" depends on
  // how many batches earlier specs made on the shared test server.
  const vol = [page.locator('.vol'), page.locator('.sc-id'), page.locator('.sc-status'), page.locator('.topbar-status'),
    page.getByLabel('자세히 볼 배치')]
  await expect(page).toHaveScreenshot('score-page.png', { ...shot, mask: vol, fullPage: false })
  for (const size of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
    await page.setViewportSize(size)
    expect(await overflowing(page, '.score-page'), `score page at ${size.width}px`).toEqual([])
  }

  // From the per-image list into the viewer, with the 정답 layer on.
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.getByRole('region', { name: '이미지별 결과' }).locator('tbody tr', { hasText: 's2_dark_ev_m0p5.jpg' }).click()
  await expect(page.locator('.overlay .ov-gt-miss')).toHaveCount(1)
  await page.locator('.result-table tbody tr').nth(2).hover()
  await expect(page.locator('.gt-card')).toContainText('꼭짓점 RMS')
  await expect(page.locator('.wb-viewer')).toHaveScreenshot('viewer-gt-layer.png', { ...shot, mask: [page.locator('.viewer-badge')] })
})

test('the built app that ./run.sh serves loads without the new-version banner', async ({ page }) => {
  await page.goto('http://127.0.0.1:8012/')
  await expect(page.getByRole('button', { name: 'JSON' })).toBeVisible()
  await expect(page.getByText('화면이 새 버전으로 바뀌었습니다')).toHaveCount(0)
  const res = await page.request.get('http://127.0.0.1:8012/')
  expect(res.headers()['cache-control']).toBe('no-cache')
})
