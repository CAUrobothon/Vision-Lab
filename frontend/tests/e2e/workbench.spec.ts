// Image flow against the real backend: upload → live preview with cubes drawn → a slider change
// re-runs the preview and the overlay follows → save two runs → they show in the strip and on
// /runs (star, note, filter) → compare them side by side and overlaid. Plus drop-anywhere upload.
import { expect, test, type Page } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { drawnCubes, mediaFile, uploadViaRail } from './helpers'

/** Moves the first single-value slider to `edge` and returns the preview that answered it. */
async function moveFirstSlider(page: Page, edge: 'End' | 'Home') {
  const control = page.locator('.pc:has(> input[type=range].slider)').first()
  const name = (await control.getAttribute('data-param'))!
  const slider = control.locator('> input[type=range]')
  const target = Number(await slider.getAttribute(edge === 'End' ? 'max' : 'min'))
  const answered = page.waitForResponse((r) => {
    if (!r.url().endsWith('/api/preview') || r.status() !== 200) return false
    return JSON.parse(r.request().postData() ?? '{}').params?.[name] === target
  })
  await slider.focus()
  await page.keyboard.press(edge)
  const body = await (await answered).json()
  return { name, target, cubes: body.cubes as { quad: unknown; box: unknown }[] }
}

test('image: upload, live preview, slider, save, history, compare', async ({ page }) => {
  await page.goto('/')
  const mediaId = await uploadViaRail(page, mediaFile('scene_a.jpg'))

  // Pipelines preview at once: the badge says so and the cubes are drawn.
  await expect(page.locator('.mode-badge.is-preview')).toContainText('미리보기 · 저장 안 됨')
  await expect(drawnCubes(page).first()).toBeVisible()
  const first = await drawnCubes(page).count()
  expect(first).toBeGreaterThan(0)
  await expect(page.locator('.result-table tbody tr')).toHaveCount(first)

  // A slider change sends a new preview with that value; the overlay shows exactly its cubes.
  const high = await moveFirstSlider(page, 'End')
  await expect(drawnCubes(page)).toHaveCount(high.cubes.filter((c) => c.quad || c.box).length)

  // Save → a finished run appears in this media's strip.
  await page.getByRole('button', { name: '기록으로 저장' }).click()
  await expect(page.locator('.run-tile')).toHaveCount(1)
  await expect(page.locator('.run-tile').first().locator('.status')).toHaveText('완료', { timeout: 60_000 })

  // A second run with the slider at the other end.
  const low = await moveFirstSlider(page, 'Home')
  await expect(drawnCubes(page)).toHaveCount(low.cubes.filter((c) => c.quad || c.box).length)
  await page.getByRole('button', { name: '기록으로 저장' }).click()
  await expect(page.locator('.run-tile')).toHaveCount(2)
  await expect(page.locator('.run-tile .status').first()).toHaveText('완료', { timeout: 60_000 })

  // Clicking a run shows it in the viewer.
  await page.locator('.run-tile .run-main').first().click()
  await expect(page).toHaveURL(/run=\d+/)
  await expect(page.locator('.mode-badge.is-run')).toContainText('기록 #')
  await expect(drawnCubes(page)).toHaveCount(low.cubes.filter((c) => c.quad || c.box).length)

  // History: both runs, star one, write a note, filter by star.
  await page.goto(`/runs?media=${mediaId}`)
  await expect(page.locator('.run-row')).toHaveCount(2)
  const newest = page.locator('.run-row').first()
  await newest.getByRole('button', { name: /별표$/ }).click()
  await expect(newest.getByRole('button', { name: /별표 해제/ })).toBeVisible()
  await newest.getByRole('button', { name: /메모 편집/ }).click()
  await newest.getByRole('textbox', { name: /메모/ }).fill('e2e 메모')
  await page.keyboard.press('Enter')
  await expect(newest.getByRole('button', { name: /메모 편집/ })).toHaveText('e2e 메모')
  await page.getByRole('button', { name: '별표만' }).click()
  await expect(page.locator('.run-row')).toHaveCount(1)
  await page.getByRole('button', { name: '별표만' }).click()
  await expect(page.locator('.run-row')).toHaveCount(2)

  // Select both and compare.
  for (const box of await page.locator('.run-row .rr-check input').all()) await box.check()
  await page.getByRole('button', { name: '비교' }).click()
  await expect(page).toHaveURL(/\/compare\?a=\d+&b=\d+/)
  await expect(page.locator('.cmp-pane .viewer')).toHaveCount(2)
  await expect(page.locator('.cmp-panel')).toContainText('파라미터 차이')
  await expect(page.locator('.cmp-panel .cmp-table').last()).not.toContainText('파라미터가 같습니다')

  // Linked zoom: a wheel on A zooms B the same way.
  const stage = await page.locator('.cmp-pane').first().locator('.viewer-stage').boundingBox()
  await page.mouse.move(stage!.x + stage!.width / 2, stage!.y + stage!.height / 2)
  await page.mouse.wheel(0, -500)
  await expect
    .poll(() => page.evaluate(() => [...document.querySelectorAll('.viewer-content')].map((e) => (e as HTMLElement).style.transform)))
    .toEqual(expect.arrayContaining([expect.stringContaining('scale(')]))
  const transforms = await page.evaluate(() => [...document.querySelectorAll('.viewer-content')].map((e) => (e as HTMLElement).style.transform))
  expect(transforms[0]).toBe(transforms[1])

  // Overlay mode: one viewer, B dashed.
  await page.getByRole('button', { name: '겹쳐 보기' }).click()
  await expect(page.locator('.cmp-pane')).toHaveCount(0)
  await expect(page.locator('.viewer')).toHaveCount(1)
  await expect(page.locator('.overlay .ov-quad[stroke-dasharray]').first()).toBeAttached()
})

test('dropping a file anywhere uploads it', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('.rail')).toBeVisible()
  const bytes = readFileSync(mediaFile('scene_b.jpg')).toString('base64')
  const uploaded = page.waitForResponse((r) => r.url().endsWith('/api/media') && r.request().method() === 'POST')
  await page.evaluate(async (b64) => {
    const bin = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))
    const dt = new DataTransfer()
    dt.items.add(new File([bin], 'dropped-scene.jpg', { type: 'image/jpeg' }))
    document.body.dispatchEvent(new DragEvent('dragenter', { dataTransfer: dt, bubbles: true, cancelable: true }))
  }, bytes)
  await expect(page.locator('.drop-overlay')).toBeVisible()
  await page.evaluate(async (b64) => {
    const bin = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))
    const dt = new DataTransfer()
    dt.items.add(new File([bin], 'dropped-scene.jpg', { type: 'image/jpeg' }))
    document.body.dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true }))
  }, bytes)
  await expect(page.locator('.drop-overlay')).toHaveCount(0)
  const id = (await (await uploaded).json()).media.id as number
  await expect.poll(() => new URL(page.url()).searchParams.get('media')).toBe(String(id))
  await expect(drawnCubes(page).first()).toBeVisible()
})

test('pasting an image from the clipboard uploads it', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('.rail')).toBeVisible()
  const bytes = readFileSync(mediaFile('scene_b.jpg'))
  // A different byte at the end of the JPEG keeps it a valid image but a new file for the lab.
  bytes[bytes.length - 3] ^= 0xff
  const uploaded = page.waitForResponse((r) => r.url().endsWith('/api/media') && r.request().method() === 'POST')
  await page.evaluate(async (b64) => {
    const bin = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))
    const dt = new DataTransfer()
    dt.items.add(new File([bin], 'image.jpg', { type: 'image/jpeg' }))
    window.dispatchEvent(new ClipboardEvent('paste', { clipboardData: dt, bubbles: true, cancelable: true }))
  }, bytes.toString('base64'))
  const body = await (await uploaded).json()
  expect(body.media.name).toMatch(/^붙여넣기-\d{14}\.jpg$/)
  await expect.poll(() => new URL(page.url()).searchParams.get('media')).toBe(String(body.media.id))
})
