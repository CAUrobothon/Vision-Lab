// Video flow against the real backend: upload a clip whose frames encode their own index →
// run every 2nd frame → the overlay is drawn for the frame on screen while stepping and playing.
//
// frames.mp4 (see make_media.py): 10 black/white blocks at the top give the frame index n, and a
// red 61 px cube sits at x = 200 + 8n. A stride-2 run has results for even frames, so on frame n
// the overlay must show the result of frame n - n % 2.
import { expect, test, type Page } from '@playwright/test'
import { mediaFile, uploadViaRail } from './helpers'

interface Probe {
  pixels: number // frame index decoded from the picture on screen
  shown: number // the transport's "f N"
  cx: number | null // overlay centre of the cube
}

function probe(page: Page): Promise<Probe> {
  return page.evaluate(() => {
    const v = document.querySelector('video')!
    const c = document.createElement('canvas')
    c.width = v.videoWidth
    c.height = v.videoHeight
    const ctx = c.getContext('2d', { willReadFrequently: true })!
    ctx.drawImage(v, 0, 0)
    const d = ctx.getImageData(0, 0, c.width, 60).data
    let pixels = 0
    for (let i = 0; i < 10; i++) {
      const p = (30 * c.width + 40 + 48 * i) * 4
      if (d[p] > 128) pixels |= 1 << i
    }
    const shown = Number(/f (\d+)/.exec(document.querySelector('.transport-frame')!.textContent!)![1])
    const dot = document.querySelector('.overlay .ov-dot')
    return { pixels, shown, cx: dot ? Number(dot.getAttribute('cx')) : null }
  })
}

const expectedCx = (frame: number) => 200 + 8 * (frame - (frame % 2))
// Consecutive stored results are 16 px apart, so 4 px still identifies the frame exactly.
const nearCx = (cx: number | null, frame: number) => cx !== null && Math.abs(cx - expectedCx(frame)) < 4

test('video: run every 2nd frame and the overlay follows the picture', async ({ page }) => {
  await page.goto('/')
  await uploadViaRail(page, mediaFile('frames.mp4'))
  await expect(page.locator('.transport')).toBeVisible()
  await expect(page.locator('.transport-frame')).toContainText('/ 59')

  await page.getByLabel('프레임 간격').fill('2')
  await expect(page.locator('.vr-summary')).toContainText('30프레임 처리')
  await page.getByRole('button', { name: '영상 전체 실행' }).click()
  await expect(page).toHaveURL(/run=\d+/)
  await expect(page.locator('.run-tile.selected .status')).toHaveText('완료', { timeout: 90_000 })
  await expect(page.locator('.overlay .ov-dot')).toHaveCount(1)

  // Paused: scrub and step; the picture, the frame number and the overlay must agree exactly.
  const scrub = page.locator('.transport input[type=range]')
  for (const target of [0, 7, 30, 45, 59]) {
    await scrub.fill(String(target))
    await expect.poll(async () => (await probe(page)).shown).toBe(target)
    const p = await probe(page)
    expect(p.pixels).toBe(target)
    expect(nearCx(p.cx, target), `overlay x ${p.cx} on frame ${target}`).toBe(true)
  }
  await scrub.fill('20')
  await expect.poll(async () => (await probe(page)).shown).toBe(20)
  await page.locator('.topbar').click()
  await page.keyboard.press('ArrowRight')
  await expect.poll(async () => (await probe(page)).shown).toBe(21)
  expect((await probe(page)).pixels).toBe(21)
  await expect(page.locator('.viewer-note')).toHaveText('처리 프레임 f 20')
  await page.keyboard.press('Shift+ArrowRight')
  await expect.poll(async () => (await probe(page)).pixels).toBe(31)
  expect((await probe(page)).shown).toBe(31)

  // Playing: sample while the video runs. The frame number and the overlay always describe the
  // same frame; the picture may be one frame ahead (requestVideoFrameCallback can fire a v-sync late).
  await scrub.fill('0')
  await expect.poll(async () => (await probe(page)).shown).toBe(0)
  await page.getByRole('button', { name: '재생', exact: true }).click()
  const samples: Probe[] = []
  for (let i = 0; i < 12; i++) {
    await page.waitForTimeout(90)
    samples.push(await probe(page))
  }
  for (const s of samples) {
    expect(Math.abs(s.pixels - s.shown)).toBeLessThanOrEqual(1)
    if (s.cx !== null) expect(nearCx(s.cx, s.shown), `overlay x ${s.cx} on frame ${s.shown}`).toBe(true)
  }
  expect(new Set(samples.map((s) => s.shown)).size).toBeGreaterThan(3) // it really played

  // Paused again (or ended): exact agreement.
  await page.getByRole('button', { name: /일시정지|재생/ }).first().click()
  await page.waitForTimeout(300)
  const end = await probe(page)
  expect(end.pixels).toBe(end.shown)
  expect(nearCx(end.cx, end.shown)).toBe(true)
})
