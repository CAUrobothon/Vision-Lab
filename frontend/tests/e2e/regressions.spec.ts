// Checks for reviewed bugs (one test per area), against the real backend. Media and runs are made
// through the API so each test only drives the part of the UI it checks. The media come from
// regress/ (a differently salted copy), so these runs never appear in the other specs' media.
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { basename, join } from 'node:path'
import type { Media, ModelInfo, Run, RunCreate } from '../../src/api/types'
import { mediaFile } from './helpers'

const MODEL = 'cv.hsv_ranges' // a pipeline with an ROI param and debug layers

async function upload(request: APIRequestContext, name: string): Promise<Media> {
  const file = mediaFile(join('regress', name))
  const mimeType = name.endsWith('.mp4') ? 'video/mp4' : 'image/jpeg'
  const res = await request.post('/api/media', { multipart: { file: { name: basename(file), mimeType, buffer: readFileSync(file) } } })
  expect(res.status()).toBe(200)
  return (await res.json()).media as Media
}

async function runAndWait(request: APIRequestContext, body: RunCreate): Promise<Run> {
  let run = (await (await request.post('/api/runs', { data: body })).json()) as Run
  await expect.poll(async () => (run = await (await request.get(`/api/runs/${run.id}`)).json()).status, { timeout: 90_000 }).toBe('done')
  return run
}

/** localStorage values set before the app's scripts run (keys without the "vl1." prefix). */
async function presetStorage(page: Page, values: Record<string, unknown>) {
  const entries = Object.entries(values).map(([k, v]) => ['vl1.' + k, JSON.stringify(v)])
  await page.addInitScript((list) => {
    for (const [k, v] of list) window.localStorage.setItem(k, v)
  }, entries)
}

const ONLY_ROI_LAYERS = { quad: false, center: false, label: false, raw: false, roi: true, debug: false }

interface PaneView {
  s: number
  tx: number
  ty: number
  cw: number
  ch: number
  w: number
  h: number
}

/** Each viewer's zoom and pan as drawn: transform of the media box, stage size and media size. */
function paneViews(page: Page): Promise<PaneView[]> {
  return page.evaluate(() =>
    [...document.querySelectorAll('.viewer')].map((v) => {
      const stage = v.querySelector('.viewer-stage')!.getBoundingClientRect()
      const box = v.querySelector('.viewer-content') as HTMLElement
      const m = /translate\(([-\d.e]+)px, ([-\d.e]+)px\) scale\(([\d.e-]+)\)/.exec(box.style.transform)!
      return { tx: Number(m[1]), ty: Number(m[2]), s: Number(m[3]), cw: stage.width, ch: stage.height, w: parseFloat(box.style.width), h: parseFloat(box.style.height) }
    }),
  )
}

/** Average grey level of screen pixels (page coordinates), read from a screenshot. */
async function screenGrey(page: Page, pts: [number, number][]): Promise<number[]> {
  const png = (await page.screenshot()).toString('base64')
  return page.evaluate(
    async ({ png, pts }) => {
      const img = new Image()
      img.src = 'data:image/png;base64,' + png
      await img.decode()
      const c = document.createElement('canvas')
      c.width = img.width
      c.height = img.height
      const ctx = c.getContext('2d')!
      ctx.drawImage(img, 0, 0)
      return pts.map(([x, y]) => {
        const d = ctx.getImageData(Math.round(x), Math.round(y), 1, 1).data
        return (d[0] + d[1] + d[2]) / 3
      })
    },
    { png, pts },
  )
}

const frameShown = async (page: Page, i = 0) => Number(/f (\d+)/.exec((await page.locator('.transport-frame').nth(i).textContent()) ?? '')?.[1])

test('ROI: overlapping include polygons form one region, like the backend', async ({ page, request }) => {
  const media = await upload(request, 'scene_a.jpg')
  const square = (x1: number, y1: number, x2: number, y2: number) => [
    [x1, y1],
    [x2, y1],
    [x2, y2],
    [x1, y2],
  ]
  await presetStorage(page, {
    model: MODEL,
    layers: ONLY_ROI_LAYERS,
    [`params.${MODEL}`]: {
      roi: [
        { mode: 'include', points: square(100, 100, 500, 400) },
        { mode: 'include', points: square(300, 200, 700, 500) },
      ],
    },
  })
  await page.goto(`/?media=${media.id}`)
  await expect(page.locator('.overlay .ov-roi-line')).toHaveCount(2)
  await page.waitForTimeout(300)

  const stage = (await page.locator('.viewer-stage').boundingBox())!
  const [view] = await paneViews(page)
  const at = ([x, y]: [number, number]): [number, number] => [stage.x + view.tx + x * view.s, stage.y + view.ty + y * view.s]
  const pts = [at([400, 300]), at([200, 380]), at([1000, 600])] // both includes, one include, outside
  const withRoi = await screenGrey(page, pts)
  await page.addStyleTag({ content: '.overlay { visibility: hidden }' })
  const plain = await screenGrey(page, pts)
  expect(Math.abs(withRoi[0] - plain[0]), 'the overlap of two includes is inside the region').toBeLessThan(4)
  expect(Math.abs(withRoi[1] - plain[1]), 'inside one include').toBeLessThan(4)
  expect(withRoi[2], 'outside every include is dimmed').toBeLessThan(plain[2] * 0.8)
})

test('a failed request while the server is down keeps the selection; a 404 drops it', async ({ page, request }) => {
  const media = await upload(request, 'scene_a.jpg')
  const run = await runAndWait(request, { media_id: media.id, model_key: MODEL, params: {} })
  await page.goto(`/?media=${media.id}&run=${run.id}`)
  await expect(page.locator('.mode-badge.is-run')).toContainText(`기록 #${run.id}`)
  await page.waitForTimeout(2500) // past the media query's staleTime, so a focus refetches it

  await page.route('**/api/**', (r) => r.abort('connectionrefused'))
  const failed = page.waitForEvent('requestfailed', { predicate: (r) => r.url().endsWith(`/api/media/${media.id}`), timeout: 10_000 })
  await page.evaluate(() => {
    for (const state of ['hidden', 'visible']) {
      Object.defineProperty(document, 'visibilityState', { value: state, configurable: true })
      document.dispatchEvent(new Event('visibilitychange', { bubbles: true })) // react-query listens on window
    }
  })
  await failed
  await page.waitForTimeout(500)
  expect(new URL(page.url()).search).toBe(`?media=${media.id}&run=${run.id}`)
  await page.unroute('**/api/**')
  await page.waitForTimeout(1500)
  expect(new URL(page.url()).search).toBe(`?media=${media.id}&run=${run.id}`)

  // A media that does not exist (404) is dropped; the newest media opens instead.
  await page.goto('/?media=987654')
  await expect.poll(() => new URL(page.url()).searchParams.get('media')).not.toBe('987654')
})

test('preview: one request in flight, the newest params win, only changed params are stored', async ({ page, request }) => {
  const media = await upload(request, 'scene_a.jpg')
  await presetStorage(page, { model: MODEL })
  const first = page.waitForResponse((r) => r.url().endsWith('/api/preview') && r.status() === 200)
  await page.goto(`/?media=${media.id}`)
  await first

  // A slow server: every preview takes 400 ms more.
  await page.route('**/api/preview', async (route) => {
    await new Promise((r) => setTimeout(r, 400))
    await route.continue()
  })
  let inFlight = 0
  let maxInFlight = 0
  let started = 0
  let aborted = 0
  const isPreview = (u: string) => u.endsWith('/api/preview')
  page.on('request', (r) => {
    if (!isPreview(r.url())) return
    started++
    maxInFlight = Math.max(maxInFlight, ++inFlight)
  })
  page.on('requestfinished', (r) => isPreview(r.url()) && inFlight--)
  page.on('requestfailed', (r) => {
    if (!isPreview(r.url())) return
    inFlight--
    aborted++
  })

  const control = page.locator('.pc:has(> input[type=range].slider)').first()
  const name = (await control.getAttribute('data-param'))!
  const slider = control.locator('> input[type=range]')
  const [min, max] = [Number(await slider.getAttribute('min')), Number(await slider.getAttribute('max'))]
  const model = ((await (await request.get('/api/models')).json()) as ModelInfo[]).find((m) => m.key === MODEL)!
  const spec = model.params.find((p) => p.name === name)!
  const last = spec.default === max ? min : max
  const values = Array.from({ length: 11 }, (_, i) => min + ((max - min) * (i + 1)) / 13).map((v) => (spec.type === 'int' ? Math.round(v) : v))
  values.push(last)

  const answered = page.waitForResponse(
    (r) => isPreview(r.url()) && r.status() === 200 && JSON.parse(r.request().postData() ?? '{}').params?.[name] === last,
  )
  for (const v of values) {
    await slider.fill(String(v))
    await page.waitForTimeout(50)
  }
  await answered
  await page.waitForTimeout(800)
  expect(maxInFlight, 'never two previews at once').toBe(1)
  expect(aborted, 'no request is aborted (the server would keep computing it)').toBe(0)
  expect(started, 'the values in between are skipped').toBeLessThan(values.length)
  await expect(page.locator('.mode-badge.is-preview')).not.toHaveClass(/pending/)

  // Only the changed value is remembered, not a copy of every default.
  const stored = await page.evaluate((k) => JSON.parse(window.localStorage.getItem(k) ?? 'null'), `vl1.params.${MODEL}`)
  expect(stored).toEqual({ [name]: last })
})

test('video: the debug image belongs to the preview frame only', async ({ page, request }) => {
  const media = await upload(request, 'frames.mp4')
  await presetStorage(page, { model: MODEL, layers: { debug: true } })
  const first = page.waitForResponse((r) => r.url().endsWith('/api/preview') && r.status() === 200)
  await page.goto(`/?media=${media.id}`)
  await first
  const debugDrawn = page.locator('.overlay .ov-debug-tint, .viewer-content > .debug-img')
  await expect(debugDrawn).toHaveCount(1)

  // Seek while the preview for the new frame is slow: the old frame's mask must not stay drawn.
  await page.route('**/api/preview', async (route) => {
    await new Promise((r) => setTimeout(r, 2000))
    await route.continue()
  })
  const at30 = page.waitForResponse((r) => r.url().endsWith('/api/preview') && JSON.parse(r.request().postData() ?? '{}').frame === 30)
  await page.locator('.transport input[type=range]').fill('30')
  await expect.poll(() => frameShown(page)).toBe(30)
  await expect(debugDrawn).toHaveCount(0)
  await at30
  await expect(debugDrawn).toHaveCount(1)
  await page.unroute('**/api/preview')

  // Playing: no preview, so no debug image either.
  await page.getByRole('button', { name: '재생', exact: true }).click()
  for (let i = 0; i < 5; i++) {
    await page.waitForTimeout(80)
    if (await page.getByRole('button', { name: '일시정지', exact: true }).isVisible()) await expect(debugDrawn).toHaveCount(0)
  }
})

test('video runs: frames fetched one copy at a time, nothing drawn outside the range, compare frame after a mode switch', async ({
  page,
  request,
}) => {
  const media = await upload(request, 'frames.mp4')
  const models = (await (await request.get('/api/models')).json()) as ModelInfo[]
  const model = models.find((m) => m.kind === 'pipeline' && m.state === 'ready')!.key

  // A full run watched while it runs: one /frames request at a time, ending with every frame.
  let framesInFlight = 0
  let framesMax = 0
  page.on('request', (r) => /\/api\/runs\/\d+\/frames$/.test(r.url()) && (framesMax = Math.max(framesMax, ++framesInFlight)))
  page.on('requestfinished', (r) => /\/api\/runs\/\d+\/frames$/.test(r.url()) && framesInFlight--)
  page.on('requestfailed', (r) => /\/api\/runs\/\d+\/frames$/.test(r.url()) && framesInFlight--)
  const full = (await (await request.post('/api/runs', { data: { media_id: media.id, model_key: model, params: {} } })).json()) as Run
  await page.goto(`/?media=${media.id}&run=${full.id}`)
  await expect(page.locator('.run-tile.selected .status')).toHaveText('완료', { timeout: 90_000 })
  // The last frame is drawn from its own result: the cached copy caught up with the finished run.
  await page.locator('.transport input[type=range]').fill('59')
  await expect.poll(() => frameShown(page)).toBe(59)
  await expect(page.locator('.overlay .ov-dot')).toHaveCount(1)
  await expect(page.locator('.viewer-note')).toHaveCount(0)
  expect(framesMax).toBe(1)

  // A run of frames 10..19, every 2nd: drawn inside, nothing (and a note) outside.
  const part = await runAndWait(request, { media_id: media.id, model_key: model, params: {}, frame_start: 10, frame_end: 20, stride: 2 })
  await page.goto(`/?media=${media.id}&run=${part.id}`)
  await expect(page.locator('.transport')).toBeVisible()
  const scrub = page.locator('.transport input[type=range]')
  const dots = page.locator('.overlay .ov-dot')
  const note = page.locator('.viewer-note')
  for (const [f, drawn, text] of [
    [15, 1, '처리 프레임 f 14'],
    [19, 1, '처리 프레임 f 18'],
    [25, 0, '이 구간은 처리하지 않음'],
    [5, 0, '이 구간은 처리하지 않음'],
    [12, 1, null],
  ] as const) {
    await scrub.fill(String(f))
    await expect.poll(() => frameShown(page)).toBe(f)
    await expect(dots).toHaveCount(drawn)
    if (text) await expect(note).toHaveText(text)
    else await expect(note).toHaveCount(0)
  }
  await scrub.fill('40')
  await expect.poll(() => frameShown(page)).toBe(40)
  await expect(page.locator('.result')).toContainText('이 구간은 처리하지 않았습니다.')

  // Compare: after switching to the overlay view the panel describes the frame on screen.
  await page.goto(`/compare?a=${full.id}&b=${part.id}`)
  await expect(page.locator('.transport')).toBeVisible()
  await page.locator('.transport input[type=range]').fill('30')
  await expect.poll(() => frameShown(page)).toBe(30)
  await expect(page.locator('.cmp-title').first()).toContainText('f 30')
  await page.getByRole('button', { name: '겹쳐 보기' }).click()
  await expect(page.locator('.cmp-pane')).toHaveCount(0)
  await expect(page.locator('.transport')).toBeVisible()
  await page.waitForTimeout(500)
  const shown = await frameShown(page)
  await expect(page.locator('.cmp-title').first()).toContainText(`f ${shown}`)
  await page.getByRole('button', { name: '나란히' }).click()
  await expect(page.locator('.cmp-pane')).toHaveCount(2)
  await page.waitForTimeout(500)
  await expect(page.locator('.cmp-title').first()).toContainText(`f ${await frameShown(page)}`)
})

test('compare: zoom and pan line up across media sizes; an image with a video keeps the video movable', async ({ page, request }) => {
  const big = await upload(request, 'scene_a.jpg')
  const small = await upload(request, 'scene_small.jpg')
  expect([small.width, small.height]).toEqual([640, 480])
  const runBig = await runAndWait(request, { media_id: big.id, model_key: MODEL, params: {} })
  const runSmall = await runAndWait(request, { media_id: small.id, model_key: MODEL, params: {} })

  await page.goto(`/compare?a=${runBig.id}&b=${runSmall.id}`)
  await expect(page.locator('.cmp-pane .viewer-content img')).toHaveCount(2)
  const stage = (await page.locator('.cmp-pane .viewer-stage').first().boundingBox())!
  await page.mouse.move(stage.x + stage.width * 0.4, stage.y + stage.height * 0.6)
  await page.mouse.wheel(0, -300)
  await page.waitForTimeout(300)
  await page.mouse.down()
  await page.mouse.move(stage.x + stage.width * 0.4 + 60, stage.y + stage.height * 0.6 + 30, { steps: 6 })
  await page.mouse.up()
  await page.waitForTimeout(300)

  // Zoom relative to "fit" and the media point (0..1) at the stage centre must agree.
  const norm = (p: PaneView) => {
    const fit = Math.max(0.01, Math.min((p.cw - 24) / p.w, (p.ch - 24) / p.h))
    return { z: p.s / fit, u: (p.cw / 2 - p.tx) / p.s / p.w, v: (p.ch / 2 - p.ty) / p.s / p.h }
  }
  const [a, b] = (await paneViews(page)).map(norm)
  expect(a.z).toBeGreaterThan(1.2) // really zoomed in
  expect(Math.abs(a.z - b.z)).toBeLessThan(1e-3)
  expect(Math.abs(a.u - b.u)).toBeLessThan(1e-3)
  expect(Math.abs(a.v - b.v)).toBeLessThan(1e-3)

  // Image A with video B: B gets its own transport and the panel follows it.
  const video = await upload(request, 'frames.mp4')
  const runVideo = await runAndWait(request, { media_id: video.id, model_key: MODEL, params: {}, stride: 5 })
  await page.goto(`/compare?a=${runBig.id}&b=${runVideo.id}`)
  const transport = page.locator('.cmp-pane').nth(1).locator('.transport')
  await expect(transport).toBeVisible()
  await transport.locator('input[type=range]').fill('27')
  await expect.poll(() => frameShown(page)).toBe(27)
  await expect(page.locator('.cmp-title').first()).toContainText('f 27')
  await expect(page.locator('.cmp-table').first()).toContainText('f 25')
})
