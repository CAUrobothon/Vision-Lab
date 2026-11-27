import { expect, type APIRequestContext, type Locator, type Page } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { basename, join } from 'node:path'

export function mediaFile(name: string): string {
  const dir = process.env.E2E_MEDIA
  if (!dir) throw new Error('E2E_MEDIA is not set (global-setup did not run)')
  return join(dir, name)
}

/** Files picked in the chooser that clicking `trigger` opens (the user's path, not the input behind it). */
export async function chooseFiles(page: Page, trigger: Locator, files: string | string[]): Promise<void> {
  const chooser = page.waitForEvent('filechooser')
  await trigger.click()
  await (await chooser).setFiles(files)
}

/** Upload with the rail's 올리기 button; resolves with the new media id once it is selected. */
export async function uploadViaRail(page: Page, file: string): Promise<number> {
  const uploaded = page.waitForResponse((r) => new URL(r.url()).pathname === '/api/media' && r.request().method() === 'POST', {
    timeout: 120_000,
  })
  const rail = page.getByRole('complementary', { name: '미디어 목록' })
  await chooseFiles(page, rail.getByRole('button', { name: '올리기', exact: true }), file)
  const res = await uploaded
  expect(res.status()).toBe(200)
  const id = (await res.json()).media.id as number
  await expect.poll(() => new URL(page.url()).searchParams.get('media')).toBe(String(id))
  return id
}

/** Cubes drawn by the overlay (quads, or boxes when a model has no quad). */
export function drawnCubes(page: Page) {
  return page.locator('.overlay .ov-quad')
}

/** Entry names of a zip, in order, read from its central directory (UTF-8 names when flagged). */
export function zipNames(buf: Buffer): string[] {
  const end = buf.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]))
  if (end < 0) throw new Error('not a zip')
  const count = buf.readUInt16LE(end + 10)
  let p = buf.readUInt32LE(end + 16)
  const names: string[] = []
  for (let i = 0; i < count; i++) {
    if (buf.readUInt32LE(p) !== 0x02014b50) throw new Error('bad zip central directory')
    const utf8 = (buf.readUInt16LE(p + 8) & 0x800) !== 0
    const nameLen = buf.readUInt16LE(p + 28)
    const skip = buf.readUInt16LE(p + 30) + buf.readUInt16LE(p + 32)
    names.push(buf.subarray(p + 46, p + 46 + nameLen).toString(utf8 ? 'utf8' : 'latin1'))
    p += 46 + nameLen + skip
  }
  return names
}

// ----- API setup for the scoring specs (datasets, uploads, ground-truth sets, batches) -----

export interface Expected {
  n_images: number
  n_preds: number
  n_cubes: number
  n_valid: number
  tolerance: [number, number]
  tolerance_tc2: [number, number]
  image_pass: [number, number]
  image_pass_tc2: [number, number]
  recall: [number, number]
  precision: [number, number]
  color_accuracy: [number, number]
  failing: string[]
}

/** The 9 labelled images of tests/e2e/make_media.py write_gt_set (3 sources x 3 variants). */
export const GT_IMAGES = ['s1', 's2', 's3'].flatMap((s) => ['original', 'dark_ev_m0p5', 'cool'].map((v) => `${s}_${v}.jpg`))

export function expected(): Expected {
  return JSON.parse(readFileSync(mediaFile('gt/expected.json'), 'utf8')) as Expected
}

/** GT cubes per file name in labels.json. */
export function gtCubeCounts(): Record<string, number> {
  const doc = JSON.parse(readFileSync(mediaFile('gt/labels.json'), 'utf8')) as {
    items: { filename: string; frames: { cubes: unknown[] }[] }[]
  }
  return Object.fromEntries(doc.items.map((it) => [it.filename, it.frames[0].cubes.length]))
}

export async function apiDataset(request: APIRequestContext, name: string): Promise<number> {
  const res = await request.post('/api/datasets', { data: { name } })
  expect(res.status(), await res.text()).toBe(200)
  return (await res.json()).id as number
}

/** Uploads a file (into a dataset when given); returns the media id. */
export async function apiUpload(request: APIRequestContext, file: string, datasetId?: number): Promise<number> {
  const name = basename(file)
  const mimeType = name.endsWith('.mp4') ? 'video/mp4' : name.endsWith('.json') ? 'application/json' : 'image/jpeg'
  const res = await request.post(`/api/media${datasetId === undefined ? '' : `?dataset_id=${datasetId}`}`, {
    multipart: { file: { name, mimeType, buffer: readFileSync(file) } },
    timeout: 120_000,
  })
  expect(res.status(), await res.text()).toBe(200)
  return (await res.json()).media.id as number
}

export async function apiGt(request: APIRequestContext, datasetId: number, file: string): Promise<number> {
  const res = await request.post(`/api/datasets/${datasetId}/gt`, {
    multipart: { file: { name: basename(file), mimeType: 'application/json', buffer: readFileSync(file) } },
  })
  expect(res.status(), await res.text()).toBe(200)
  return (await res.json()).id as number
}

/** A dataset with the 9 labelled images and labels.json; returns ids by file name. */
export async function apiScoredDataset(request: APIRequestContext, name: string) {
  const datasetId = await apiDataset(request, name)
  const media: Record<string, number> = {}
  for (const f of GT_IMAGES) media[f] = await apiUpload(request, mediaFile(`gt/${f}`), datasetId)
  const gtSetId = await apiGt(request, datasetId, mediaFile('gt/labels.json'))
  return { datasetId, gtSetId, media }
}

export async function apiBatch(
  request: APIRequestContext,
  datasetId: number,
  body: { model_key?: string; params?: Record<string, unknown>; folder?: string | null } = {},
  wait = true,
): Promise<number> {
  const res = await request.post(`/api/datasets/${datasetId}/batches`, {
    data: { model_key: 'cv.hsv_ranges', params: {}, stride: 1, folder: null, ...body },
  })
  expect(res.status(), await res.text()).toBe(200)
  const id = (await res.json()).id as number
  if (wait) {
    await expect
      .poll(async () => (await (await request.get(`/api/batches/${id}`)).json()).finished, { timeout: 90_000 })
      .toBe(true)
  }
  return id
}
