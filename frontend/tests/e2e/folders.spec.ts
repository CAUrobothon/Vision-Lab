// Folders inside a dataset, the user's flow: make 원본 and 증강1, upload a directory whose tree
// (원본/a.png, 증강1/b.png, 증강1/sub/c.png, plus OS junk) fills them, see the counts, run the
// current model on 증강1 only, download both zips with the folder paths kept, move a file, then
// rename and delete a folder from the tree. Plus dropped directories (entries read recursively).
import { expect, test, type Page } from '@playwright/test'
import { mediaFile, zipNames } from './helpers'

const folderRow = (page: Page, path: string) => page.locator(`.ft-row[data-path="${path}"]`)
const count = (page: Page, path: string) => folderRow(page, path).locator('.ft-count')

async function newDataset(page: Page, name: string): Promise<number> {
  const made = page.waitForResponse((r) => new URL(r.url()).pathname === '/api/datasets' && r.request().method() === 'POST')
  await page.getByRole('button', { name: '새로' }).click()
  await page.getByLabel('데이터셋 이름').fill(name)
  await page.getByRole('button', { name: '만들기' }).click()
  return (await (await made).json()).id as number
}

async function newFolder(page: Page, name: string) {
  await page.getByRole('button', { name: '새 폴더' }).click()
  await page.getByLabel('새 폴더 이름').fill(name)
  await page.getByLabel('새 폴더 이름').press('Enter')
  await expect(folderRow(page, name)).toBeVisible()
}

test('folders: make, upload a directory into them, run one folder, zips keep the tree, move, rename, delete', async ({ page, request }) => {
  await page.goto('/')
  const dsId = await newDataset(page, 'e2e 폴더')

  // Two folders at the top (a new folder goes into the selected one, which is still the top).
  await newFolder(page, '원본')
  await newFolder(page, '증강1')
  await expect(count(page, '원본')).toHaveText('0')
  await expect(count(page, '')).toHaveText('0')

  // A whole directory: the picked directory holds only folders, so its 원본/ and 증강1/ (with
  // 증강1/sub/) fill the dataset's folders. .DS_Store, Thumbs.db and ._a.png are never sent.
  const posted: string[] = []
  page.on('request', (r) => {
    if (new URL(r.url()).pathname === '/api/media' && r.method() === 'POST') posted.push(r.url())
  })
  await page.locator('.rail input[webkitdirectory]').setInputFiles(mediaFile('folders/업로드'))
  await expect(count(page, '')).toHaveText('3', { timeout: 60_000 })
  await expect(count(page, '원본')).toHaveText('1')
  await expect(count(page, '증강1')).toHaveText('2')
  await expect(count(page, '증강1/sub')).toHaveText('1')
  expect(posted).toHaveLength(3)

  // The same directory again: every file is already in the dataset, stays where it is, and is reported.
  await page.locator('.rail input[webkitdirectory]').setInputFiles(mediaFile('folders/업로드'))
  await expect(page.locator('.toast').filter({ hasText: '이미 데이터셋에 있음, 3개' })).toBeVisible({ timeout: 60_000 })
  await expect(count(page, '')).toHaveText('3')
  await expect(count(page, '증강1')).toHaveText('2')

  // Selecting 증강1 lists it with its subfolder; the item below it shows where it is.
  await folderRow(page, '증강1').locator('.ft-main').click()
  await expect(folderRow(page, '증강1').locator('.ft-main')).toHaveAttribute('aria-current', 'true')
  await expect(page.locator('.media-row')).toHaveCount(2)
  await expect(page.locator('.media-row', { hasText: 'c.png' }).locator('.media-folder')).toHaveText('sub/')
  await expect(page.locator('.media-row', { hasText: 'b.png' }).locator('.media-folder')).toHaveCount(0)

  // Run the current model on 증강1 only; the batch says its scope.
  const run = page.getByRole('button', { name: '지금 모델로 이 폴더 실행' })
  await expect(run).toHaveAttribute('title', /증강1/)
  await run.click()
  const batch = page.locator('.ds-batch-row').first()
  await expect(batch.locator('.ds-scope')).toHaveText('증강1')
  await expect(batch.locator('.ds-state')).toHaveText('완료', { timeout: 60_000 })
  await expect(batch.locator('.ds-item')).toHaveCount(2)
  const results = await request.get((await batch.getByRole('link', { name: '전체 zip' }).getAttribute('href'))!)
  expect(results.status()).toBe(200)
  // Paths stay relative to the dataset, so zips of different folders merge.
  expect(zipNames(await results.body()).sort()).toEqual(['results.csv', 'results.json', '증강1/b_result.png', '증강1/sub/c_result.png'])

  // The originals zip keeps every folder.
  const originals = await request.get((await page.getByRole('link', { name: '원본 zip' }).getAttribute('href'))!)
  expect(originals.status()).toBe(200)
  expect(zipNames(await originals.body()).sort()).toEqual(['원본/a.png', '증강1/b.png', '증강1/sub/c.png'])

  // Move a.png from 원본 into 증강1/sub with the selection mode.
  await folderRow(page, '원본').locator('.ft-main').click()
  await expect(page.locator('.media-row')).toHaveCount(1)
  await page.getByRole('button', { name: '선택' }).click()
  await page.locator('.media-row', { hasText: 'a.png' }).locator('.media-main').click()
  await page.getByLabel('선택한 미디어를 옮길 폴더').selectOption({ label: '증강1/sub' })
  await expect(count(page, '원본')).toHaveText('0')
  await expect(count(page, '증강1/sub')).toHaveText('2')
  await page.getByRole('button', { name: '완료', exact: true }).click()

  // Rename 증강1 → 증강A: the subfolder and the media go with it, the selection and the batch follow.
  await folderRow(page, '증강1').locator('.ft-main').click()
  await folderRow(page, '증강1').getByRole('button', { name: '폴더 증강1 이름 바꾸기' }).click()
  await page.getByLabel('폴더 이름', { exact: true }).fill('증강A')
  await page.getByLabel('폴더 이름', { exact: true }).press('Enter')
  await expect(folderRow(page, '증강1')).toHaveCount(0)
  await expect(count(page, '증강A')).toHaveText('3')
  await expect(count(page, '증강A/sub')).toHaveText('2')
  await expect(folderRow(page, '증강A').locator('.ft-main')).toHaveAttribute('aria-current', 'true')
  await expect(page.locator('.media-row')).toHaveCount(3)
  await expect(batch.locator('.ds-scope')).toHaveText('증강A')

  // Delete 증강A (two clicks): its media leave this dataset only; files and run records stay.
  const batches = await (await request.get(`/api/datasets/${dsId}/batches`)).json()
  const detail = await (await request.get(`/api/batches/${batches[0].id}`)).json()
  await folderRow(page, '증강A').getByRole('button', { name: /폴더 증강A 삭제/ }).click()
  await folderRow(page, '증강A').getByRole('button', { name: '폴더 삭제' }).click()
  await expect(folderRow(page, '증강A')).toHaveCount(0)
  await expect(count(page, '')).toHaveText('0')
  await expect(folderRow(page, '').locator('.ft-main')).toHaveAttribute('aria-current', 'true')
  await expect(page.locator('.rail-empty')).toContainText('비어 있습니다')
  for (const it of detail.items) {
    expect((await request.get(`/api/runs/${it.run_id}`)).status()).toBe(200)
    const m = await (await request.get(`/api/media/${it.media_id}`)).json()
    expect(m.dataset_ids).not.toContain(dsId)
  }
})

test('folders: dropped directories are read recursively; a lone wrapper directory adds no level', async ({ page }) => {
  await page.goto('/')
  await newDataset(page, 'e2e 끌어 놓기')

  // Fake FileSystemEntry trees (a real drop cannot be scripted): readEntries returns in batches and
  // an empty batch at the end, like Chromium.
  const drop = (spec: Record<string, string[]>) =>
    page.evaluate(async (tree) => {
      const png = async (seed: number) => {
        const c = new OffscreenCanvas(64, 48)
        const g = c.getContext('2d')!
        g.fillStyle = `rgb(${seed % 256}, ${(seed >> 8) % 256}, ${(seed >> 16) % 256})`
        g.fillRect(0, 0, 64, 48)
        return c.convertToBlob({ type: 'image/png' })
      }
      type Entry = Record<string, unknown>
      const fileEntry = (name: string, fullPath: string, blob: Blob): Entry => ({
        isFile: true, isDirectory: false, name, fullPath,
        file: (ok: (f: File) => void) => ok(new File([blob], name, { type: 'image/png' })),
      })
      const dirEntry = (name: string, fullPath: string, children: Entry[]): Entry => ({
        isFile: false, isDirectory: true, name, fullPath,
        createReader: () => {
          let at = 0
          return { readEntries: (ok: (e: Entry[]) => void) => { ok(children.slice(at, at + 1)); at += 1 } }
        },
      })
      let seed = Date.now() % 16_000_000
      // tree: {"top/dir": ["a.png", ...]} -> nested entries
      const build = async (prefix: string): Promise<Entry[]> => {
        const out: Entry[] = []
        const here = Object.keys(tree).filter((k) => k === prefix)
        for (const k of here) for (const f of tree[k]) out.push(fileEntry(f, `/${k}/${f}`, f.startsWith('.') || f.startsWith('._') ? new Blob(['junk']) : await png(seed++)))
        const kids = new Set(Object.keys(tree).filter((k) => k.startsWith(prefix ? prefix + '/' : '')).map((k) => k.slice(prefix ? prefix.length + 1 : 0).split('/')[0]).filter(Boolean))
        for (const kid of kids) {
          const path = prefix ? `${prefix}/${kid}` : kid
          out.push(dirEntry(kid, `/${path}`, await build(path)))
        }
        return out
      }
      const top = await build('')
      const dt = {
        types: ['Files'],
        files: [] as File[],
        items: top.map((e) => ({ kind: 'file', webkitGetAsEntry: () => e, getAsFile: () => null })),
        dropEffect: 'copy',
      }
      for (const type of ['dragenter', 'drop']) {
        const ev = new Event(type, { bubbles: true, cancelable: true })
        Object.defineProperty(ev, 'dataTransfer', { value: dt })
        document.body.dispatchEvent(ev)
      }
    }, spec)

  // Two directories dropped together keep their names.
  await drop({ 증강2: ['d.png', '.DS_Store'], 증강3: ['e.png'] })
  await expect(count(page, '증강2')).toHaveText('1', { timeout: 60_000 })
  await expect(count(page, '증강3')).toHaveText('1')
  // One directory that holds only directories is a wrapper: its folders land here.
  await drop({ 'wrap/증강4': ['f.png'] })
  await expect(count(page, '증강4')).toHaveText('1', { timeout: 60_000 })
  await expect(folderRow(page, 'wrap')).toHaveCount(0)
  // One directory with files of its own becomes a folder.
  await drop({ 증강5: ['g.png', '._g.png'] })
  await expect(count(page, '증강5')).toHaveText('1', { timeout: 60_000 })
  await expect(count(page, '')).toHaveText('4')
})
