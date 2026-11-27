// Datasets in the rail: create one, upload into it, put an existing media in with the selection
// mode, run the current model on the whole dataset, then download the results (zip and one item)
// and the originals.
import { expect, test } from '@playwright/test'
import { mediaFile, uploadViaRail } from './helpers'

test('datasets: group, upload into, select, run all, download', async ({ page, request }) => {
  await page.goto('/')
  const outside = await uploadViaRail(page, mediaFile('datasets/ds_a.jpg'))

  // A new dataset is selected at once and starts empty.
  await page.getByRole('button', { name: '새로' }).click()
  await page.getByLabel('데이터셋 이름').fill('e2e 데이터셋')
  await page.getByRole('button', { name: '만들기' }).click()
  const picker = page.getByLabel('데이터셋', { exact: true })
  await expect(picker.locator('option:checked')).toHaveText(/e2e 데이터셋 · 0/)
  await expect(page.locator('.rail-empty')).toContainText('비어 있습니다')

  // Uploading while it is selected puts the file into it.
  const inside = await uploadViaRail(page, mediaFile('datasets/ds_b.jpg'))
  await expect(page.locator('.media-row')).toHaveCount(1)
  await expect(picker.locator('option:checked')).toHaveText(/e2e 데이터셋 · 1/)

  // Selection mode on all media: tick the other image and put it in.
  await picker.selectOption({ label: '전체 미디어' })
  await page.getByRole('button', { name: '선택' }).click()
  await page.locator('.media-row', { hasText: 'ds_a.jpg' }).locator('.media-main').click()
  await expect(page.getByRole('region', { name: '선택한 미디어' })).toContainText('1개 선택')
  await page.getByLabel('선택한 미디어를 넣을 데이터셋').selectOption({ label: 'e2e 데이터셋' })
  // Selection mode stays on (the same files may go into another dataset); 완료 closes it.
  await expect(page.getByRole('region', { name: '선택한 미디어' })).toContainText('1개 선택')
  await page.getByRole('button', { name: '완료' }).click()
  await expect(page.getByRole('region', { name: '선택한 미디어' })).toBeHidden()
  const ds = (await (await request.get('/api/datasets')).json()).find((d: { name: string }) => d.name === 'e2e 데이터셋')
  await picker.selectOption(String(ds.id))
  await expect(page.locator('.media-row')).toHaveCount(2)
  const ids = (await (await request.get(`/api/media?dataset_id=${ds.id}`)).json()).map((m: { id: number }) => m.id)
  expect(ids.sort()).toEqual([outside, inside].sort())

  // Run the current model on all of it; both items finish and can be downloaded.
  await page.getByRole('button', { name: '지금 모델로 전체 실행' }).click()
  const row = page.locator('.ds-batch-row').first()
  await expect(row.locator('.ds-state')).toHaveText('완료', { timeout: 60_000 })
  await expect(row.getByRole('link', { name: '전체 zip' })).toBeVisible()
  await expect(row.locator('.ds-item')).toHaveCount(2)
  const itemLink = row.getByRole('link', { name: 'ds_a.jpg 결과 받기' })
  const itemRes = await request.get((await itemLink.getAttribute('href'))!)
  expect(itemRes.status()).toBe(200)
  expect(itemRes.headers()['content-disposition']).toContain('ds_a_result.png')

  const zipRes = await request.get((await row.getByRole('link', { name: '전체 zip' }).getAttribute('href'))!)
  expect(zipRes.status()).toBe(200)
  const zipText = (await zipRes.body()).toString('latin1')
  for (const name of ['ds_a_result.png', 'ds_b_result.png', 'results.json', 'results.csv']) expect(zipText).toContain(name)

  const originals = await request.get((await page.getByRole('link', { name: '원본 zip' }).getAttribute('href'))!)
  expect(originals.status()).toBe(200)
  expect((await originals.body()).toString('latin1')).toContain('ds_b.jpg')

  // Clicking an item opens its result in the workbench.
  await row.getByRole('button', { name: /ds_b\.jpg/ }).click()
  await expect(page).toHaveURL(new RegExp(`media=${inside}.*run=\\d+`))
  await expect(page.locator('.mode-badge.is-run')).toContainText('기록 #')
})
