// Ground truth and scoring, the user's flows: upload a GT JSON into a dataset and confirm the one
// link that only a file name suggests; run a batch and read it in 최종 순위 with the values the
// labels were built to give (tests/e2e/make_media.py write_gt_set: hsv_ranges' own output with six
// known changes); a batch that waits or was cancelled stays under 중간 결과; τc re-scores; the CSVs;
// the viewer's 정답 layer; the workbench's live score; and an old server whose health lacks them.
import { expect, test, type Page } from '@playwright/test'
import { readFileSync } from 'node:fs'
import {
  GT_IMAGES,
  apiBatch,
  apiDataset,
  apiScoredDataset,
  apiUpload,
  expected,
  gtCubeCounts,
  mediaFile,
} from './helpers'

const CUBES_HEADER =
  'batch,status,image,sha256,variant,source_sha256,layout_group,row,gt_id,gt_color,valid_geometry,pred_color,pred_x,pred_y,' +
  'match_distance_px,center_err_px,angle_err_deg,vertex_rms_px,passed,region,image_side,near_contact,weak_boundary,run_failed,' +
  'tau_center_px,tau_angle_deg,tau_vertex_px,gate_px'
const SUMMARY_HEADER = 'batch,status,metric,label,unit,value,numerator,denominator,n,median,p95,mean,tau_center_px,tau_angle_deg,tau_vertex_px,gate_px'

const pct = ([n, d]: [number, number]) => `${((n / d) * 100).toFixed(1)}%`
const nd = ([n, d]: [number, number]) => `${n.toLocaleString('en-US')}/${d.toLocaleString('en-US')}`

const ranking = (page: Page) => page.getByRole('region', { name: '최종 순위' })
const interim = (page: Page) => page.getByRole('region', { name: '중간 결과 · 비교 제외' })
const cell = (page: Page, batchId: number, metric: string) => ranking(page).locator(`tr[data-batch="${batchId}"] td[data-metric="${metric}"]`)

async function openScore(page: Page, datasetId: number, gtSetId: number) {
  await page.goto(`/score?dataset=${datasetId}&gt=${gtSetId}`)
  await expect(page.getByRole('heading', { name: '채점' })).toBeVisible()
}

test('GT row: upload a GT JSON, see the link count; a same-name file is only a candidate until confirmed', async ({ page, request }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '새로' }).click()
  await page.getByLabel('데이터셋 이름').fill('e2e 정답 연결')
  await page.getByRole('button', { name: '만들기' }).click()
  await expect(page.getByLabel('데이터셋', { exact: true }).locator('option:checked')).toHaveText(/e2e 정답 연결 · 0/)
  const files = [...GT_IMAGES, 'gt_extra.jpg'].map((f) => mediaFile(`gt/${f}`))
  await page.locator('.rail-head input[type=file]:not([webkitdirectory])').setInputFiles(files)
  await expect(page.locator('.ft-row[data-path=""] .ft-count')).toHaveText('10', { timeout: 90_000 })

  const row = page.getByRole('region', { name: '정답셋' })
  await expect(row).toContainText('정답셋 없음')
  const made = page.waitForResponse((r) => /\/api\/datasets\/\d+\/gt$/.test(new URL(r.url()).pathname) && r.request().method() === 'POST')
  await row.locator('input[type=file]').setInputFiles(mediaFile('gt/labels_link.json'))
  const gid = (await (await made).json()).id as number
  await expect(row.getByLabel('정답셋 선택').locator('option:checked')).toHaveText(/labels_link\.json/)
  await expect(row.locator('.gt-links')).toContainText('연결 9/10')
  const open = row.getByRole('button', { name: '연결 안 됨 1' })
  await expect(open).toBeVisible()

  // The file of the same name is a candidate only: nothing is linked by looking at the list.
  await open.click()
  const dialog = page.getByRole('dialog', { name: /정답 연결 확인/ })
  const gl = dialog.locator('.gl-row', { hasText: 'gt_extra.jpg' })
  await expect(gl).toContainText('후보')
  await expect(gl.getByLabel('gt_extra.jpg 후보')).toContainText('gt_extra.jpg')
  let detail = await (await request.get(`/api/gt/${gid}`)).json()
  expect(detail.links.linked_confirmed).toBe(0)
  expect(detail.unlinked.map((u: { filename: string }) => u.filename)).toEqual(['gt_extra.jpg'])

  // Confirmed on its row: linked as "confirmed"; then removed again; then all candidates at once.
  await gl.getByRole('button', { name: '연결', exact: true }).click()
  await expect(row.locator('.gt-links')).toContainText('연결 10/10')
  await expect(dialog.locator('.gl-confirmed')).toContainText('gt_extra.jpg')
  detail = await (await request.get(`/api/gt/${gid}`)).json()
  expect(detail.links.linked_confirmed).toBe(1)
  await dialog.locator('.gl-confirmed').getByRole('button', { name: /연결 끊기/ }).click()
  await expect(row.locator('.gt-links')).toContainText('연결 9/10')
  await dialog.getByRole('button', { name: '후보 모두 연결' }).click()
  await expect(row.locator('.gt-links')).toContainText('연결 10/10')
  await dialog.getByRole('button', { name: '닫기' }).click()
  await expect(dialog).toBeHidden()

  // The set's name is editable; two clicks sort the linked media into variant folders.
  await row.getByRole('button', { name: '정답셋 이름 바꾸기' }).click()
  await page.getByLabel('정답셋 이름').fill('e2e 정답 10장')
  await page.getByLabel('정답셋 이름').press('Enter')
  await expect(row.getByLabel('정답셋 선택').locator('option:checked')).toHaveText(/e2e 정답 10장/)
  await row.getByRole('button', { name: '증강 종류별 폴더로 정리' }).click()
  await row.getByRole('button', { name: /폴더로 옮기기/ }).click()
  await expect(page.locator('.ft-row[data-path="cool"] .ft-count')).toHaveText('3')
  await expect(page.locator('.ft-row[data-path="original"] .ft-count')).toHaveText('4')
})

test('score page: a finished batch from the rail ranks with the values the labels were built for', async ({ page, request }) => {
  const exp = expected()
  const { datasetId, gtSetId } = await apiScoredDataset(request, 'e2e 채점 순위')
  await page.goto('/')
  await page.getByLabel('데이터셋', { exact: true }).selectOption(String(datasetId))
  await page.getByRole('button', { name: '지금 모델로 전체 실행' }).click()
  const batchRow = page.locator('.ds-batch-row').first()
  await expect(batchRow.locator('.ds-state')).toHaveText('완료', { timeout: 60_000 })
  // The rail's score chip, for the dataset's GT set with the shared criteria.
  await expect(batchRow.locator('.ds-batch-score')).toContainText(`${pct(exp.tolerance)} · ${pct(exp.image_pass)}`, { timeout: 30_000 })
  await expect(batchRow.locator('.ds-batch-score')).toContainText('완료')
  const batchId = Number((await batchRow.locator('.ds-batch-name').textContent())!.match(/#(\d+)/)![1])

  await page.getByRole('link', { name: '채점', exact: true }).click()
  await expect(page).toHaveURL(/\/score/)
  await expect(page.getByRole('heading', { name: '채점' })).toBeVisible()
  await page.getByLabel('데이터셋', { exact: true }).selectOption(String(datasetId))
  await page.getByLabel('정답셋', { exact: true }).selectOption(String(gtSetId))
  await expect(page.getByText('AI 보조로 만든 근사 이미지 라벨')).toBeVisible()
  await expect(cell(page, batchId, 'tolerance_rate')).toContainText(nd(exp.tolerance), { timeout: 30_000 })
  await expect(cell(page, batchId, 'tolerance_rate')).toContainText(pct(exp.tolerance))
  await expect(cell(page, batchId, 'image_pass_rate')).toContainText(nd(exp.image_pass))
  await expect(cell(page, batchId, 'recall')).toContainText(nd(exp.recall))
  await expect(cell(page, batchId, 'precision')).toContainText(nd(exp.precision))
  await expect(cell(page, batchId, 'color_accuracy')).toContainText(nd(exp.color_accuracy))
  // Centre errors over the matched valid-geometry labels (all but the missed one): every label but
  // the 3 px and 8 px ones is the output itself.
  await expect(cell(page, batchId, 'center_err_median')).toContainText('0.00 px')
  await expect(cell(page, batchId, 'center_err_median')).toContainText(`n ${exp.n_valid - 1}`)
  await expect(ranking(page).locator(`tr[data-batch="${batchId}"]`)).toContainText('기준')
  await expect(interim(page).locator(`tr[data-batch="${batchId}"]`)).toHaveCount(0)

  // Per image: the failures first; per source: originals by variants.
  const images = page.getByRole('region', { name: '이미지별 결과' })
  const firstNames = await images.locator('tbody tr .im-name').evaluateAll((els) => els.slice(0, 4).map((e) => e.textContent?.trim()))
  expect(firstNames.sort()).toEqual([...exp.failing].sort())
  const sources = page.getByRole('region', { name: '원본별 결과' })
  await expect(sources.locator('tbody tr')).toHaveCount(3)
  await expect(sources.locator('tbody tr', { hasText: 's2_original.jpg' }).locator('td[data-variant="original"]')).toHaveAttribute('data-pass', 'false')
  await expect(sources.locator('tbody tr', { hasText: 's3_original.jpg' }).locator('td[data-variant="original"]')).toHaveAttribute('data-pass', 'true')

  // Slices carry their counts (the cool images: every label valid and passing); the augmentation
  // effect a verdict.
  const counts = gtCubeCounts()
  const cool = Object.entries(counts).filter(([f]) => f.endsWith('_cool.jpg')).reduce((a, [, n]) => a + n, 0)
  const slicesRegion = page.getByRole('region', { name: '조각 비교' })
  await expect(slicesRegion.locator('tr[data-slice="variant:cool"]')).toContainText(`${cool}/${cool}`)
  await expect(slicesRegion.locator('tr[data-slice="variant:cool"]')).toContainText('이미지 3')
  const aug = page.getByRole('region', { name: '증강 영향' })
  await expect(aug.locator('tr[data-variant="dark_ev_m0p5"]')).toContainText(/높음|낮음|차이 불분명/)
})

test('interim: a batch waiting behind another run, then cancelled, stays under 중간 결과 and is never ranked', async ({ page, request }) => {
  const { datasetId, gtSetId } = await apiScoredDataset(request, 'e2e 채점 중간')
  // Keep the worker busy with a long video so the batch's runs stay queued.
  const slow = await apiUpload(request, mediaFile('gt/slow.mp4'))
  const blocker = (await (await request.post('/api/runs', { data: { media_id: slow, model_key: 'cv.hsv_ranges', params: {} } })).json()).id
  try {
    await expect.poll(async () => (await (await request.get(`/api/runs/${blocker}`)).json()).status, { timeout: 30_000 }).toBe('running')
    const batchId = await apiBatch(request, datasetId, {}, false)
    await openScore(page, datasetId, gtSetId)
    const row = interim(page).locator(`tr[data-batch="${batchId}"]`)
    await expect(row).toContainText('대기 9', { timeout: 30_000 })
    await expect(row).toContainText('중간 점수')
    await expect(ranking(page).locator(`tr[data-batch="${batchId}"]`)).toHaveCount(0)
    await expect(ranking(page)).toContainText('최종 순위에 넣을 배치가 없습니다')

    await request.post(`/api/batches/${batchId}/cancel`)
    await expect(row).toContainText('취소 9', { timeout: 30_000 })
    await expect(ranking(page).locator(`tr[data-batch="${batchId}"]`)).toHaveCount(0)
  } finally {
    await request.post(`/api/runs/${blocker}/cancel`)
  }
})

test.describe('one scored batch', () => {
  let ids: Awaited<ReturnType<typeof apiScoredDataset>> & { batchId: number }

  test.beforeAll(async ({ request }) => {
    // A worker restarted after a failure runs this again: the name must not clash.
    const set = await apiScoredDataset(request, `e2e 채점 기준 ${Date.now().toString(36).slice(-5)}`)
    ids = { ...set, batchId: await apiBatch(request, set.datasetId) }
  })

  test('criteria: changing τc re-scores and the value changes; the provisional label stays', async ({ page }) => {
    const exp = expected()
    await openScore(page, ids.datasetId, ids.gtSetId)
    await expect(page.getByText('잠정 기준 — 로봇 실제 허용 오차 아님').first()).toBeVisible()
    await expect(cell(page, ids.batchId, 'tolerance_rate')).toContainText(nd(exp.tolerance), { timeout: 30_000 })

    const scored = page.waitForRequest((r) => new URL(r.url()).pathname === '/api/score' && r.postDataJSON()?.criteria?.center_px === 2)
    await page.getByLabel('중심 허용 오차 τc (px)').fill('2')
    await scored
    await expect(cell(page, ids.batchId, 'tolerance_rate')).toContainText(nd(exp.tolerance_tc2))
    await expect(cell(page, ids.batchId, 'image_pass_rate')).toContainText(nd(exp.image_pass_tc2))

    // One criteria state for every screen: the rail's chip uses τc 2 as well.
    await page.goto('/')
    await page.getByLabel('데이터셋', { exact: true }).selectOption(String(ids.datasetId))
    await expect(page.locator('.ds-batch-row').first().locator('.ds-batch-score')).toContainText(pct(exp.tolerance_tc2), { timeout: 30_000 })

    await openScore(page, ids.datasetId, ids.gtSetId)
    await expect(page.getByLabel('중심 허용 오차 τc (px)')).toHaveValue('2')
    await page.getByRole('button', { name: '기본값으로' }).click()
    await expect(page.getByLabel('중심 허용 오차 τc (px)')).toHaveValue('5')
    await expect(cell(page, ids.batchId, 'tolerance_rate')).toContainText(nd(exp.tolerance))
  })

  test('csv: cubes.csv and summary.csv carry their header and the criteria used', async ({ page }) => {
    await openScore(page, ids.datasetId, ids.gtSetId)
    await page.getByLabel('매칭 게이트 g (px)').fill('20')
    await expect(page.getByRole('link', { name: 'cubes.csv 받기' })).toHaveAttribute('href', /gate_px=20/)
    for (const [name, header, rows] of [
      ['cubes.csv 받기', CUBES_HEADER, expected().n_cubes + 1],
      ['summary.csv 받기', SUMMARY_HEADER, 9],
    ] as const) {
      const download = page.waitForEvent('download')
      await page.getByRole('link', { name }).click()
      const file = await (await download).path()
      const text = readFileSync(file, 'utf8')
      expect(text.charCodeAt(0)).toBe(0xfeff)
      const lines = text.slice(1).trim().split('\n')
      expect(lines[0]).toBe(header)
      expect(lines.length - 1).toBe(rows)
      expect(lines[1].split(',').at(-1)).toBe('20.0')
    }
  })

  test('viewer: the 정답 layer draws GT quads, match lines, the missed cube and the false positive', async ({ page }) => {
    const counts = gtCubeCounts()
    await openScore(page, ids.datasetId, ids.gtSetId)
    const images = page.getByRole('region', { name: '이미지별 결과' })
    await images.locator('tbody tr', { hasText: 's2_dark_ev_m0p5.jpg' }).click()
    await expect(page).toHaveURL(new RegExp(`media=${ids.media['s2_dark_ev_m0p5.jpg']}.*run=\\d+`))
    await expect(page.locator('.overlay .ov-gt-quad')).toHaveCount(counts['s2_dark_ev_m0p5.jpg'])
    await expect(page.locator('.overlay .ov-gt-miss')).toHaveCount(1)
    await expect(page.locator('.overlay .ov-gt-link')).toHaveCount(counts['s2_dark_ev_m0p5.jpg'] - 1)
    await expect(page.locator('.gt-card')).toContainText('미검출 1')

    // Off and on again from the layer menu.
    await page.getByRole('button', { name: '오버레이 레이어' }).click()
    await page.getByRole('checkbox', { name: '정답' }).uncheck()
    await expect(page.locator('.overlay .ov-gt-quad')).toHaveCount(0)
    await page.getByRole('checkbox', { name: '정답' }).check()
    await expect(page.locator('.overlay .ov-gt-quad')).toHaveCount(counts['s2_dark_ev_m0p5.jpg'])
    await page.keyboard.press('Escape')

    // The image with the unlabelled detection: one false positive is marked.
    await openScore(page, ids.datasetId, ids.gtSetId)
    await page.getByRole('region', { name: '이미지별 결과' }).locator('tbody tr', { hasText: 's3_cool.jpg' }).click()
    await expect(page.locator('.overlay .ov-gt-fp')).toHaveCount(1)
    await expect(page.locator('.gt-card')).toContainText('오검출 1')
    // Hovering a detection shows its errors against its label (detection 1 is the unlabelled one).
    await page.locator('.result-table tbody tr').nth(1).hover()
    await expect(page.locator('.gt-card')).toContainText('중심')
    await expect(page.locator('.gt-card')).toContainText('꼭짓점 RMS')
  })

  test('workbench: the live score follows the preview for a GT image and is absent for one without', async ({ page }) => {
    await page.goto(`/?media=${ids.media['s1_original.jpg']}`)
    const live = page.getByRole('region', { name: '정답 비교' })
    await expect(live).toContainText('이미지 통과', { timeout: 30_000 })
    await expect(live).toContainText(`정답 ${gtCubeCounts()['s1_original.jpg']}`)
    // A red S gate nothing passes: the red cubes are lost and the panel says so.
    const lo = page.locator('[data-param=red_s] .num-input').first()
    await lo.fill('254')
    await lo.press('Enter')
    await expect(live).toContainText('이미지 미달')
    await expect(live).toContainText('미검출 3')

    const plain = await apiUpload(page.request, mediaFile('scene_a.jpg'))
    await page.goto(`/?media=${plain}`)
    await expect(page.locator('.result-table')).toBeVisible({ timeout: 30_000 })
    await expect(page.getByRole('region', { name: '정답 비교' })).toHaveCount(0)
    await page.getByRole('button', { name: '오버레이 레이어' }).click()
    await expect(page.getByRole('checkbox', { name: '정답' })).toHaveCount(0)
  })

  test('old server: a health without the gt and score features hides them, asks for none, no error toast', async ({ page }) => {
    // A server started before ground truth and scoring existed: its health names folders only.
    await page.route('**/api/health', async (route) => {
      const res = await route.fetch()
      await route.fulfill({ response: res, json: { ...(await res.json()), features: ['folders'] } })
    })
    const asked: string[] = []
    page.on('request', (r) => {
      const path = new URL(r.url()).pathname
      if (/^\/api\/(gt\/|score|datasets\/\d+\/gt|media\/\d+\/gt)/.test(path)) asked.push(`${r.method()} ${path}`)
    })

    await page.goto(`/?media=${ids.media['s1_original.jpg']}`)
    await page.getByLabel('데이터셋', { exact: true }).selectOption(String(ids.datasetId))
    await expect(page.locator('.ft-row[data-path=""] .ft-count')).toHaveText('9')
    await expect(page.locator('.media-row')).toHaveCount(9)
    await expect(page.locator('.ds-batch-row')).toHaveCount(1)
    await expect(page.locator('.result-table')).toBeVisible({ timeout: 30_000 })
    await expect(page.getByRole('region', { name: '정답셋' })).toHaveCount(0)
    await expect(page.getByRole('region', { name: '정답 비교' })).toHaveCount(0)
    await expect(page.locator('.ds-batch-score')).toHaveCount(0)
    await page.getByRole('button', { name: '오버레이 레이어' }).click()
    await expect(page.getByRole('checkbox', { name: '정답' })).toHaveCount(0)
    await page.keyboard.press('Escape')
    await page.locator('.media-row').nth(1).locator('.media-main').click()
    await expect(page.locator('.result-table')).toBeVisible()

    await page.getByRole('link', { name: '채점', exact: true }).click()
    await expect(page.getByText('서버를 다시 켜야')).toBeVisible()
    await page.waitForTimeout(500)
    await expect(page.locator('.toast.error')).toHaveCount(0)
    expect(asked).toEqual([])
  })
})

test('a dataset without a GT set: the rail offers the upload, the 채점 page explains', async ({ page, request }) => {
  const datasetId = await apiDataset(request, 'e2e 정답 없음')
  await apiUpload(request, mediaFile('gt/s1_cool.jpg'), datasetId)
  await page.goto('/')
  await page.getByLabel('데이터셋', { exact: true }).selectOption(String(datasetId))
  await expect(page.getByRole('region', { name: '정답셋' })).toContainText('정답셋 없음')
  await page.goto(`/score?dataset=${datasetId}`)
  await expect(page.getByText('이 데이터셋에는 정답셋이 없습니다')).toBeVisible()
})
