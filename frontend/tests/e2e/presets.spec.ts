// Two parameter sets of one model kept as presets through the JSON dialog (the user's flow):
// each preset keeps its own values; choosing one shows those values in the controls, in the
// preset select and in the JSON box, where "적용" is disabled so the other set cannot be
// re-applied by accident; a moved slider marks the preset "(수정됨)", "되돌리기" restores it, and
// overwriting it takes a labelled second click.
import { expect, test, type Page } from '@playwright/test'
import { mediaFile, uploadViaRail } from './helpers'

/** canny of this test's presets (other specs share the data folder). */
function canny(list: { name: string; params: { canny: number[] } }[]) {
  return Object.fromEntries(list.filter((p) => p.name.startsWith('옵션')).map((p) => [p.name, p.params.canny]))
}

const dialog = (page: Page) => page.getByRole('dialog', { name: /파라미터 JSON/ })

async function applyJson(page: Page, json: string) {
  await page.getByRole('button', { name: 'JSON' }).click()
  const d = dialog(page)
  await d.getByLabel('파라미터 JSON').fill(json)
  const btn = d.locator('.pj-apply')
  await expect(btn).toBeEnabled()
  await btn.click()
  await expect(btn).toHaveText('적용됨')
  await d.getByRole('button', { name: '닫기' }).last().click()
}

async function savePreset(page: Page, name: string) {
  await page.getByRole('button', { name: '새 프리셋으로 저장' }).click()
  await page.getByLabel('새 프리셋 이름').fill(name)
  await page.getByRole('button', { name: '저장', exact: true }).click()
  await expect(page.getByLabel('프리셋', { exact: true }).locator('option:checked')).toHaveText(name)
}

test('presets made from two JSONs keep their own values', async ({ page, request }) => {
  await page.goto('/')
  await uploadViaRail(page, mediaFile('scene_b.jpg'))
  const lo = page.locator('[data-param=canny] .num-input').first()
  const select = page.getByLabel('프리셋', { exact: true })

  await applyJson(page, '{"model": "cv.edge_quads", "params": {"canny": [20, 60]}}')
  await expect(lo).toHaveValue('20')
  await savePreset(page, '옵션A')
  await applyJson(page, '{"model": "cv.edge_quads", "params": {"canny": [40, 120]}}')
  await expect(lo).toHaveValue('40')
  await expect(select.locator('option:checked')).toHaveText('옵션A (수정됨)')
  await savePreset(page, '옵션B')

  const stored = await (await request.get('/api/presets?model_key=cv.edge_quads')).json()
  expect(canny(stored)).toEqual({ 옵션A: [20, 60], 옵션B: [40, 120] })

  // Choosing A shows A everywhere, and the dialog has nothing to apply.
  await select.selectOption({ label: '옵션A' })
  await expect(lo).toHaveValue('20')
  await page.getByRole('button', { name: 'JSON' }).click()
  const d = dialog(page)
  await expect(d).toContainText('프리셋 "옵션A"와 같음')
  await expect(d.getByLabel('파라미터 JSON')).toHaveValue(/"canny": \[20,60\]/)
  await expect(d.locator('.pj-apply')).toBeDisabled()
  await d.getByRole('button', { name: '닫기' }).last().click()

  // Choosing B, then A again, never mixes them.
  await select.selectOption({ label: '옵션B' })
  await expect(lo).toHaveValue('40')
  await select.selectOption({ label: '옵션A' })
  await expect(lo).toHaveValue('20')

  // A slider move marks A modified; saving over A keeps B untouched.
  await lo.fill('25')
  await lo.press('Enter')
  await expect(select.locator('option:checked')).toHaveText('옵션A (수정됨)')
  // "되돌리기" restores A; overwriting needs a labelled second click.
  await page.getByRole('button', { name: '되돌리기' }).click()
  await expect(lo).toHaveValue('20')
  await expect(select.locator('option:checked')).toHaveText('옵션A')
  await lo.fill('25')
  await lo.press('Enter')
  const overwrite = page.getByRole('button', { name: '"옵션A"에 덮어쓰기' })
  await overwrite.click()
  const sure = page.getByRole('button', { name: '정말 덮어쓰기' })
  await expect(sure).toBeVisible()
  const before = await (await request.get('/api/presets?model_key=cv.edge_quads')).json()
  expect(before.find((p: { name: string }) => p.name === '옵션A').params.canny, 'one click overwrites nothing').toEqual([20, 60])
  await sure.click()
  await expect(select.locator('option:checked')).toHaveText('옵션A')
  const after = await (await request.get('/api/presets?model_key=cv.edge_quads')).json()
  expect(canny(after)).toEqual({ 옵션A: [25, 60], 옵션B: [40, 120] })
})
