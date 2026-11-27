// A native <select> fires no change when the option already shown is chosen again. So after a JSON
// edit moved the values away from preset A, choosing A in the list must be a real change: the
// select may not keep showing A's own option.
import { expect, test } from '@playwright/test'
import { mediaFile, uploadViaRail } from './helpers'

test('after a JSON edit, choosing the same preset again restores it', async ({ page }) => {
  await page.goto('/')
  await uploadViaRail(page, mediaFile('scene_b.jpg'))
  await page.locator('.model-current').click()
  await page.getByRole('option', { name: /윤곽으로 닫힌 사각형/ }).click()
  const select = page.getByLabel('프리셋', { exact: true })
  const lo = page.locator('[data-param=canny] .num-input').first()

  await page.getByRole('button', { name: /새 프리셋으로 저장|다른 이름으로 저장/ }).click()
  await page.getByLabel('새 프리셋 이름').fill('재선택A')
  await page.getByRole('button', { name: '저장', exact: true }).click()
  const idA = await select.inputValue()

  await page.getByRole('button', { name: 'JSON' }).click()
  const dialog = page.getByRole('dialog', { name: /파라미터 JSON/ })
  await dialog.getByLabel('파라미터 JSON').fill('{"params": {"canny": [33, 99]}}')
  await dialog.locator('.pj-apply').click()
  await expect(dialog.locator('.pj-apply')).toHaveText('적용됨')
  await page.keyboard.press('Escape')
  await expect(lo).toHaveValue('33')
  await expect(select.locator('option:checked')).toHaveText('재선택A (수정됨)')
  expect(await select.inputValue(), 'choosing A again must be a change of the select').not.toBe(idA)

  await select.selectOption(idA)
  await expect(lo).toHaveValue('30')
  await expect(select.locator('option:checked')).toHaveText('재선택A')
})
