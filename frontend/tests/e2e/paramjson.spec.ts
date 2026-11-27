// The JSON dialog: it opens with the model's current values; a pasted answer (switching the model,
// with prose, a fence and a trailing comma around it) is checked at once and the dialog lists what
// applying would change; "n개 적용" applies exactly that, then the button reads "적용됨" and stays
// disabled; out-of-range and unknown keys are reported; a model change refills the box.
import { expect, test } from '@playwright/test'
import { mediaFile, uploadViaRail } from './helpers'

test('param JSON: describe, preview changes, apply once, report', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  await page.goto('/')
  await uploadViaRail(page, mediaFile('scene_a.jpg'))
  await expect(page.locator('.mode-badge.is-preview')).toBeVisible()

  await page.getByRole('button', { name: 'JSON' }).click()
  const dialog = page.getByRole('dialog', { name: /파라미터 JSON/ })
  const box = dialog.getByLabel('파라미터 JSON')
  const state = dialog.locator('#pj-state')
  const applyBtn = dialog.locator('.pj-apply')
  await expect(dialog).toBeVisible()
  // Opens with the current values: nothing to apply.
  await expect(box).toHaveValue(/"model": "cv\.hsv_ranges"/)
  await expect(box).toHaveValue(/"red_s": \[\s*90,\s*255\s*\]/)
  await expect(state).toContainText('지금 값과 같습니다 — 바뀔 값이 없습니다')
  await expect(applyBtn).toHaveText('바뀔 값 없음')
  await expect(applyBtn).toBeDisabled()
  await dialog.getByRole('button', { name: 'JSON 복사' }).click()
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain('"wood_yields": true')

  const answer = [
    '판 차이 방식으로 바꿔 보세요.',
    '```json',
    '{"model": "cv.board_diff_lab", "params": {"t_chroma_hi": 25, "blur_ksize": 99, "foo": 1,}}',
    '```',
  ].join('\n')
  await box.fill(answer)
  // Checked before applying: what changes, what was clamped, what is unknown.
  await expect(state).toContainText('모델이 "판 차이 + Lab 최근접색"으로 바뀌고, 적용하면 값 2개가 바뀝니다')
  await expect(state).toContainText('t_chroma_hi')
  await expect(state).toContainText('18.0 → 25.0')
  await expect(state).toContainText('blur_ksize 99 → 31')
  await expect(state).toContainText('foo')
  await expect(applyBtn).toHaveText('모델 바꾸고 2개 적용')
  const preview = page.waitForResponse((r) => {
    if (!r.url().endsWith('/api/preview') || r.status() !== 200) return false
    const body = JSON.parse(r.request().postData() ?? '{}')
    return body.model_key === 'cv.board_diff_lab' && body.params?.t_chroma_hi === 25
  })
  await applyBtn.click()
  const sent = JSON.parse((await preview).request().postData() ?? '{}')
  expect(sent.params.blur_ksize).toBe(31)
  // Applied: the button can not be pressed again and says so.
  await expect(state).toContainText('적용했습니다 — 지금 값과 같습니다')
  await expect(applyBtn).toHaveText('적용됨')
  await expect(applyBtn).toBeDisabled()
  await expect(dialog).toContainText('· 판 차이 + Lab 최근접색')

  // The description for the assistant carries the new current values.
  await dialog.getByRole('button', { name: /ChatGPT용 설명 복사/ }).click()
  const text = await page.evaluate(() => navigator.clipboard.readText())
  expect(text).toContain('키 cv.board_diff_lab')
  expect(text).toContain('"name":"t_chroma_hi","group":"색 기준","label":"전경 씨앗 채도","type":"float","current":25')

  await dialog.getByRole('button', { name: '닫기' }).last().click()
  await expect(dialog).toBeHidden()
  await expect(page.locator('.model-current')).toContainText('판 차이 + Lab 최근접색')
  await expect(page.locator('[data-param=t_chroma_hi] .num-input')).toHaveValue('25.0')

  // Reopening shows the current values again (the applied text is not kept as a draft).
  await page.getByRole('button', { name: 'JSON' }).click()
  await expect(box).toHaveValue(/"model": "cv\.board_diff_lab"/)
  await expect(box).toHaveValue(/"t_chroma_hi": 25,/)
  await dialog.getByRole('button', { name: '기본값으로' }).click()
  await expect(box).toHaveValue(/"t_chroma_hi": 18,/)
  await expect(state).toContainText('적용하면 값 2개가 바뀝니다')
  // A parse error is shown and nothing can be applied.
  await box.fill('t_chroma_hi = 30')
  await expect(state.locator('.error-text')).toContainText('JSON으로 읽을 수 없습니다')
  await expect(applyBtn).toBeDisabled()
  await page.keyboard.press('Escape')
  await expect(page.locator('[data-param=t_chroma_hi] .num-input')).toHaveValue('25.0')

  // An unapplied draft survives closing; a model change in the picker replaces it.
  await page.getByRole('button', { name: 'JSON' }).click()
  await expect(box).toHaveValue('t_chroma_hi = 30')
  await box.fill('{"model": "cv.board_diff_lab", "params": {"t_chroma_hi": 20}}')
  await page.keyboard.press('Escape')
  await page.locator('.model-current').click()
  await page.getByRole('option', { name: /HSV 색 범위/ }).click()
  await expect(page.locator('.model-current')).toContainText('HSV 색 범위')
  await page.getByRole('button', { name: 'JSON' }).click()
  await expect(dialog).toContainText('· HSV 색 범위')
  await expect(box).toHaveValue(/"model": "cv\.hsv_ranges"/)
  await expect(state).toContainText('지금 값과 같습니다')
})
