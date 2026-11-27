// End-to-end checks against the real backend (Chromium via the installed Google Chrome, so no
// browser download is needed). The backend runs on 8012 with a fresh temporary data folder and
// Vite on 5190 forwards /api to it. Running servers on those ports are reused.
//
//   npm run test:e2e                       # all specs
//   SCREENSHOTS=1 npx playwright test screenshots   # also refresh docs/screenshots/
import { defineConfig } from '@playwright/test'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const API_PORT = 8012
const WEB_PORT = 5190
// Pinned through the environment so the worker processes see the same folder.
process.env.LAB_E2E_DATA ??= mkdtempSync(join(tmpdir(), 'vision-lab-e2e-'))

export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 180_000,
  expect: { timeout: 20_000 },
  workers: 1,
  fullyParallel: false,
  reporter: [['list']],
  globalSetup: './tests/e2e/global-setup.ts',
  use: {
    baseURL: `http://127.0.0.1:${WEB_PORT}`,
    browserName: 'chromium',
    channel: process.env.PW_CHANNEL ?? 'chrome',
    viewport: { width: 1440, height: 900 },
    trace: 'retain-on-failure',
  },
  webServer: [
    {
      command: `uv run --offline lab --port ${API_PORT} --data "${process.env.LAB_E2E_DATA}"`,
      cwd: '../backend',
      url: `http://127.0.0.1:${API_PORT}/api/health`,
      reuseExistingServer: true,
      timeout: 90_000,
    },
    {
      command: 'npm run dev',
      url: `http://127.0.0.1:${WEB_PORT}`,
      env: { LAB_API_URL: `http://127.0.0.1:${API_PORT}` },
      reuseExistingServer: true,
      timeout: 30_000,
    },
  ],
})
