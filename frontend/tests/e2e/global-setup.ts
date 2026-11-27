// Generates the test media once per run (see make_media.py) and passes the folder to the specs.
import { execFileSync } from 'node:child_process'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

export default function globalSetup() {
  const out = mkdtempSync(join(tmpdir(), 'vision-lab-e2e-media-'))
  const frontend = resolve(import.meta.dirname, '../..')
  const salt = String(Date.now() % 65536)
  execFileSync('uv', ['run', '--offline', '--project', '../backend', 'python', 'tests/e2e/make_media.py', out, salt], {
    cwd: frontend,
    stdio: 'inherit',
  })
  process.env.E2E_MEDIA = out
}
