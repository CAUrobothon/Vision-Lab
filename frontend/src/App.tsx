// App shell: top bar with the three sections, the "server unreachable" banner and the routes.
// The Workbench loads eagerly; the other screens are split into their own chunks.
import { lazy, Suspense, useMemo, useRef } from 'react'
import { NavLink, Route, Routes } from 'react-router'
import { RefreshCw, ServerOff } from 'lucide-react'
import { FeaturesContext } from './api/features'
import { useHealth, useSystem } from './api/queries'
import { formatBytes } from './lib/format'
import WorkbenchPage from './workbench/WorkbenchPage'

const RunsPage = lazy(() => import('./runs/RunsPage'))
const ComparePage = lazy(() => import('./compare/ComparePage'))
const ModelsPage = lazy(() => import('./models/ModelsPage'))
const ScorePage = lazy(() => import('./score/ScorePage'))

function TopStatus() {
  const system = useSystem()
  const s = system.data
  if (!s) return null
  const worker = s.worker
  return (
    <div className="topbar-status num">
      <span title={worker.pid ? `작업자 PID ${worker.pid}` : undefined}>
        <i className={`dot ${worker.alive ? 'alive' : 'dead'}`} />
        작업자 {worker.alive ? (worker.busy_run_id ? `기록 #${worker.busy_run_id} 처리 중` : '대기') : '꺼짐'}
      </span>
      <span title="가중치 모델이 쓸 장치">
        {s.mps_available === null ? 'MPS 확인 전' : s.mps_available ? 'MPS' : 'CPU만'}
      </span>
      <span>디스크 여유 {formatBytes(s.storage.free_bytes)}</span>
    </div>
  )
}

function NotFound() {
  return (
    <div className="empty">
      <strong>없는 화면입니다</strong>
      <NavLink to="/">작업대로 가기</NavLink>
    </div>
  )
}

/** The build this page was loaded with is the first one the server reported; a different one later
 *  means the frontend was rebuilt and this tab still runs the old code. */
function useStaleBuild(build: string | null | undefined): boolean {
  const first = useRef<string | null>(null)
  if (build && first.current === null) first.current = build
  return !!build && first.current !== null && build !== first.current
}

export function App() {
  const health = useHealth()
  const down = health.isError
  const stale = useStaleBuild(health.data?.build)
  // What the server has (folders, GT, scoring); kept while it is briefly unreachable.
  const features = useMemo(() => (health.data ? new Set(health.data.features ?? []) : null), [health.data])
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark reg" aria-hidden="true" />
          Vision Lab
        </div>
        <nav className="nav" aria-label="화면">
          <NavLink to="/" end>
            작업대
          </NavLink>
          <NavLink to="/runs">기록</NavLink>
          <NavLink to="/score">채점</NavLink>
          <NavLink to="/models">모델 · 저장공간</NavLink>
        </nav>
        {down ? null : <TopStatus />}
      </header>
      {down ? (
        <div className="banner" role="alert">
          <ServerOff size={16} strokeWidth={1.75} />
          <span>
            서버에 연결할 수 없습니다 — <code>./run.sh</code> 로 서버를 켜세요. 켜지면 자동으로 다시 연결합니다.
          </span>
        </div>
      ) : stale ? (
        <div className="banner banner-info" role="status">
          <RefreshCw size={16} strokeWidth={1.75} />
          <span>화면이 새 버전으로 바뀌었습니다. 이 탭은 아직 예전 화면입니다.</span>
          <button type="button" className="btn btn-sm" onClick={() => window.location.reload()}>
            새로고침
          </button>
        </div>
      ) : (
        <div />
      )}
      <main className="page-host">
        <FeaturesContext.Provider value={features}>
          <Suspense fallback={<p className="route-loading">불러오는 중…</p>}>
            <Routes>
              <Route path="/" element={<WorkbenchPage />} />
              <Route path="/runs" element={<RunsPage />} />
              <Route path="/compare" element={<ComparePage />} />
              <Route path="/models" element={<ModelsPage />} />
              <Route path="/score" element={<ScorePage />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
        </FeaturesContext.Provider>
      </main>
    </div>
  )
}
