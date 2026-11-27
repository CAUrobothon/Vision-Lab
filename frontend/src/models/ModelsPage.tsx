// 모델 · 저장공간 — built-in pipelines, uploaded .pt weights, disk usage with cleanup, worker status.
import { useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { ChevronDown, Cpu, FileUp, RotateCw, Upload } from 'lucide-react'
import { api } from '../api/client'
import { invalidateRuns, keys, queryClient, useModels, useStorage, useSystem } from '../api/queries'
import type { CleanupRequest, ModelInfo, ParamSpec, StorageInfo } from '../api/types'
import { formatBytes, formatDate } from '../lib/format'
import { formatParamValue } from '../lib/params'
import { ConfirmButton, InlineText } from '../ui/controls'
import { useToast } from '../ui/toast'
import './models.css'
import { named } from '../lib/josa'

const TYPE_LABEL: Record<string, string> = {
  int: '정수',
  float: '실수',
  bool: '켜기/끄기',
  select: '선택',
  multiselect: '여러 개 선택',
  range: '범위',
  color: '색',
  roi: '영역',
}
const TASK_LABEL: Record<string, string> = { detect: '검출 (박스)', segment: '분할 (마스크)', obb: '회전 박스' }

function specRange(s: ParamSpec): string {
  if (s.type === 'select' || s.type === 'multiselect') return (s.options ?? []).map((o) => o.label).join(' / ')
  if (s.min !== null && s.max !== null) return `${s.min} – ${s.max}${s.unit ? ' ' + s.unit : ''}`
  return ''
}

function PipelineEntry({ m }: { m: ModelInfo }) {
  const [open, setOpen] = useState(false)
  return (
    <li className="pipe">
      <div className="pipe-head">
        <h3>{m.name}</h3>
        <code className="pipe-key">{m.key}</code>
      </div>
      <p className="pipe-summary">{m.summary}</p>
      {m.description ? <p className="pipe-desc">{m.description}</p> : null}
      <button type="button" className="btn btn-sm btn-quiet pipe-toggle" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
        <ChevronDown size={13} strokeWidth={1.75} className="pg-chevron" /> 파라미터 {m.params.length}개
      </button>
      {open ? (
        <table className="table pipe-params">
          <thead>
            <tr>
              <th>그룹</th>
              <th>이름</th>
              <th>종류</th>
              <th>기본값</th>
              <th>범위</th>
              <th>설명</th>
            </tr>
          </thead>
          <tbody>
            {m.params.map((s) => (
              <tr key={s.name}>
                <td className="faint">{s.group}</td>
                <td title={s.name}>{s.label}</td>
                <td className="faint">{TYPE_LABEL[s.type] ?? s.type}</td>
                <td className="num">{formatParamValue(s, s.default)}</td>
                <td className="num faint">{specRange(s)}</td>
                <td className="pipe-help">{s.help}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </li>
  )
}

function WeightsEntry({ m }: { m: ModelInfo }) {
  const toast = useToast()
  const patch = useMutation({
    mutationFn: (p: { name?: string; note?: string }) => api.patchModel(m.key, p),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: keys.models }),
    onError: (e: Error) => toast.error(e.message),
  })
  const remove = useMutation({
    mutationFn: () => api.deleteModel(m.key),
    onSuccess: () => {
      toast.ok(`가중치 ${named(m.name, '을/를')} 지웠습니다. 이 모델로 만든 기록은 남아 있습니다`)
      void queryClient.invalidateQueries({ queryKey: keys.models })
      void queryClient.invalidateQueries({ queryKey: keys.storage })
    },
    onError: (e: Error) => toast.error(e.message),
  })
  return (
    <li className={`weights${m.state === 'error' ? ' is-error' : ''}`}>
      <div className="w-main">
        <InlineText value={m.name} placeholder="이름 없음" label="가중치 이름" onSave={(name) => name && patch.mutate({ name })} />
        <code className="pipe-key">{m.key}</code>
      </div>
      {m.state === 'error' ? (
        <p className="error-text">{m.error}</p>
      ) : (
        <dl className="w-facts">
          <div>
            <dt>작업</dt>
            <dd>{TASK_LABEL[m.task ?? ''] ?? '—'}</dd>
          </div>
          <div>
            <dt>클래스</dt>
            <dd title={m.class_names?.join(', ')}>
              {m.class_names?.length ? (
                <>
                  <span className="num">{m.class_names.length}개</span> · {m.class_names.slice(0, 8).join(', ')}
                  {m.class_names.length > 8 ? ' …' : ''}
                </>
              ) : (
                '—'
              )}
            </dd>
          </div>
          <div>
            <dt>크기</dt>
            <dd className="num">{formatBytes(m.file_bytes)}</dd>
          </div>
          <div>
            <dt>올린 때</dt>
            <dd className="num">{formatDate(m.created_at)}</dd>
          </div>
        </dl>
      )}
      <div className="w-foot">
        <InlineText value={m.note} placeholder="메모 추가" label="가중치 메모" onSave={(note) => patch.mutate({ note })} />
        <ConfirmButton label="삭제" onConfirm={() => remove.mutate()} />
      </div>
    </li>
  )
}

function WeightsUpload() {
  const toast = useToast()
  const fileRef = useRef<HTMLInputElement>(null)
  const [name, setName] = useState('')
  const [over, setOver] = useState(false)
  const [progress, setProgress] = useState<{ file: string; pct: number } | null>(null)
  const [result, setResult] = useState<ModelInfo | null>(null)
  const upload = useMutation({
    mutationFn: (file: File) => {
      setProgress({ file: file.name, pct: 0 })
      setResult(null)
      return api.uploadWeights(file, name.trim(), (p) => setProgress({ file: file.name, pct: Math.round((p.loaded / p.total) * 100) }))
    },
    onSuccess: (m) => {
      setResult(m)
      setName('')
      void queryClient.invalidateQueries({ queryKey: keys.models })
      void queryClient.invalidateQueries({ queryKey: keys.storage })
    },
    onError: (e: Error) => toast.error(e.message),
    onSettled: () => setProgress(null),
  })
  const take = (files: FileList | null) => {
    const f = files?.[0]
    if (!f) return
    if (!f.name.toLowerCase().endsWith('.pt')) {
      toast.error('Ultralytics .pt 파일만 올릴 수 있습니다')
      return
    }
    upload.mutate(f)
  }
  return (
    <div
      className={`w-upload${over ? ' over reg' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        take(e.dataTransfer.files)
      }}
    >
      <FileUp size={22} strokeWidth={1.5} />
      <div className="w-upload-text">
        <strong>.pt 가중치를 끌어다 놓거나 고르세요</strong>
        <span className="muted">Ultralytics detect · segment · obb 체크포인트. 올리면 작업 종류와 클래스 이름을 읽어 등록합니다.</span>
      </div>
      <input className="input" value={name} placeholder="표시 이름 (선택)" aria-label="가중치 표시 이름" onChange={(e) => setName(e.target.value)} />
      <button type="button" className="btn btn-primary" disabled={upload.isPending} onClick={() => fileRef.current?.click()}>
        <Upload size={14} strokeWidth={1.75} /> 파일 고르기
      </button>
      <input
        ref={fileRef}
        type="file"
        accept=".pt"
        hidden
        onChange={(e) => {
          take(e.target.files)
          e.target.value = ''
        }}
      />
      {progress ? (
        <div className="w-progress">
          <span className="num">
            {progress.file} · {progress.pct < 100 ? `${progress.pct}%` : '검사 중…'}
          </span>
          <div className="progress">
            <span style={{ width: `${progress.pct}%` }} />
          </div>
        </div>
      ) : null}
      {result ? (
        <p className={`w-result ${result.state === 'error' ? 'error-text' : ''}`} role="status">
          {result.state === 'error'
            ? `등록했지만 쓸 수 없습니다: ${result.error}`
            : `“${result.name}” 등록됨 — ${TASK_LABEL[result.task ?? ''] ?? result.task}, 클래스 ${result.class_names?.length ?? 0}개. 작업대 모델 목록 “내 가중치”에서 고르세요.`}
        </p>
      ) : null}
    </div>
  )
}

const USAGE_ROWS: [keyof StorageInfo['usage'], string][] = [
  ['media_bytes', '원본 미디어'],
  ['proxy_bytes', '영상 프록시'],
  ['weights_bytes', '가중치'],
  ['exports_bytes', '내보낸 파일'],
  ['thumbs_bytes', '썸네일'],
  ['db_bytes', '데이터베이스'],
]

function StorageSection() {
  const toast = useToast()
  const storage = useStorage()
  const [req, setReq] = useState<CleanupRequest>({ delete_exports: true, runs_older_than_days: null, unstarred_only: true })
  const cleanup = useMutation({
    mutationFn: () => api.cleanup(req),
    onSuccess: (r) => {
      toast.ok(`${formatBytes(r.freed_bytes)}를 비웠습니다`)
      void queryClient.invalidateQueries({ queryKey: keys.storage })
      void queryClient.invalidateQueries({ queryKey: keys.system })
      invalidateRuns()
    },
    onError: (e: Error) => toast.error(e.message),
  })
  const s = storage.data
  if (!s) return <p className="faint">{storage.isError ? storage.error.message : '불러오는 중…'}</p>
  const lab = Object.values(s.usage).reduce((a, b) => a + b, 0)
  const used = s.total_bytes - s.free_bytes
  const other = Math.max(0, used - lab)
  const pct = (n: number) => `${(n / s.total_bytes) * 100}%`
  const maxRow = Math.max(...Object.values(s.usage), 1)
  const nothing = !req.delete_exports && req.runs_older_than_days === null
  return (
    <div className="storage">
      <div className="disk">
        <div className="disk-bar" role="img" aria-label={`디스크 ${formatBytes(s.total_bytes)} 중 여유 ${formatBytes(s.free_bytes)}`}>
          <span className="seg lab" style={{ width: pct(lab) }} />
          <span className="seg other" style={{ width: pct(other) }} />
        </div>
        <div className="disk-legend num">
          <span>
            <i className="sw lab" /> Vision Lab {formatBytes(lab)}
          </span>
          <span>
            <i className="sw other" /> 다른 파일 {formatBytes(other)}
          </span>
          <span>
            <i className="sw free" /> 여유 <strong>{formatBytes(s.free_bytes)}</strong> / {formatBytes(s.total_bytes)}
          </span>
        </div>
        {s.free_bytes < 2 * 1024 ** 3 ? <p className="error-text">여유 공간이 2 GB보다 적어 올리기와 실행이 막힙니다. 아래에서 정리하세요.</p> : null}
      </div>
      <table className="table usage">
        <tbody>
          {USAGE_ROWS.map(([k, label]) => (
            <tr key={k}>
              <th scope="row">{label}</th>
              <td className="usage-bar">
                <span style={{ width: `${(s.usage[k] / maxRow) * 100}%` }} />
              </td>
              <td className="r num">{formatBytes(s.usage[k])}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="faint data-dir">
        데이터 폴더 <code>{s.data_dir}</code>
      </p>
      <form
        className="cleanup"
        onSubmit={(e) => {
          e.preventDefault()
        }}
      >
        <h3 className="cmp-title">정리</h3>
        <label className="check-row">
          <input type="checkbox" checked={req.delete_exports} onChange={(e) => setReq({ ...req, delete_exports: e.target.checked })} />
          내보낸 PNG · MP4 지우기 <span className="faint">(필요하면 기록에서 다시 만들 수 있습니다)</span>
        </label>
        <label className="check-row">
          <input
            type="checkbox"
            checked={req.runs_older_than_days !== null}
            onChange={(e) => setReq({ ...req, runs_older_than_days: e.target.checked ? 7 : null })}
          />
          <span>
            <input
              className="input num-input"
              inputMode="numeric"
              aria-label="며칠보다 오래된 기록"
              disabled={req.runs_older_than_days === null}
              value={req.runs_older_than_days ?? 7}
              onChange={(e) => setReq({ ...req, runs_older_than_days: Math.max(0, Math.round(Number(e.target.value) || 0)) })}
            />{' '}
            일보다 오래된 기록 지우기
          </span>
        </label>
        <label className="check-row">
          <input type="checkbox" checked={req.unstarred_only} onChange={(e) => setReq({ ...req, unstarred_only: e.target.checked })} />
          별표 없는 기록만 <span className="faint">(끄면 별표한 기록도 지웁니다)</span>
        </label>
        <div>
          <ConfirmButton label="정리하기" confirmLabel="정말 지우기" small={false} disabled={nothing || cleanup.isPending} onConfirm={() => cleanup.mutate()} />
        </div>
      </form>
    </div>
  )
}

function WorkerSection() {
  const toast = useToast()
  const system = useSystem(true)
  const restart = useMutation({
    mutationFn: api.restartWorker,
    onSuccess: () => {
      toast.ok('작업자를 다시 시작했습니다')
      void queryClient.invalidateQueries({ queryKey: keys.system })
      invalidateRuns()
    },
    onError: (e: Error) => toast.error(e.message),
  })
  const s = system.data
  if (!s) return <p className="faint">{system.isError ? system.error.message : '불러오는 중…'}</p>
  const w = s.worker
  return (
    <div className="worker">
      <dl className="w-facts">
        <div>
          <dt>상태</dt>
          <dd>
            <span className={`status ${w.alive ? (w.busy_run_id ? 'running' : 'done') : 'failed'}`}>
              {w.alive ? (w.busy_run_id ? `기록 #${w.busy_run_id} 처리 중` : '대기 중') : '꺼짐'}
            </span>
          </dd>
        </div>
        <div>
          <dt>PID</dt>
          <dd className="num">{w.pid ?? '—'}</dd>
        </div>
        <div>
          <dt>올린 모델</dt>
          <dd>{w.loaded_model ?? '없음'}</dd>
        </div>
        <div>
          <dt>가속</dt>
          <dd>
            {s.torch_available ? 'torch 있음' : 'torch 없음'} · {s.mps_available === null ? 'MPS 확인 전' : s.mps_available ? 'MPS 사용 가능' : 'MPS 없음 (CPU)'}
          </dd>
        </div>
        <div>
          <dt>버전</dt>
          <dd className="num">{s.version}</dd>
        </div>
      </dl>
      <p className="faint worker-help">
        작업자는 기록을 순서대로 처리하고 가중치 모델은 한 번에 하나만 메모리에 올립니다. 멈춘 것 같거나 메모리를 비우고 싶을 때 다시 시작하세요. 처리 중인 기록은 실패로 표시됩니다.
      </p>
      <ConfirmButton label="작업자 재시작" confirmLabel="정말 재시작" small={false} icon={false} onConfirm={() => restart.mutate()} disabled={restart.isPending} />
    </div>
  )
}

export default function ModelsPage() {
  const models = useModels()
  const pipelines = models.data?.filter((m) => m.kind === 'pipeline') ?? []
  const weights = models.data?.filter((m) => m.kind === 'weights') ?? []
  return (
    <div className="page">
      <div className="page-inner models-page">
        <div className="models-main">
          <section aria-labelledby="h-pipes">
            <div className="page-head">
              <h1 id="h-pipes">내장 파이프라인</h1>
              <span className="faint num">{pipelines.length}개 · OpenCV, 학습 없이 슬라이더로 바로 조절</span>
            </div>
            {models.isError ? <p className="error-text">{models.error.message}</p> : null}
            {models.isPending ? (
              <p className="faint">불러오는 중…</p>
            ) : pipelines.length === 0 ? (
              <p className="faint">등록된 파이프라인이 없습니다.</p>
            ) : (
              <ul className="pipe-list">
                {pipelines.map((m) => (
                  <PipelineEntry key={m.key} m={m} />
                ))}
              </ul>
            )}
          </section>

          <section aria-labelledby="h-weights" className="models-section">
            <div className="page-head">
              <h2 id="h-weights" className="section-title">
                내 가중치
              </h2>
              <span className="faint num">{weights.length}개</span>
            </div>
            <WeightsUpload />
            {weights.length > 0 ? (
              <ul className="weights-list">
                {weights.map((m) => (
                  <WeightsEntry key={m.key} m={m} />
                ))}
              </ul>
            ) : (
              <p className="faint w-none">아직 올린 가중치가 없습니다. 학습은 사이트 밖에서 하고, 결과 .pt만 올려 시험합니다.</p>
            )}
          </section>
        </div>

        <aside className="models-side">
          <section aria-labelledby="h-storage">
            <h2 id="h-storage" className="section-title side-title">
              저장공간
            </h2>
            <StorageSection />
          </section>
          <section aria-labelledby="h-worker" className="models-section">
            <h2 id="h-worker" className="section-title side-title">
              <Cpu size={15} strokeWidth={1.75} /> 작업자
            </h2>
            <WorkerSection />
          </section>
          <p className="faint side-foot">
            <RotateCw size={12} strokeWidth={1.75} /> 상태는 3초마다 새로 고칩니다.
          </p>
        </aside>
      </div>
    </div>
  )
}
