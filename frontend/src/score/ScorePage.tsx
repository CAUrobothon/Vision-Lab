// 채점 (Scoring) — `/score?dataset=&gt=&ref=&skip=`. Batches of a dataset scored against one of its
// ground-truth sets (docs/SCORING.md): the criteria (shared with every screen), the final ranking
// of complete full-coverage batches with paired-bootstrap verdicts against the reference, the
// interim / excluded batches with their reason, slices, augmentation effect, and one batch image
// by image. The selection lives in the URL; the criteria in this viewer's storage.
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { Link, useSearchParams } from 'react-router'
import { Download, Info, ServerOff } from 'lucide-react'
import { api } from '../api/client'
import { useFeatures } from '../api/features'
import { useBatches, useDatasets, useGtSets, useModels, useScore, useScoreImages } from '../api/queries'
import type { ScoreResult } from '../api/types'
import { formatDate, idParam } from '../lib/format'
import { railDataset } from '../workbench/DatasetPanel'
import { batchStamp, chooseGt, useGtPick, useSettledCriteria } from './criteria'
import { batchLabel, int, levelPct } from './format'
import { BatchScope, CriteriaFields } from './parts'
import { ImageList, SourceGrid, useOpenInViewer } from './ScoreDetail'
import { AugmentationTable, InterimTable, RankingTable, SliceTable } from './ScoreTables'
import './score.css'

function Section({ title, note, children, stale }: { title: string; note?: ReactNode; children: ReactNode; stale: boolean }) {
  return (
    <section className={`sc-section${stale ? ' stale' : ''}`} aria-label={title}>
      <div className="sc-section-head">
        <h2 className="section-title">{title}</h2>
        {note ? <span className="faint sc-section-note">{note}</span> : null}
      </div>
      {children}
    </section>
  )
}

function DetailSection({ result, datasetId, stale, stamp, poll }: { result: ScoreResult; datasetId: number; stale: boolean; stamp: string; poll: boolean }) {
  const scored = result.batches.filter((b) => b.n_eval_images > 0)
  const [picked, setPicked] = useState<number | null>(null)
  const batchId = scored.find((b) => b.batch_id === picked)?.batch_id ?? result.reference_batch_id ?? scored[0]?.batch_id ?? null
  const { thresholds } = useSettledCriteria()
  const images = useScoreImages(result.gt_set.id, batchId, thresholds, stamp, poll)
  const open = useOpenInViewer()
  const variants = result.gt_set.variants.map((v) => v.name)
  if (batchId === null) return null
  const rows = images.data ?? []
  const picker = (
    <label className="sc-detail-pick">
      <span className="faint">배치</span>
      <select className="select" aria-label="자세히 볼 배치" value={batchId} onChange={(e) => setPicked(Number(e.target.value))}>
        {scored.map((b) => (
          <option key={b.batch_id} value={b.batch_id}>
            {batchLabel(b)}
            {b.status === 'interim' ? ' · 중간' : ''}
          </option>
        ))}
      </select>
    </label>
  )
  const err = images.error ? <p className="notice error">{images.error.message}</p> : null
  return (
    <>
      <Section
        title="원본별 결과"
        stale={stale || images.isPlaceholderData}
        note={
          <>
            칸 = 이미지 통과 여부. 미 미검출 · 오 오검출 · 색 색 틀림 · 기 기하 실패. 누르면 작업대에서 정답과 함께 엽니다.
          </>
        }
      >
        <div className="sc-tools">{picker}</div>
        {err}
        {images.isPending ? <p className="sc-empty">이미지별 채점을 불러오는 중…</p> : <SourceGrid rows={rows} variants={variants} onOpen={(r) => open(r, datasetId)} />}
      </Section>
      <Section title="이미지별 결과" stale={stale || images.isPlaceholderData} note={`실패가 많은 이미지부터 · ${int(rows.length)}장`}>
        {images.isPending ? null : <ImageList rows={rows} onOpen={(r) => open(r, datasetId)} />}
      </Section>
    </>
  )
}

export default function ScorePage() {
  const [sp, setSp] = useSearchParams()
  const datasets = useDatasets()
  const models = useModels()
  const modelMap = useMemo(() => new Map((models.data ?? []).map((m) => [m.key, m])), [models.data])
  const list = datasets.data ?? []

  // The dataset: the URL's, else the one the workbench rail shows, else the first.
  const wanted = idParam(sp.get('dataset')) ?? railDataset()
  const dataset = list.find((d) => d.id === wanted) ?? list[0] ?? null
  const datasetId = dataset?.id ?? null
  const gtSets = useGtSets(datasetId)
  const [pickedGt, pickGt] = useGtPick(datasetId)
  const gt = chooseGt(gtSets.data, idParam(sp.get('gt')) ?? pickedGt)
  // A server started before ground truth and scoring existed: it says so in its features.
  const features = useFeatures()
  const unsupported = features !== null && !(features.has('gt') && features.has('score'))

  const batches = useBatches(datasetId)
  const all = batches.data ?? []
  const skip = useMemo(() => (sp.get('skip') ?? '').split(',').map(Number).filter((n) => n > 0), [sp])
  const selected = all.filter((b) => !skip.includes(b.id))
  const ref = idParam(sp.get('ref'))
  const { criteria: c, settling: typing } = useSettledCriteria()
  const stamp = batchStamp(all)
  const poll = selected.some((b) => !b.finished)
  const req = gt && selected.length > 0 ? { gt_set_id: gt.id, batch_ids: selected.map((b) => b.id), criteria: c, reference_batch_id: ref } : null
  const score = useScore(req, stamp, poll)
  const result = score.data ?? null
  const stale = score.isPlaceholderData || typing

  const set = (patch: Record<string, string | null>) =>
    setSp(
      (prev) => {
        const next = new URLSearchParams(prev)
        for (const [k, v] of Object.entries(patch)) {
          if (v === null || v === '') next.delete(k)
          else next.set(k, v)
        }
        return next
      },
      { replace: true },
    )

  // Keep the URL naming what is shown, so a reload or a shared link opens the same scoring.
  useEffect(() => {
    if (datasetId !== null && sp.get('dataset') !== String(datasetId)) set({ dataset: String(datasetId) })
  }, [datasetId]) // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = (id: number, on: boolean) => {
    const next = on ? skip.filter((x) => x !== id) : [...skip, id]
    set({ skip: next.length ? next.join(',') : null, ref: !on && ref === id ? null : sp.get('ref') })
  }

  const csv = (kind: 'cubes' | 'summary') => (gt ? api.scoreCsvUrl(kind, gt.id, req?.batch_ids ?? [], c) : undefined)

  return (
    <div className="page">
      <div className="page-inner score-page">
        <div className="page-head">
          <h1>채점</h1>
          {gt ? (
            <span className="faint num">
              {gt.name} · 이미지 {int(gt.n_images)} · 큐브 {int(gt.n_cubes)} (기하 유효 {int(gt.n_valid)}) · layout_group {int(gt.n_layout_groups)}
            </span>
          ) : null}
          <span className="sc-head-actions">
            {(['cubes', 'summary'] as const).map((k) =>
              gt && req ? (
                <a key={k} className="btn btn-sm" href={csv(k)} download title={k === 'cubes' ? '정답 큐브 한 줄 + 오검출 한 줄 (BOM UTF-8, 기준값 열 포함)' : '배치 × 지표 한 줄 (중간 배치는 status 열이 중간)'}>
                  <Download size={12} strokeWidth={1.75} /> {k}.csv 받기
                </a>
              ) : (
                <button key={k} type="button" className="btn btn-sm" disabled>
                  <Download size={12} strokeWidth={1.75} /> {k}.csv 받기
                </button>
              ),
            )}
          </span>
        </div>

        <p className="notice sc-notice">
          <Info size={14} strokeWidth={1.75} />
          <span>
            정답셋은 AI 보조로 만든 근사 이미지 라벨입니다(보정된 물리 정답이 아님). 작은 차이로 순위를 정하지 않습니다 — 차이는 layout_group 단위 paired
            bootstrap 구간으로만 판정합니다: 구간이 0을 포함하면 <strong>차이 불분명</strong>.
          </span>
        </p>

        <div className="sc-bar">
          <div className="sc-pick">
            <label className="sc-field">
              <span className="faint">데이터셋</span>
              <select className="select" aria-label="데이터셋" value={datasetId ?? ''} onChange={(e) => setSp({ dataset: e.target.value }, { replace: true })}>
                {list.length === 0 ? <option value="">데이터셋 없음</option> : null}
                {list.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} · 이미지 {d.image_count}
                  </option>
                ))}
              </select>
            </label>
            <label className="sc-field">
              <span className="faint">정답셋</span>
              <select
                className="select"
                aria-label="정답셋"
                value={gt?.id ?? ''}
                disabled={!gtSets.data?.length}
                onChange={(e) => {
                  pickGt(Number(e.target.value))
                  set({ gt: e.target.value })
                }}
              >
                {!gtSets.data?.length ? <option value="">정답셋 없음</option> : null}
                {gtSets.data?.map((g) => (
                  <option key={g.id} value={g.id}>
                    {g.name} · {g.n_images}장 · 연결 {g.links.linked_sha256 + g.links.linked_confirmed}
                  </option>
                ))}
              </select>
            </label>
            <label className="sc-field">
              <span className="faint">기준 배치</span>
              <select className="select" aria-label="기준 배치" value={ref ?? ''} onChange={(e) => set({ ref: e.target.value || null })} disabled={selected.length === 0}>
                <option value="">자동 — 처음 나오는 완료 · 전체 포함 배치</option>
                {selected.map((b) => (
                  <option key={b.id} value={b.id}>
                    {batchLabel(b)}
                  </option>
                ))}
              </select>
            </label>
            <span className={`sc-status num${(score.isFetching && result) || typing ? ' busy' : ''}`} aria-live="polite">
              {(score.isFetching && result) || typing ? <span className="sc-loading-bar" aria-hidden="true" /> : null}
              {score.isFetching && result
                ? '다시 채점하는 중… 아래 표는 이전 결과'
                : typing
                  ? '기준값을 바꾸는 중…'
                  : result
                    ? `채점 ${formatDate(result.scored_at)}`
                    : ''}
            </span>
          </div>
          <CriteriaFields />
          {all.length > 0 ? (
            <fieldset className="sc-batches">
              <legend className="cf-title">일괄 실행</legend>
              <div className="sc-batch-list">
                {all.map((b) => {
                  const on = !skip.includes(b.id)
                  return (
                    <label key={b.id} className={`sc-batch-chip${on ? ' on' : ''}`} title={`${b.model_name} · ${b.folder ? `폴더 ${b.folder}` : '전체'} · ${formatDate(b.created_at)}`}>
                      <input type="checkbox" checked={on} onChange={(e) => toggle(b.id, e.target.checked)} />
                      <span className="sc-id num">#{b.id}</span>
                      <span className="sc-chip-name">{b.model_name}</span>
                      {b.folder ? <BatchScope folder={b.folder} /> : null}
                      {!b.finished ? <span className="tag-interim">실행 중</span> : null}
                    </label>
                  )
                })}
              </div>
              <span className="sc-batch-actions">
                <button type="button" className="btn btn-sm btn-quiet" disabled={skip.length === 0} onClick={() => set({ skip: null })}>
                  모두
                </button>
                <button type="button" className="btn btn-sm btn-quiet" disabled={selected.length === 0} onClick={() => set({ skip: all.map((b) => b.id).join(','), ref: null })}>
                  모두 해제
                </button>
              </span>
            </fieldset>
          ) : null}
        </div>

        {unsupported ? (
          <div className="notice warn sc-state-note" role="status">
            <ServerOff size={14} strokeWidth={1.75} />
            <span>
              서버를 다시 켜야 채점을 쓸 수 있습니다. 지금 떠 있는 서버는 정답셋 · 채점 기능이 생기기 전의 코드입니다 — 서버를 끄고 <code>./run.sh</code> 로 다시 켜세요.
            </span>
          </div>
        ) : !datasets.isPending && list.length === 0 ? (
          <div className="empty">
            <strong>데이터셋이 없습니다</strong>
            <span>
              <Link to="/">작업대</Link> 왼쪽에서 데이터셋을 만들고 이미지와 정답 JSON을 올리세요.
            </span>
          </div>
        ) : gtSets.isSuccess && gtSets.data.length === 0 ? (
          <div className="empty">
            <strong>이 데이터셋에는 정답셋이 없습니다</strong>
            <span>
              <Link to="/">작업대</Link> 왼쪽 데이터셋 칸의 “정답 JSON 올리기”로 정답셋을 올리면 여기서 채점합니다.
            </span>
          </div>
        ) : gtSets.isError ? (
          <p className="notice error">{gtSets.error.message}</p>
        ) : batches.isSuccess && all.length === 0 ? (
          <div className="empty">
            <strong>채점할 일괄 실행이 없습니다</strong>
            <span>
              <Link to="/">작업대</Link>에서 이 데이터셋을 고른 뒤 “지금 모델로 전체 실행”을 누르면 여기에 나옵니다.
            </span>
          </div>
        ) : selected.length === 0 && all.length > 0 ? (
          <p className="sc-empty">고른 일괄 실행이 없습니다. 위에서 하나 이상 고르세요.</p>
        ) : score.isError ? (
          <p className="notice error" role="alert">
            채점하지 못했습니다: {score.error.message}
          </p>
        ) : !result ? (
          <div className="sc-loading" role="status">
            <span className="sc-loading-bar" aria-hidden="true" />
            <span>
              채점하는 중… 서버를 켠 뒤 처음이거나 τ · g를 바꾸면 이미지를 모두 다시 채점해서 몇 초 걸립니다. 같은 기준으로 다시 열면 바로 나옵니다.
            </span>
          </div>
        ) : (
          <>
            {result.reference_note ? (
              <p className="notice warn">
                <Info size={14} strokeWidth={1.75} />
                <span>{result.reference_note}</span>
              </p>
            ) : null}
            <Section
              title="최종 순위"
              stale={stale}
              note={`완료 · 정답셋 전체를 채점한 배치만 · 허용 오차 충족률, 이미지 전체 기준 충족률 순 · Δ = 이 배치 − 기준, ${levelPct(result.criteria.bootstrap_level)} 구간 (반복 ${int(result.criteria.bootstrap_reps)}회, 시드 ${result.criteria.bootstrap_seed})`}
            >
              <RankingTable result={result} models={modelMap} />
            </Section>
            <Section title="중간 결과 · 비교 제외" stale={stale} note="끝나지 않았거나 범위가 정답셋 전체가 아닌 배치. 값은 끝난 이미지만의 중간 점수입니다.">
              <InterimTable result={result} models={modelMap} />
            </Section>
            <Section title="조각 비교" stale={stale}>
              <SliceTable result={result} />
            </Section>
            <Section title="증강 영향" stale={stale}>
              <AugmentationTable result={result} />
            </Section>
            {datasetId !== null ? <DetailSection result={result} datasetId={datasetId} stale={stale} stamp={stamp} poll={poll} /> : null}
          </>
        )}
      </div>
    </div>
  )
}
