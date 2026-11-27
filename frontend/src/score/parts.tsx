// Pieces every scoring view shares: the verdict tag (word + arrow + tone, never colour alone), the
// Δ line with its bootstrap interval, the criteria fields with the provisional label, a batch's
// scope, and one image's pass, counts and criteria (the viewer's card, the workbench's live score).
import type { ReactNode } from 'react'
import { ArrowDown, ArrowUp, Check, Minus, RotateCcw, X } from 'lucide-react'
import type { ScoreCriteria, ScoreCriteriaInfo, ScoreDiff, ScoreImageDetail, Verdict } from '../api/types'
import { NumberField } from '../ui/controls'
import { isDefaultCriteria, resetCriteria, useCriteria, useCriteriaInfo } from './criteria'
import { DASH, delta, interval, levelPct, tone, VERDICT_TEXT, type ValueKind } from './format'
import './parts.css'

export const PROVISIONAL = '잠정 기준 — 로봇 실제 허용 오차 아님'

function VerdictTag({ verdict, better }: { verdict: Verdict; better: 'higher' | 'lower' }) {
  const t = tone(verdict, better)
  const Icon = verdict === 'higher' ? ArrowUp : verdict === 'lower' ? ArrowDown : Minus
  const meaning = t === 'good' ? ' (나아짐)' : t === 'bad' ? ' (나빠짐)' : ''
  return (
    <span className={`verdict ${t}`} title={`기준 대비 ${VERDICT_TEXT[verdict]}${meaning}`}>
      <Icon size={11} strokeWidth={2.25} aria-hidden="true" />
      {VERDICT_TEXT[verdict]}
    </span>
  )
}

/** Δ (this − the reference) with the paired bootstrap interval and the verdict. */
export function DiffLine({ diff, kind, unit, better, level }: { diff: ScoreDiff | null; kind: ValueKind; unit: string; better: 'higher' | 'lower'; level: number }) {
  if (!diff || diff.verdict === null)
    return <span className="sc-diff faint" title="한쪽이라도 값이 없어 차이를 판정하지 않았습니다">Δ —</span>
  return (
    <span className="sc-diff" title={`${levelPct(level)} 구간 ${interval(diff.ci_low, diff.ci_high, kind)} · 재표집 ${diff.reps_used}회`}>
      <span className="num">Δ {delta(diff.delta, kind, unit)}</span>
      <span className="sc-ci num">{interval(diff.ci_low, diff.ci_high, kind)}</span>
      <VerdictTag verdict={diff.verdict} better={better} />
    </span>
  )
}

function Field({ k, label, aria, unit, value, info, onChange }: {
  k: keyof ScoreCriteria
  label: string
  aria: string
  unit?: string
  value: ScoreCriteria
  info: ScoreCriteriaInfo
  onChange: (patch: Partial<ScoreCriteria>) => void
}) {
  // The server's limits; a field without a maximum has none here either.
  const lim = info.limits[k]
  const def = info.defaults[k]
  return (
    <label className={`cf-field${value[k] !== def ? ' changed' : ''}`} title={`기본값 ${def}`}>
      <span className="cf-label">{label}</span>
      <NumberField
        integer={lim.integer}
        label={aria}
        value={value[k]}
        min={lim.min ?? -Infinity}
        max={lim.max ?? Infinity}
        minExclusive={lim.min_exclusive}
        maxExclusive={lim.max_exclusive}
        onCommit={(v) => onChange({ [k]: v })}
      />
      {unit ? <span className="cf-unit">{unit}</span> : null}
    </label>
  )
}

/** The criteria editor. `bootstrap` adds the difference test's settings (the 채점 page). */
export function CriteriaFields({ bootstrap = true }: { bootstrap?: boolean }) {
  const [c, set] = useCriteria()
  const info = useCriteriaInfo()
  const d = info.defaults
  const defaultsText = `τc ${d.center_px} · τa ${d.angle_deg} · τv ${d.vertex_px} · g ${d.gate_px} · 반복 ${d.bootstrap_reps} · 수준 ${d.bootstrap_level} · 시드 ${d.bootstrap_seed}`
  const field = { value: c, info, onChange: set }
  return (
    <div className="cf" role="group" aria-label="채점 기준">
      <div className="cf-group">
        <span className="cf-title">허용 오차</span>
        <Field k="center_px" label="중심 τc" aria="중심 허용 오차 τc (px)" unit="px" {...field} />
        <Field k="angle_deg" label="회전각 τa" aria="회전각 허용 오차 τa (°)" unit="°" {...field} />
        <Field k="vertex_px" label="꼭짓점 RMS τv" aria="꼭짓점 RMS 허용 오차 τv (px)" unit="px" {...field} />
        <span className="provisional">{PROVISIONAL}</span>
      </div>
      <div className="cf-group">
        <span className="cf-title">매칭</span>
        <Field k="gate_px" label="게이트 g" aria="매칭 게이트 g (px)" unit="px" {...field} />
        {bootstrap ? (
          <>
            <span className="cf-title cf-gap" title="차이는 layout_group 단위 paired bootstrap 구간으로만 판정합니다">
              차이 판정
            </span>
            <Field k="bootstrap_reps" label="반복" aria="bootstrap 반복 횟수" unit="회" {...field} />
            <Field k="bootstrap_level" label="수준" aria="bootstrap 구간 수준" {...field} />
            <Field k="bootstrap_seed" label="시드" aria="bootstrap 시드" {...field} />
          </>
        ) : null}
        <button type="button" className="btn btn-sm btn-quiet cf-reset" disabled={isDefaultCriteria(c, d)} onClick={resetCriteria} title={defaultsText}>
          <RotateCcw size={12} strokeWidth={1.75} /> 기본값으로
        </button>
      </div>
    </div>
  )
}

/** A batch's scope: its folder (with the folders below it) or the whole dataset. */
export function BatchScope({ folder }: { folder: string | null }) {
  return (
    <span className="ds-scope" title={folder ? `실행 범위: 폴더 "${folder}" (하위 폴더 포함)` : '실행 범위: 데이터셋 전체'}>
      {folder ?? '전체'}
    </span>
  )
}

// ----- one image's score (ScoreImageDetail): the viewer's 정답 card, the workbench's live score -----

/** 이미지 통과 / 미달 with its mark, in the pass / fail colour. */
export function PassBadge({ passed, className }: { passed: boolean; className: string }) {
  const Icon = passed ? Check : X
  return (
    <span className={`${className} ${passed ? 'ok-text' : 'fail-text'}`}>
      <Icon size={12} strokeWidth={2.5} />
      {passed ? '이미지 통과' : '이미지 미달'}
    </span>
  )
}

/** 정답 · 미검출 · 오검출 · 색 틀림 · 기하 실패 of the image, as the server counts them (`short`:
 *  색 · 기하), then `children`. */
export function CountsLine({ d, short = false, className, title, children }: { d: ScoreImageDetail; short?: boolean; className: string; title?: string; children?: ReactNode }) {
  const n = (v: number | null) => v ?? DASH
  return (
    <span className={className} title={title}>
      {`정답 ${d.n_gt} · 미검출 ${n(d.n_missed)} · 오검출 ${n(d.n_fp)} · ${short ? '색' : '색 틀림'} ${n(d.n_color_wrong)} · ${short ? '기하' : '기하 실패'} ${n(d.n_geom_fail)}`}
      {children}
    </span>
  )
}

/** The four thresholds an image is scored with, and that they are provisional. */
export function CriteriaLine({ c, className }: { c: Pick<ScoreCriteria, 'center_px' | 'angle_deg' | 'vertex_px' | 'gate_px'>; className: string }) {
  return (
    <p className={className} title={PROVISIONAL}>
      τc {c.center_px} px · τa {c.angle_deg}° · τv {c.vertex_px} px · g {c.gate_px} px · 잠정 기준
    </p>
  )
}
