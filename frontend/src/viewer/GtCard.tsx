// The 정답 layer's card over the viewer: the image's outcome and counts, and for the detection
// under the pointer (or picked in the result table) its errors against the paired GT cube.
import type { ScoreImageDetail } from '../api/types'
import { colorName } from '../lib/cubes'
import { stat } from '../score/format'
import { gtOfPred } from '../score/image'
import { CountsLine, CriteriaLine, PassBadge } from '../score/parts'

export function GtCard({ d, highlight }: { d: ScoreImageDetail; highlight: number | null }) {
  const k = d.criteria
  const gi = highlight === null ? null : gtOfPred(d, highlight)
  const isFp = highlight !== null && d.false_positives.includes(highlight)
  // The highlighted detection's GT cube and its result against it.
  const g = gi === null ? null : d.gt[gi]
  const c = gi === null ? null : d.cubes[gi]
  const over = (ok: boolean | null) => (ok === false ? 'fail-text' : '') // the server's tolerance check
  return (
    <div className="gt-card num" aria-live="polite">
      <div className="gtc-head">
        {!d.scored ? <span className="faint">실행이 끝나면 채점합니다</span> : <PassBadge passed={!!d.passed} className="gtc-pass" />}
        <span className="gtc-set" title={`정답셋 ${d.gt_set_name} · 정답 이미지 ${d.gt_index}번 (${d.method === 'sha256' ? 'sha256 연결' : '확인한 연결'})`}>
          {d.filename} · {d.variant}
        </span>
      </div>
      {d.scored ? (
        <CountsLine d={d} className="gtc-counts">
          {d.ran === false ? ' · 실행 실패 (검출 0개로 셈)' : ''}
        </CountsLine>
      ) : null}
      {highlight !== null && d.scored ? (
        <div className="gtc-cube">
          {g && c ? (
            <>
              <div className="gtc-line">
                검출 {highlight + 1} ↔ 정답 <strong>{g.id}</strong>
                <span className={`gtc-outcome ${c.outcome === 'pass' ? 'ok-text' : 'fail-text'}`}>{c.outcome === 'pass' ? '통과' : '미달'}</span>
              </div>
              <dl className="gtc-errs">
                <dt>짝 거리</dt>
                <dd>{stat(c.distance, 'px')}</dd>
                <dt>색</dt>
                <dd className={c.color_ok ? '' : 'fail-text'}>
                  {colorName(g.color)}
                  {c.color_ok ? '' : ` ≠ ${colorName(c.pred_color)}`}
                </dd>
                {g.valid_geometry ? (
                  <>
                    <dt>중심</dt>
                    <dd className={over(c.center_ok)}>{stat(c.center_err, 'px')}</dd>
                    <dt>회전각</dt>
                    <dd className={over(c.angle_ok)}>{stat(c.angle_err, '°', 1)}</dd>
                    <dt>꼭짓점 RMS</dt>
                    <dd className={over(c.vertex_ok)}>{stat(c.vertex_err, 'px')}</dd>
                  </>
                ) : (
                  <>
                    <dt>기하</dt>
                    <dd className="faint">정답 기하 무효 — 짝과 색만 봄</dd>
                  </>
                )}
              </dl>
            </>
          ) : isFp ? (
            <div className="gtc-line">
              검출 {highlight + 1}: <strong className="fail-text">오검출</strong> — 게이트 {k.gate_px} px 안에 짝이 되는 정답이 없습니다
            </div>
          ) : null}
        </div>
      ) : null}
      <CriteriaLine c={k} className="gtc-crit faint" />
    </div>
  )
}
