// One image's score (ScoreImageDetail) read for the viewer card and the workbench's result panel.
// Outcomes, tolerance checks and counts are the server's; this only finds a detection's GT cube.
import type { ScoreImageDetail } from '../api/types'

/** The GT cube a prediction is paired with (index into d.gt / d.cubes), or null. */
export function gtOfPred(d: ScoreImageDetail, pred: number): number | null {
  const i = d.cubes.findIndex((c) => c.matched && c.pred_index === pred)
  return i < 0 ? null : i
}
