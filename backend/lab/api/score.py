"""/api/score: batches and single images scored against a ground-truth set (docs/SCORING.md)."""

from __future__ import annotations

from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Query, Response

from lab import scoring
from lab.deps import Conn, Paths
from lab.errors import LabError
from lab.schemas import (
    ScoreCriteriaIn, ScoreCriteriaInfo, ScoreFrameRequest, ScoreImageDetail, ScoreImageRow, ScoreRequest, ScoreResult,
)

router = APIRouter(prefix="/api/score", tags=["score"])


class ScoreQuery(ScoreCriteriaIn):
    """The selection of POST /api/score as query parameters; batch_ids is a comma list."""

    gt_set_id: int
    batch_ids: str = ""


class ImageQuery(ScoreCriteriaIn):
    run_id: int
    gt_set_id: int | None = None  # null = the newest set that links the run's media


class ImagesQuery(ScoreCriteriaIn):
    gt_set_id: int
    batch_id: int


def _batch_ids(raw: str) -> list[int]:
    try:
        return [int(p) for p in raw.split(",") if p.strip()]
    except ValueError:
        raise LabError(400, f"batch_ids는 쉼표로 나눈 번호여야 합니다: {raw}") from None


def _csv(data: bytes, filename: str) -> Response:
    """Like FileResponse's Content-Disposition: RFC 5987 for a non-ASCII (Korean) name."""
    quoted = quote(filename)
    disposition = f"attachment; filename*=utf-8''{quoted}" if quoted != filename else f'attachment; filename="{filename}"'
    return Response(data, media_type="text/csv; charset=utf-8", headers={"Content-Disposition": disposition})


@router.post("")
def score(req: ScoreRequest, conn: Conn, paths: Paths) -> ScoreResult:
    """Metrics, slices and augmentation effect per batch, paired bootstrap against the reference,
    and the final ranking of complete full-coverage batches. ``detail: "summary"`` answers with
    status, pending, coverage and metrics per batch only (slices, augmentation effect and
    comparison null): no bootstrap is run."""
    return scoring.score(conn, paths, req)


@router.get("/criteria")
def criteria() -> ScoreCriteriaInfo:
    """The criteria defaults and limits (min / max, exclusive or not, step, integer), as the
    request schema states them."""
    return scoring.criteria_info()


@router.get("/image")
def score_image(q: Annotated[ImageQuery, Query()], conn: Conn, paths: Paths) -> ScoreImageDetail:
    """A run's frame 0 against its media's GT image (the viewer overlay). 404 when the media has no GT."""
    return scoring.image_detail(conn, paths, q.run_id, q.gt_set_id, q)


@router.get("/images")
def score_images(q: Annotated[ImagesQuery, Query()], conn: Conn, paths: Paths) -> list[ScoreImageRow]:
    """Every evaluation image of one batch with its pass and counts (missed, FP, colour, geometry), in GT order."""
    return scoring.image_rows(conn, paths, q.gt_set_id, q.batch_id, q)


@router.post("/frame")
def score_frame(req: ScoreFrameRequest, conn: Conn, paths: Paths) -> ScoreImageDetail:
    """Given cubes (a workbench preview) against the media's GT image. 404 when the media has no GT."""
    return scoring.frame_detail(conn, paths, req.media_id, req.cubes, req.gt_set_id, req.criteria)


@router.get("/cubes.csv", response_class=Response, responses={200: {"content": {"text/csv": {}}}})
def cubes_csv(q: Annotated[ScoreQuery, Query()], conn: Conn, paths: Paths) -> Response:
    """One row per GT cube and per false positive (UTF-8 with BOM, thresholds as columns)."""
    return _csv(*scoring.cubes_csv(conn, paths, q.gt_set_id, _batch_ids(q.batch_ids), q))


@router.get("/summary.csv", response_class=Response, responses={200: {"content": {"text/csv": {}}}})
def summary_csv(q: Annotated[ScoreQuery, Query()], conn: Conn, paths: Paths) -> Response:
    """One row per batch and metric, with its status (완료 / 중간)."""
    return _csv(*scoring.summary_csv(conn, paths, q.gt_set_id, _batch_ids(q.batch_ids), q))
