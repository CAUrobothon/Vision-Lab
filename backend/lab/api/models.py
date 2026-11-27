"""/api/models (pipelines + uploaded weights) and /api/presets."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from lab import models
from lab.config import MAX_WEIGHTS_BYTES, DataPaths
from lab.db import connect, dumps, loads, now_iso
from lab.deps import Conn, Paths
from lab.errors import LabError
from lab.schemas import ModelInfo, ModelPatch, Ok, ParamCheck, ParamCheckRequest, Preset, PresetCreate
from lab.storage import guard_disk
from lab.upload import Upload, receive_upload

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
def list_models(conn: Conn) -> list[ModelInfo]:
    return models.list_models(conn)


def _register(paths: DataPaths, up: Upload) -> ModelInfo:
    conn = connect(paths.db)
    try:
        return models.register_weights(conn, paths, up)
    finally:
        conn.close()


@router.post("/models/weights", openapi_extra={"requestBody": {"content": {"multipart/form-data": {"schema": {
    "type": "object", "required": ["file"],
    "properties": {"file": {"type": "string", "format": "binary"}, "name": {"type": "string"}}}}}}})
async def upload_weights(request: Request) -> ModelInfo:
    """Multipart ``file`` (.pt) and optional ``name``. A file Ultralytics cannot use is kept with state=error."""
    paths: DataPaths = request.app.state.paths
    guard_disk(paths)
    up = await receive_upload(request, paths.tmp, max_bytes=MAX_WEIGHTS_BYTES)
    if up.ext != ".pt":
        up.discard()
        raise LabError(415, "Ultralytics 가중치(.pt) 파일만 올릴 수 있습니다")
    return await run_in_threadpool(_register, paths, up)


def _only_weights(key: str) -> None:
    if models.weights_id(key) is None:
        raise LabError(400, "내장 파이프라인은 수정하거나 지울 수 없습니다")


@router.patch("/models/{key}")
def patch_model(key: str, patch: ModelPatch, conn: Conn) -> ModelInfo:
    _only_weights(key)
    row = models.weights_row(conn, key)
    if patch.name is not None:
        name = patch.name.strip()
        if not name:
            raise LabError(400, "이름을 비울 수 없습니다")
        conn.execute("UPDATE weights SET name = ? WHERE id = ?", (name[:80], row["id"]))
    if patch.note is not None:
        conn.execute("UPDATE weights SET note = ? WHERE id = ?", (patch.note, row["id"]))
    return models.weights_info(models.weights_row(conn, key))


@router.delete("/models/{key}")
def delete_model(key: str, conn: Conn, paths: Paths) -> Ok:
    _only_weights(key)
    models.delete_weights(conn, paths, key)
    return Ok()


def _preset(row: sqlite3.Row) -> Preset:
    return Preset(id=row["id"], model_key=row["model_key"], name=row["name"], params=loads(row["params"], {}),
                  created_at=row["created_at"])


@router.post("/params/check")
def check_params(req: ParamCheckRequest, conn: Conn) -> ParamCheck:
    """Resolve hand-written params (e.g. JSON pasted from a chat assistant) the way a run would,
    and report which were applied, which changed to fit, and which the model does not have."""
    return ParamCheck(**models.check_params(conn, req.model_key, req.params))


@router.get("/presets")
def list_presets(conn: Conn, model_key: str | None = None) -> list[Preset]:
    if model_key:
        rows = conn.execute("SELECT * FROM presets WHERE model_key = ? ORDER BY name", (model_key,))
    else:
        rows = conn.execute("SELECT * FROM presets ORDER BY model_key, name")
    return [_preset(r) for r in rows]


@router.post("/presets")
def create_preset(req: PresetCreate, conn: Conn) -> Preset:
    """Saving under an existing name for the same model overwrites that preset."""
    name = req.name.strip()
    if not name:
        raise LabError(400, "프리셋 이름을 입력하세요")
    params = models.resolve(conn, req.model_key, req.params).params
    conn.execute(
        """INSERT INTO presets (model_key, name, params, created_at) VALUES (?, ?, ?, ?)
           ON CONFLICT (model_key, name) DO UPDATE SET params = excluded.params, created_at = excluded.created_at""",
        (req.model_key, name, dumps(params), now_iso()),
    )
    row = conn.execute("SELECT * FROM presets WHERE model_key = ? AND name = ?", (req.model_key, name)).fetchone()
    return _preset(row)


@router.delete("/presets/{preset_id}")
def delete_preset(preset_id: int, conn: Conn) -> Ok:
    if conn.execute("DELETE FROM presets WHERE id = ?", (preset_id,)).rowcount == 0:
        raise LabError(404, "프리셋을 찾을 수 없습니다")
    return Ok()

