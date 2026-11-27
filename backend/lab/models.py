"""The model catalog: built-in pipelines (``cv.*``) plus uploaded checkpoints (``w.<id>``)."""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cubevision.pipelines.base import Pipeline
from cubevision.registry import get_pipeline, list_pipelines
from cubevision.types import ParamError, ParamSpec, resolve_params

from lab import ultra
from lab.config import DataPaths
from lab.db import dumps, loads, now_iso, transaction
from lab.errors import LabError
from lab.schemas import ModelInfo
from lab.upload import Upload

WEIGHTS_PREFIX = "w."


def weights_id(key: str) -> int | None:
    if not key.startswith(WEIGHTS_PREFIX):
        return None
    try:
        return int(key[len(WEIGHTS_PREFIX):])
    except ValueError:
        return None


def weights_row(conn: sqlite3.Connection, key: str) -> sqlite3.Row:
    wid = weights_id(key)
    row = conn.execute("SELECT * FROM weights WHERE id = ?", (wid,)).fetchone() if wid is not None else None
    if row is None:
        raise LabError(404, "모델을 찾을 수 없습니다")
    return row


def find_pipeline(key: str) -> Pipeline:
    try:
        return get_pipeline(key)
    except KeyError:
        raise LabError(404, "모델을 찾을 수 없습니다") from None


def weights_specs(row: sqlite3.Row) -> list[ParamSpec]:
    return ultra.param_specs(row["task"], row["imgsz"], loads(row["class_names"], []))


def weights_path(paths: DataPaths, row: sqlite3.Row) -> Path:
    return paths.root / row["file"]


def _size_mb(n: int) -> str:
    return f"{n / 1024**2:.1f} MB"


def weights_info(row: sqlite3.Row) -> ModelInfo:
    task = row["task"] if row["task"] in ultra.TASKS else None
    names = loads(row["class_names"], [])
    if row["state"] == "ready":
        shown = ", ".join(names[:8]) + (f" 외 {len(names) - 8}개" if len(names) > 8 else "")
        summary = f"{ultra.TASK_LABELS[task]} · 클래스 {len(names)}개 · {_size_mb(row['bytes'])}"
        description = (
            f"올린 Ultralytics 가중치 ({ultra.TASK_LABELS[task]}).\n"
            f"클래스: {shown}\n"
            f"학습 입력 크기: {row['imgsz'] or '알 수 없음'}\n"
            "결과 큐브는 모든 모델이 같은 후처리로 윗면 사각형·중심·각도를 만듭니다. "
            "클래스 이름이 색 이름(red, green, blue, wood ...)이면 그 색을, 아니면 윗면 픽셀로 색을 정합니다."
        )
    else:
        summary = "사용할 수 없음"
        description = row["error"] or ""
    return ModelInfo(
        key=f"{WEIGHTS_PREFIX}{row['id']}",
        kind="weights",
        name=row["name"],
        summary=summary,
        description=description,
        params=[p.to_dict() for p in weights_specs(row)] if row["state"] == "ready" else [],
        state=row["state"],
        error=row["error"],
        task=task,
        class_names=names,
        file_bytes=row["bytes"],
        note=row["note"],
        created_at=row["created_at"],
    )


def pipeline_info(p: Pipeline) -> ModelInfo:
    info = p.info()
    return ModelInfo(
        key=info["key"], kind="pipeline", name=info["name"], summary=info["summary"],
        description=info["description"], params=info["params"], state="ready",
        debug_labels=info.get("debug_labels", {}),
    )


def list_models(conn: sqlite3.Connection) -> list[ModelInfo]:
    out = [pipeline_info(p) for p in list_pipelines()]
    out += [weights_info(r) for r in conn.execute("SELECT * FROM weights ORDER BY id DESC")]
    return out


def register_weights(conn: sqlite3.Connection, paths: DataPaths, up: Upload) -> ModelInfo:
    """Store an uploaded .pt and inspect it in a subprocess. Bad files stay listed with state=error."""
    same = conn.execute("SELECT * FROM weights WHERE sha256 = ?", (up.sha256,)).fetchone()
    if same is not None:
        up.discard()
        return weights_info(same)
    name = (up.fields.get("name") or "").strip()[:80] or Path(up.filename).stem
    with transaction(conn):
        wid = int(conn.execute(
            """INSERT INTO weights (name, file, bytes, sha256, state, error, created_at)
               VALUES (?, '', ?, ?, 'error', '검사 중', ?)""",
            (name, up.bytes, up.sha256, now_iso()),
        ).lastrowid)
        rel = f"weights/{wid}.pt"
        conn.execute("UPDATE weights SET file = ? WHERE id = ?", (rel, wid))
    try:
        shutil.move(up.path, paths.root / rel)
    except BaseException:
        conn.execute("DELETE FROM weights WHERE id = ?", (wid,))
        up.discard()
        raise
    result = ultra.inspect_in_subprocess(paths.root / rel, paths.root)
    conn.execute(
        "UPDATE weights SET state = ?, error = ?, task = ?, class_names = ?, imgsz = ? WHERE id = ?",
        ("ready" if result.get("ok") else "error", None if result.get("ok") else result.get("error"),
         result.get("task"), dumps(result.get("class_names") or []), result.get("imgsz"), wid),
    )
    return weights_info(conn.execute("SELECT * FROM weights WHERE id = ?", (wid,)).fetchone())


def delete_weights(conn: sqlite3.Connection, paths: DataPaths, key: str) -> None:
    """Runs made with these weights stay (they keep the model name); queued ones will fail."""
    row = weights_row(conn, key)
    conn.execute("DELETE FROM weights WHERE id = ?", (row["id"],))
    weights_path(paths, row).unlink(missing_ok=True)


@dataclass(frozen=True)
class ResolvedModel:
    key: str
    name: str
    kind: str  # "pipeline" | "weights"
    params: dict[str, Any]


def _same(a: Any, b: Any, spec_type: str | None = None) -> bool:
    """Whether a given value and the value actually used are the same (5 == 5.0, "5" == 5,
    a multiselect in any order)."""
    if spec_type == "multiselect" and isinstance(a, list) and isinstance(b, list):
        return sorted(map(str, a)) == sorted(map(str, b))
    if spec_type == "select":  # 5, 5.0 and "5" are the same option
        def norm(x: Any) -> str:
            if isinstance(x, bool):
                return str(x).lower()
            return str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)

        return norm(a) == norm(b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    try:
        if isinstance(a, (int, float, str)) and isinstance(b, (int, float)):
            return float(a) == float(b)
    except ValueError:
        return False
    return bool(a == b)


def check_params(conn: sqlite3.Connection, key: str, given: dict[str, Any]) -> dict[str, Any]:
    """Resolve params someone wrote by hand (or another program did) and say what happened to
    each: ``applied`` = params the input set, ``adjusted`` = those whose value had to change
    (clamped, coerced, filtered), ``ignored`` = input keys the model does not have. Older names
    a pipeline still understands (e.g. ``red_s_min``) count as applied under today's name."""
    resolved = resolve(conn, key, given)
    if weights_id(key) is not None:
        specs = weights_specs(weights_row(conn, key))
        migrated = dict(given)
    else:
        pipe = find_pipeline(key)
        specs, migrated = pipe.params, pipe.migrate(given)
    types = {s.name: s.type for s in specs}
    applied = [s.name for s in specs if s.name in migrated]
    return {
        "model_key": key,
        "params": resolved.params,
        "applied": applied,
        "ignored": [k for k in given if k not in types and k in migrated],
        "adjusted": [
            {"name": n, "given": migrated[n], "used": resolved.params[n]}
            for n in applied if not _same(migrated[n], resolved.params[n], types[n])
        ],
    }


def resolve(conn: sqlite3.Connection, key: str, given: dict[str, Any] | None) -> ResolvedModel:
    """Look a model up and resolve its params (defaults filled, numbers clamped)."""
    try:
        if weights_id(key) is not None:
            row = weights_row(conn, key)
            if row["state"] != "ready":
                raise LabError(400, f"사용할 수 없는 가중치입니다: {row['error'] or row['state']}")
            return ResolvedModel(key, row["name"], "weights", resolve_params(weights_specs(row), given))
        p = find_pipeline(key)
        return ResolvedModel(key, p.name, "pipeline", p.resolve(given))
    except ParamError as exc:
        raise LabError(400, f"파라미터 값이 올바르지 않습니다: {exc}") from exc
