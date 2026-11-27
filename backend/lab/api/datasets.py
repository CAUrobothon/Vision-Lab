"""/api/datasets (named groups of media with folders, their originals as a zip, batch runs) and /api/batches."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from lab import datasets as ds_mod
from lab import gtsets
from lab.deps import Conn, Paths
from lab.schemas import (
    Batch, BatchCreate, BatchDetail, Dataset, DatasetCreate, DatasetMediaChange, DatasetMediaMove, DatasetPatch,
    FolderCreate, FolderDelete, FolderRename, Ok,
)

router = APIRouter(prefix="/api", tags=["datasets"])


@router.get("/datasets")
def list_datasets(conn: Conn) -> list[Dataset]:
    return ds_mod.list_datasets(conn)


@router.post("/datasets")
def create_dataset(req: DatasetCreate, conn: Conn) -> Dataset:
    return ds_mod.create(conn, req.name, req.note, req.media_ids)


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: int, conn: Conn) -> Dataset:
    return ds_mod.get(conn, dataset_id)


@router.patch("/datasets/{dataset_id}")
def patch_dataset(dataset_id: int, patch: DatasetPatch, conn: Conn) -> Dataset:
    return ds_mod.patch(conn, dataset_id, patch.name, patch.note)


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: int, conn: Conn, paths: Paths) -> Ok:
    """The grouping, its batch records and its ground-truth sets (rows and files); the media and
    every run stay."""
    gt_ids = gtsets.set_ids_of_dataset(conn, dataset_id)
    ds_mod.delete(conn, dataset_id)  # the gt rows go with the dataset (ON DELETE CASCADE)
    gtsets.forget(paths, gt_ids)
    return Ok()


@router.post("/datasets/{dataset_id}/media")
def add_media(dataset_id: int, req: DatasetMediaChange, conn: Conn) -> Dataset:
    return ds_mod.add_media(conn, dataset_id, req.media_ids)


@router.post("/datasets/{dataset_id}/media/remove")
def remove_media(dataset_id: int, req: DatasetMediaChange, conn: Conn) -> Dataset:
    """Takes the media out of this dataset; the media themselves stay."""
    return ds_mod.remove_media(conn, dataset_id, req.media_ids)


@router.post("/datasets/{dataset_id}/media/move")
def move_media(dataset_id: int, req: DatasetMediaMove, conn: Conn) -> Dataset:
    """Moves media of this dataset into another of its folders ('' = the top level)."""
    return ds_mod.move_media(conn, dataset_id, req.media_ids, req.folder)


@router.post("/datasets/{dataset_id}/folders")
def create_folder(dataset_id: int, req: FolderCreate, conn: Conn) -> Dataset:
    """An empty folder, e.g. "증강/어둡게" (missing parents are made too)."""
    return ds_mod.create_folder(conn, dataset_id, req.path)


@router.post("/datasets/{dataset_id}/folders/rename")
def rename_folder(dataset_id: int, req: FolderRename, conn: Conn) -> Dataset:
    """Renames or moves a folder with its subfolders and media."""
    return ds_mod.rename_folder(conn, dataset_id, req.path, req.new_path)


@router.post("/datasets/{dataset_id}/folders/delete")
def delete_folder(dataset_id: int, req: FolderDelete, conn: Conn) -> Dataset:
    """Deletes a folder and its subfolders; their media leave this dataset only (files and runs stay)."""
    return ds_mod.delete_folder(conn, dataset_id, req.path)


@router.get("/datasets/{dataset_id}/originals.zip")
def originals_zip(dataset_id: int, conn: Conn, paths: Paths) -> FileResponse:
    path, filename = ds_mod.originals_zip(conn, paths, dataset_id)
    return FileResponse(path, filename=filename, media_type="application/zip", content_disposition_type="attachment")


@router.get("/datasets/{dataset_id}/batches")
def list_batches(dataset_id: int, conn: Conn) -> list[Batch]:
    return ds_mod.list_batches(conn, dataset_id)


@router.post("/datasets/{dataset_id}/batches")
def create_batch(dataset_id: int, req: BatchCreate, conn: Conn, paths: Paths) -> BatchDetail:
    """One queued run per media of the dataset (or of one folder with its subfolders), all with
    the same model and params."""
    return ds_mod.create_batch(conn, paths, dataset_id, req)


@router.get("/batches/{batch_id}")
def get_batch(batch_id: int, conn: Conn, paths: Paths) -> BatchDetail:
    return ds_mod.get_batch(conn, paths, batch_id)


@router.post("/batches/{batch_id}/cancel")
def cancel_batch(batch_id: int, conn: Conn, paths: Paths) -> BatchDetail:
    return ds_mod.cancel_batch(conn, paths, batch_id)


@router.delete("/batches/{batch_id}")
def delete_batch(batch_id: int, conn: Conn, paths: Paths) -> Ok:
    """The batch and all its runs."""
    ds_mod.delete_batch(conn, paths, batch_id)
    return Ok()


@router.get("/batches/{batch_id}/results.zip")
def results_zip(batch_id: int, conn: Conn, paths: Paths) -> FileResponse:
    """Drawn results of the finished runs (PNG / MP4) + results.json + results.csv. Synchronous:
    video results that were never exported are encoded first."""
    path, filename = ds_mod.results_zip(conn, paths, batch_id)
    return FileResponse(path, filename=filename, media_type="application/zip", content_disposition_type="attachment")
