"""Receive a multipart upload straight into data/tmp while hashing it.

Starlette's form parser first spools files into the system temp directory; for videos of
several GB that doubles the disk traffic. Here the request body is parsed as it streams in
and the file part is written once, to its final volume, with its sha256 computed on the way.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from fastapi import Request
from python_multipart.exceptions import FormParserError
from python_multipart.multipart import MultipartParser, parse_options_header

from lab.errors import LabError
from lab.text import nfc

MAX_FIELD_BYTES = 64 * 1024


@dataclass
class Upload:
    path: Path  # temporary file in data/tmp; the caller moves or deletes it
    filename: str  # client file name (NFC, no directories)
    sha256: str
    bytes: int
    fields: dict[str, str] = field(default_factory=dict)  # the non-file form fields

    @property
    def ext(self) -> str:
        return Path(self.filename).suffix.lower()

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)


def size_text(n: int) -> str:
    """'4 GB', '50 MB': the limit in a 413 message."""
    return f"{n / 1024**3:.0f} GB" if n >= 1024**3 else f"{n / 1024**2:.0f} MB"


def clean_filename(raw: str) -> str:
    name = nfc(Path(raw.replace("\\", "/")).name).strip()
    return name or "upload"


class _Receiver:
    def __init__(self, tmp_dir: Path, file_field: str, max_bytes: int):
        self.tmp_dir = tmp_dir
        self.file_field = file_field
        self.max_bytes = max_bytes
        self.fields: dict[str, str] = {}
        self.filename: str | None = None
        self.path: Path | None = None
        self.size = 0
        self.hasher = hashlib.sha256()
        self._out: BinaryIO | None = None
        self._header_name = b""
        self._header_value = b""
        self._disposition = b""
        self._part_name = ""
        self._part_is_file = False
        self._field_data = bytearray()

    # python-multipart callbacks -------------------------------------------------------------
    def on_part_begin(self) -> None:
        self._disposition = b""
        self._field_data = bytearray()

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_name += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def on_header_end(self) -> None:
        if self._header_name.lower() == b"content-disposition":
            self._disposition = self._header_value
        self._header_name = b""
        self._header_value = b""

    def on_headers_finished(self) -> None:
        _, options = parse_options_header(self._disposition)
        self._part_name = options.get(b"name", b"").decode("utf-8", "replace")
        self._part_is_file = b"filename" in options
        if self._part_is_file and self._part_name == self.file_field:
            if self.path is not None:
                raise LabError(400, "파일은 한 번에 하나만 올릴 수 있습니다")
            self.filename = clean_filename(options[b"filename"].decode("utf-8", "replace"))
            suffix = Path(self.filename).suffix.lower()
            if not suffix.isascii() or len(suffix) > 12:
                suffix = ""
            self.path = self.tmp_dir / f"upload-{uuid.uuid4().hex}{suffix}"
            self._out = open(self.path, "wb")

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        chunk = data[start:end]
        if self._out is not None and self._part_name == self.file_field and self._part_is_file:
            self.size += len(chunk)
            if self.size > self.max_bytes:
                raise LabError(413, f"파일이 너무 큽니다 (최대 {size_text(self.max_bytes)})")
            self.hasher.update(chunk)
            self._out.write(chunk)
        elif not self._part_is_file:
            if len(self._field_data) + len(chunk) > MAX_FIELD_BYTES:
                raise LabError(413, "폼 값이 너무 깁니다")
            self._field_data.extend(chunk)

    def on_part_end(self) -> None:
        if self._part_is_file and self._out is not None and self._part_name == self.file_field:
            self._out.close()
            self._out = None
        elif not self._part_is_file:
            self.fields[self._part_name] = self._field_data.decode("utf-8", "replace")

    def close(self) -> None:
        if self._out is not None:
            self._out.close()
            self._out = None


async def receive_upload(request: Request, tmp_dir: Path, *, max_bytes: int, file_field: str = "file") -> Upload:
    ctype = request.headers.get("content-type", "")
    _, params = parse_options_header(ctype)
    boundary = params.get(b"boundary")
    if not ctype.lower().startswith("multipart/form-data") or not boundary:
        raise LabError(400, "multipart/form-data 형식으로 파일을 보내세요")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes + MAX_FIELD_BYTES:
        raise LabError(413, f"파일이 너무 큽니다 (최대 {size_text(max_bytes)})")

    rx = _Receiver(tmp_dir, file_field, max_bytes)
    callbacks = {name: getattr(rx, name) for name in (
        "on_part_begin", "on_part_data", "on_part_end", "on_header_field",
        "on_header_value", "on_header_end", "on_headers_finished",
    )}
    parser = MultipartParser(boundary, callbacks)  # type: ignore[arg-type]
    try:
        async for chunk in request.stream():
            parser.write(chunk)
        parser.finalize()
    except (FormParserError, LabError) as exc:
        rx.close()
        if rx.path:
            rx.path.unlink(missing_ok=True)
        if isinstance(exc, LabError):
            raise
        raise LabError(400, "업로드 내용을 해석하지 못했습니다") from exc
    except BaseException:
        rx.close()
        if rx.path:
            rx.path.unlink(missing_ok=True)
        raise
    rx.close()
    if rx.path is None or rx.filename is None:
        raise LabError(400, f"'{file_field}' 필드에 파일이 없습니다")
    if rx.size == 0:
        rx.path.unlink(missing_ok=True)
        raise LabError(400, "빈 파일입니다")
    return Upload(path=rx.path, filename=rx.filename, sha256=rx.hasher.hexdigest(), bytes=rx.size, fields=rx.fields)
