"""Every error leaves the API as {"detail": "<Korean sentence>"}."""

from __future__ import annotations

import http

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class LabError(Exception):
    """An expected failure with an HTTP status and a Korean message for the user."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


# Korean replacements for Starlette's stock English messages (unknown route, wrong method, ...).
_GENERIC = {
    400: "요청이 올바르지 않습니다",
    404: "찾을 수 없습니다",
    405: "허용되지 않는 요청 방식입니다",
    413: "파일이 너무 큽니다",
    415: "지원하지 않는 형식입니다",
}


def _error(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


async def _lab_error(_: Request, exc: LabError) -> JSONResponse:
    return _error(exc.status, exc.detail)


async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail
    try:
        stock = http.HTTPStatus(exc.status_code).phrase
    except ValueError:
        stock = ""
    if not isinstance(detail, str) or detail == stock:
        detail = _GENERIC.get(exc.status_code, "요청을 처리하지 못했습니다")
    return JSONResponse({"detail": detail}, status_code=exc.status_code, headers=getattr(exc, "headers", None))


async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    parts = []
    for err in exc.errors()[:3]:
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        parts.append(f"{loc}: {err.get('msg', '')}" if loc else str(err.get("msg", "")))
    return _error(400, "요청 값이 올바르지 않습니다 (" + "; ".join(parts) + ")")


async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
    # Starlette re-raises the exception after this response is sent, so uvicorn prints the
    # traceback to stderr once; logging it here as well would only duplicate it.
    return _error(500, "서버 내부 오류가 났습니다. 서버 로그를 확인하세요.")


def install(app: FastAPI) -> None:
    app.add_exception_handler(LabError, _lab_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unexpected)
