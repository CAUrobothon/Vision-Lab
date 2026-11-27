"""FastAPI dependencies shared by the routers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request

from lab.config import DataPaths
from lab.db import connect
from lab.supervisor import WorkerSupervisor


def get_paths(request: Request) -> DataPaths:
    return request.app.state.paths


def get_conn(request: Request) -> Iterator[sqlite3.Connection]:
    conn = connect(request.app.state.paths.db)
    try:
        yield conn
    finally:
        conn.close()


def get_supervisor(request: Request) -> WorkerSupervisor:
    return request.app.state.supervisor


Conn = Annotated[sqlite3.Connection, Depends(get_conn)]
Paths = Annotated[DataPaths, Depends(get_paths)]
Supervisor = Annotated[WorkerSupervisor, Depends(get_supervisor)]
