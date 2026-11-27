"""Registry of built-in pipelines. The lab and the competition code look models up here.

Contract:
- list_pipelines() -> list[Pipeline] in display order
- get_pipeline(key) -> Pipeline, raises KeyError for an unknown key
Pipelines register themselves by being listed in cubevision/pipelines/__init__.py (ALL).
"""

from __future__ import annotations

from cubevision.pipelines.base import Pipeline


def list_pipelines() -> list[Pipeline]:
    from cubevision.pipelines import ALL

    return list(ALL)


def get_pipeline(key: str) -> Pipeline:
    for p in list_pipelines():
        if p.key == key:
            return p
    raise KeyError(key)
