"""lab.cache.Lru: bounded LRU order, stamps, and single-flight get_or_make."""

from __future__ import annotations

import threading
import time

import pytest

from lab.cache import Lru


def test_lru_order_bound_and_stamps() -> None:
    c: Lru[str, int] = Lru(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1  # "a" is now the newest
    c.put("c", 3)
    assert (c.get("b"), c.get("a"), c.get("c"), len(c)) == (None, 1, 3, 2)
    assert c.pop("a") == 1 and c.pop("a") is None and len(c) == 1
    # A stamp names what the value was made from: another stamp is a miss and replaces the entry.
    c.put("k", 10, stamp="v1")
    assert c.get("k", "v1") == 10 and c.get("k", "v2") is None and c.get("k") is None
    assert c.get_or_make("k", lambda: 11, stamp="v2") == 11 and c.get("k", "v1") is None and len(c) == 2
    c.clear()
    assert len(c) == 0


def test_get_or_make_is_single_flight() -> None:
    """Eight threads asking for one missing key at once: it is made once and all get that value."""
    c: Lru[str, object] = Lru(4)
    calls = []
    start = threading.Barrier(8)

    def make() -> object:
        calls.append(1)
        time.sleep(0.1)
        return object()

    got: list[object] = []

    def ask() -> None:
        start.wait()
        got.append(c.get_or_make("k", make))

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and len(got) == 8 and all(v is got[0] for v in got)


def test_get_or_make_shares_a_failure_and_tries_again_later() -> None:
    c: Lru[str, int] = Lru(4)
    start = threading.Barrier(3)
    errors: list[BaseException] = []

    def fail() -> int:
        time.sleep(0.1)
        raise ValueError("boom")

    def ask() -> None:
        start.wait()
        try:
            c.get_or_make("k", fail)
        except ValueError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=ask) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(errors) == 3 and len(c) == 0
    with pytest.raises(ValueError):
        c.get_or_make("k", fail)
    assert c.get_or_make("k", lambda: 5) == 5, "a failure is not cached"
