"""A small bounded LRU map for per-process caches (parsed GT sets, scores).

Thread-safe. ``get_or_make`` is single-flight: while one thread makes the value of a key, other
threads asking for the same key wait for it instead of making it again (they also get its
exception). An entry may carry a ``stamp`` (what it was made from, e.g. a fingerprint of the
runs): a lookup with another stamp is a miss, and the new value replaces the old one, so a key
holds one entry however often its inputs change.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from concurrent.futures import Future
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class Lru(Generic[K, V]):
    def __init__(self, size: int) -> None:
        self.size = size
        self._data: OrderedDict[K, tuple[Hashable, V]] = OrderedDict()
        self._flights: dict[tuple[K, Hashable], Future[V]] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._data)

    def _hit(self, key: K, stamp: Hashable) -> tuple[bool, V | None]:
        entry = self._data.get(key)
        if entry is None or entry[0] != stamp:
            return False, None
        self._data.move_to_end(key)
        return True, entry[1]

    def _store(self, key: K, value: V, stamp: Hashable) -> None:
        self._data[key] = (stamp, value)
        self._data.move_to_end(key)
        while len(self._data) > self.size:
            self._data.popitem(last=False)

    def get(self, key: K, stamp: Hashable = None) -> V | None:
        with self._lock:
            return self._hit(key, stamp)[1]

    def put(self, key: K, value: V, stamp: Hashable = None) -> None:
        with self._lock:
            self._store(key, value, stamp)

    def pop(self, key: K) -> V | None:
        with self._lock:
            entry = self._data.pop(key, None)
        return None if entry is None else entry[1]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def get_or_make(self, key: K, make: Callable[[], V], stamp: Hashable = None) -> V:
        """The value of ``key`` (made with ``stamp``), made once by ``make`` if missing."""
        with self._lock:
            found, value = self._hit(key, stamp)
            if found:
                return value  # type: ignore[return-value]
            flight = self._flights.get((key, stamp))
            owner = flight is None
            if owner:
                flight = self._flights[(key, stamp)] = Future()
        assert flight is not None
        if not owner:
            return flight.result()
        try:
            value = make()
        except BaseException as exc:
            with self._lock:
                del self._flights[(key, stamp)]
            flight.set_exception(exc)
            raise
        with self._lock:
            self._store(key, value, stamp)
            del self._flights[(key, stamp)]
        flight.set_result(value)
        return value
