"""Cluster-wide limit on concurrent OpenAI-compatible API callers."""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from typing import Iterator, Optional

_HELD_FD: Optional[object] = None


def _max_parallel() -> int:
    raw = os.environ.get("OPENAI_COMPAT_MAX_PARALLEL", "").strip()
    if not raw:
        return 0
    return max(0, int(raw))


def _slot_dir() -> str:
    return os.environ.get(
        "OPENAI_COMPAT_SLOT_DIR",
        os.path.join(os.environ.get("TMPDIR", "/tmp"), "openai_compat_api_slots"),
    )


@contextmanager
def openai_compat_api_slot() -> Iterator[None]:
    """Block until a concurrency slot is free, then hold it for one API call."""
    global _HELD_FD
    max_n = _max_parallel()
    if max_n <= 0:
        yield
        return

    slot_dir = _slot_dir()
    os.makedirs(slot_dir, exist_ok=True)
    poll = float(os.environ.get("OPENAI_COMPAT_SLOT_POLL", "5"))

    fd = None
    while fd is None:
        for idx in range(max_n):
            path = os.path.join(slot_dir, f"slot_{idx}.lock")
            candidate = open(path, "a+")
            try:
                fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                candidate.close()
                continue
            fd = candidate
            break
        if fd is None:
            time.sleep(poll)

    _HELD_FD = fd
    try:
        yield
    finally:
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        fd.close()
        _HELD_FD = None
