"""Shared wall-clock timeout helper (SIGALRM + setitimer, main thread on Unix)."""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional


@contextmanager
def wall_clock_timeout(
    timeout_seconds: Optional[float],
    *,
    timeout_message: str,
) -> Iterator[None]:
    """Best-effort wall-clock timeout guard (main thread on Unix).

    If ``timeout_seconds`` is None or non-positive, or the runtime cannot use
    ``setitimer`` (non-main thread, or missing API), this is a no-op context.
    """
    if timeout_seconds is None or timeout_seconds <= 0:
        yield
        return

    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        # Fallback: no hard timeout support in this execution context.
        yield
        return

    def _handle_timeout(_signum, _frame):
        raise TimeoutError(timeout_message)

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
