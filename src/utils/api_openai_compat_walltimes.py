"""
Defaults for ``MODEL_TYPE=openai_compat`` (HTTP API): longer Slurm walltimes and
Python timeouts than local vLLM, since API latency is higher.

Slurm ``HH:MM:SS`` strings scale by 4× when *using_api* is true unless the caller
passes an explicit override. Per-timeout floats use :func:`env_float_openai_compat_scaled`:
if the env var is unset and ``MODEL_TYPE=openai_compat``, return *base_default* × 4.
"""

from __future__ import annotations

import os
from typing import Optional

_API_MULT = float(os.environ.get("OPENAI_COMPAT_RUNTIME_MULT", "4"))

# Hard cap on one post_chat_completion poll loop (FunSearch draw, synthesis, etc.).
OPENAI_COMPAT_CALL_MAX_WAIT_SECONDS = 1800.0  # 30 minutes (model cold start)

# Per-job sbatch --time suggestions for shell wrappers (export before sbatch).
API_SBATCH_TIME_ENV = {
    "TEST_TASKS_DEFAULT_TIME": "10:00:00",
    "TEST_TASKS_LONG_TIME": "10:00:00",
    "API_CHAIN_SBATCH_TIME_AGGREGATE": "4:00:00",
    "API_CHAIN_SBATCH_TIME_FILE_GEN": "10:00:00",
    "API_CHAIN_MEM": "64G",
    "API_CHAIN_SBATCH_TIME_IMPLEMENT_SINGLE": "20:00:00",
    "API_CHAIN_SBATCH_TIME_IMPLEMENT_ARRAY": "96:00:00",
    "API_CHAIN_SBATCH_TIME_FUNSEARCH": "40:00:00",
    "API_CHAIN_SBATCH_TIME_FUNSEARCH_SINGLE": "30:00:00",
    "API_CHAIN_SBATCH_TIME_EXPLICIT_FB": "24:00:00",
    "API_CHAIN_SBATCH_TIME_EVOLVE_DSL": "16:00:00",
    "API_CHAIN_SBATCH_TIME_EVOLVE_FUNC": "40:00:00",
    "API_CHAIN_SBATCH_TIME_GET_CFG": "1:00:00",
}


def _is_openai_compat_mode() -> bool:
    return os.environ.get("MODEL_TYPE", "").strip() == "openai_compat"


def env_float_openai_compat_scaled(env_var: str, base_default: float) -> float:
    """Env float if set; else *base_default* or *base_default* × mult in API mode."""
    raw = os.environ.get(env_var, "").strip()
    if raw:
        return float(raw)
    b = base_default * _API_MULT if _is_openai_compat_mode() else base_default
    return float(b)


def resolve_openai_compat_http_timeout(
    explicit_seconds: Optional[float] = None,
    *,
    base_default: float = 120.0,
) -> float:
    """Per-request HTTP read timeout, capped at :data:`OPENAI_COMPAT_CALL_MAX_WAIT_SECONDS`."""
    if explicit_seconds is not None:
        return min(float(explicit_seconds), OPENAI_COMPAT_CALL_MAX_WAIT_SECONDS)
    return min(
        env_float_openai_compat_scaled("OPENAI_COMPAT_HTTP_TIMEOUT", base_default),
        OPENAI_COMPAT_CALL_MAX_WAIT_SECONDS,
    )


def resolve_openai_compat_request_max_wait() -> float:
    """Total wall-clock budget for one API call (retries + cold-start polling)."""
    return min(
        env_float_openai_compat_scaled("OPENAI_COMPAT_REQUEST_MAX_WAIT", 450.0),
        OPENAI_COMPAT_CALL_MAX_WAIT_SECONDS,
    )


def _parse_slurm_hhmmss(s: str) -> int:
    s = s.strip()
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
        return h * 3600 + m * 60 + sec
    if len(parts) == 2:
        m, sec = int(parts[0]), int(parts[1])
        return m * 60 + sec
    raise ValueError(f"unsupported Slurm time format: {s!r}")


def _format_slurm_hhmmss(total_seconds: int) -> str:
    total_seconds = max(0, int(total_seconds))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    sec = total_seconds % 60
    return f"{h:d}:{m:02d}:{sec:02d}"


def scale_slurm_walltime_for_api(default_hhmmss: str, *, using_api: bool) -> str:
    """Return *default_hhmmss* or the same wall duration × ``OPENAI_COMPAT_RUNTIME_MULT`` when API."""
    if not using_api:
        return default_hhmmss
    secs = _parse_slurm_hhmmss(default_hhmmss)
    return _format_slurm_hhmmss(int(secs * _API_MULT))


def export_api_walltime_defaults() -> None:
    """Set ``os.environ`` Slurm suggestion keys for API mode without clobbering user exports."""
    if not _is_openai_compat_mode():
        return
    for key, default in API_SBATCH_TIME_ENV.items():
        if not str(os.environ.get(key, "")).strip():
            os.environ[key] = default


def print_export_shell() -> None:
    """Print ``export KEY=val`` lines for copy-paste into bash (API mode defaults)."""
    lines = ["# When MODEL_TYPE=openai_compat, optional Slurm walltime exports (4× typical GPU defaults)"]
    for key, default in API_SBATCH_TIME_ENV.items():
        lines.append(f'export {key}="${{{key}:-{default}}}"')
    print("\n".join(lines))


if __name__ == "__main__":
    print_export_shell()
