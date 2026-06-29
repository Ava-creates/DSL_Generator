"""Wait for an OpenAI-compatible API model to finish cold start before batch jobs."""

from __future__ import annotations

import argparse
import os
from typing import Optional

from src.utils.api_openai_compat_walltimes import env_float_openai_compat_scaled
from src.utils.openai_compat_http import post_chat_completion, resolve_openai_compat_endpoint


def cold_start_openai_compat(
    *,
    key_file: Optional[str] = None,
    max_wait_seconds: Optional[float] = None,
    poll_interval_seconds: Optional[float] = None,
    request_timeout_seconds: Optional[float] = None,
) -> None:
    """Poll until a minimal chat completion succeeds (model loaded)."""
    if max_wait_seconds is not None:
        os.environ["OPENAI_COMPAT_REQUEST_MAX_WAIT"] = str(max_wait_seconds)
    if poll_interval_seconds is not None:
        os.environ["OPENAI_COMPAT_REQUEST_POLL"] = str(poll_interval_seconds)

    endpoint, model, _ = resolve_openai_compat_endpoint()
    timeout = (
        request_timeout_seconds
        if request_timeout_seconds is not None
        else env_float_openai_compat_scaled("OPENAI_COMPAT_COLD_START_TIMEOUT", 120.0)
    )
    max_wait = (
        max_wait_seconds
        if max_wait_seconds is not None
        else env_float_openai_compat_scaled("OPENAI_COMPAT_COLD_START_MAX_WAIT", 600.0)
    )
    poll = (
        poll_interval_seconds
        if poll_interval_seconds is not None
        else float(os.environ.get("OPENAI_COMPAT_COLD_START_POLL", "15"))
    )

    print(
        f"[cold_start] Waiting for model {model!r} at {endpoint} "
        f"(max {max_wait:.0f}s, poll every {poll:.0f}s)"
    )

    status, body_text, parsed = post_chat_completion(
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        temperature=0,
        max_tokens=8,
        key_file=key_file,
        timeout_seconds=timeout,
        label="cold_start",
    )
    if status < 400 and parsed is not None:
        print("[cold_start] Model ready")
        return

    raise TimeoutError(
        f"Model {model!r} did not become ready (last status={status}, body={body_text[:500]})"
    )


def maybe_cold_start_openai_compat(*, key_file: Optional[str] = None) -> None:
    """Run cold start when enabled for openai_compat mode."""
    if os.environ.get("OPENAI_COMPAT_SKIP_COLD_START", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        print("[cold_start] Skipped (OPENAI_COMPAT_SKIP_COLD_START is set)")
        return
    if os.environ.get("MODEL_TYPE", "").strip() != "openai_compat":
        return
    cold_start_openai_compat(key_file=key_file)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for Vulcan / OpenAI-compatible model cold start"
    )
    parser.add_argument(
        "--key-file",
        default=None,
        help="API key file (default: OPENAI_COMPAT_KEY_FILE or repo key.txt)",
    )
    args = parser.parse_args()
    cold_start_openai_compat(key_file=args.key_file)
    print("[cold_start] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
