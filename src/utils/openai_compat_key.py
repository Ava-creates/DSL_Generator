"""Resolve OpenAI-compatible API key: env OPENAI_COMPAT_API_KEY, else a key file."""

from __future__ import annotations

import os
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_openai_compat_api_key(key_file: Optional[str] = None) -> str:
    """Return API key for OpenAI-compatible endpoints.

    Precedence:
    1. Environment variable ``OPENAI_COMPAT_API_KEY`` (non-empty after strip).
    2. First non-empty, non-``#`` line from a file:
       - ``key_file`` argument if given (absolute path, or relative to project root);
       - else environment ``OPENAI_COMPAT_KEY_FILE`` (same path rules);
       - else ``<project root>/key.txt``.

    Raises:
        FileNotFoundError: if the key file is missing.
        ValueError: if no key is found in the file and env is unset.
    """
    env = os.environ.get("OPENAI_COMPAT_API_KEY", "").strip()
    if env:
        return env

    if key_file:
        path = key_file if os.path.isabs(key_file) else os.path.join(_PROJECT_ROOT, key_file)
    else:
        env_path = os.environ.get("OPENAI_COMPAT_KEY_FILE", "").strip()
        if env_path:
            path = env_path if os.path.isabs(env_path) else os.path.join(_PROJECT_ROOT, env_path)
        else:
            path = os.path.join(_PROJECT_ROOT, "key.txt")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                return s

    raise ValueError(
        f"No API key found in {path} (set OPENAI_COMPAT_API_KEY or add a key line to this file)."
    )
