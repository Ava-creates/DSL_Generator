"""Shared HTTP client for OpenAI-compatible API calls (Vulcan)."""

from __future__ import annotations

import os
import time
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.api_openai_compat_walltimes import (
    resolve_openai_compat_http_timeout,
    resolve_openai_compat_request_max_wait,
)
from src.utils.openai_compat_key import resolve_openai_compat_api_key
from src.utils.openai_compat_concurrency import openai_compat_api_slot


class OpenAICompatFatalError(RuntimeError):
    """Non-recoverable API connectivity failure (proxy block, misconfig, etc.)."""


def resolve_openai_compat_endpoint() -> tuple[str, str, str]:
    base_url = os.environ.get(
        "OPENAI_COMPAT_BASE_URL", "https://inference.vulcan.alliancecan.ca"
    ).rstrip("/")
    model = os.environ.get("OPENAI_COMPAT_MODEL", "gpt-oss-120b").strip()
    chat_path = os.environ.get(
        "OPENAI_COMPAT_CHAT_PATH", "/v1/chat/completions"
    ).strip()
    if chat_path.startswith("http://") or chat_path.startswith("https://"):
        endpoint = chat_path
    else:
        endpoint = f"{base_url}/{chat_path.lstrip('/')}"
    return endpoint, model, base_url


def no_proxy_hosts_for_endpoint(endpoint: str) -> str:
    host = urlparse(endpoint).hostname or ""
    existing = os.environ.get("NO_PROXY", os.environ.get("no_proxy", "")).strip()
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    for item in (host, ".alliancecan.ca", "inference.vulcan.alliancecan.ca"):
        if item and item not in parts:
            parts.append(item)
    return ",".join(parts)


def is_fatal_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.ProxyError):
        return True
    if isinstance(exc, requests.exceptions.ConnectionError):
        msg = str(exc)
        if "403 Forbidden" in msg or "Tunnel connection failed" in msg:
            return True
    return False


def build_openai_compat_session(*, retry_total: Optional[int] = None) -> requests.Session:
    """Session that uses cluster proxy env (if set) and retries transient HTTP failures."""
    total = retry_total if retry_total is not None else int(
        os.environ.get("OPENAI_COMPAT_MAX_RETRIES", "8")
    )
    backoff = float(os.environ.get("OPENAI_COMPAT_BACKOFF_FACTOR", "2.0"))
    retry = Retry(
        total=total,
        connect=0,
        read=total,
        status=total,
        backoff_factor=backoff,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
        raise_on_status=False,
    )
    session = requests.Session()
    # Compute nodes reach Vulcan via the cluster squid proxy; bypassing it (NO_PROXY +
    # trust_env=False) yields Errno 101 on many racks even when curl -4 works.
    session.trust_env = True
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def is_openai_compat_cold_start(status_code: int, body_text: str) -> bool:
    """True when the model/backend is warming up (retry, do not fail the job)."""
    if status_code in (502, 503, 504):
        return True
    if status_code >= 500:
        return True
    lower = (body_text or "").lower()
    if status_code == 400 and "server connection error" in lower:
        return True
    if "not ready" in lower:
        return True
    if status_code in (400, 503) and "warming" in lower:
        return True
    if status_code in (400, 503) and "loading" in lower and "model" in lower:
        return True
    return False


def _is_cold_start_response(status_code: int, body_text: str) -> bool:
    return is_openai_compat_cold_start(status_code, body_text)


def _raise_connection_error(label: str, exc: BaseException) -> None:
    if is_fatal_connection_error(exc):
        raise OpenAICompatFatalError(f"[{label}] {exc}") from exc
    raise exc


def post_chat_completion(
    *,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 10000,
    key_file: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    label: str = "OpenAICompat",
) -> tuple[int, str, Optional[dict[str, Any]]]:
    """POST to chat/completions with cluster proxy support and cold-start polling.

    Returns:
        (status_code, raw_text, parsed_json_or_none)

    Raises:
        OpenAICompatFatalError: proxy blocks and other non-recoverable connectivity errors.
    """
    endpoint, model, _ = resolve_openai_compat_endpoint()

    resolved_key = api_key or resolve_openai_compat_api_key(key_file)
    timeout = resolve_openai_compat_http_timeout(timeout_seconds)
    poll = float(os.environ.get("OPENAI_COMPAT_REQUEST_POLL", "10"))
    max_wait = resolve_openai_compat_request_max_wait()

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }

    session = build_openai_compat_session()
    deadline = time.monotonic() + max_wait
    attempt = 0

    with openai_compat_api_slot():
        while time.monotonic() < deadline:
            attempt += 1
            try:
                response = session.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
            except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError) as exc:
                _raise_connection_error(label, exc)

            body_text = response.text if response.text is not None else ""
            if response.status_code < 400:
                parsed = response.json()
                return response.status_code, body_text, parsed

            if _is_cold_start_response(response.status_code, body_text):
                remaining = deadline - time.monotonic()
                print(
                    f"[{label}] API not ready attempt {attempt}: "
                    f"HTTP {response.status_code}: {body_text[:200]}. "
                    f"Retrying in {poll:.0f}s ({remaining:.0f}s left)"
                )
                time.sleep(poll)
                continue

            return response.status_code, body_text, None

    raise OpenAICompatFatalError(
        f"[{label}] API request failed after {max_wait:.0f}s (endpoint={endpoint})"
    )


def extract_chat_content(body: Optional[dict[str, Any]]) -> str:
    if not body or not isinstance(body, dict):
        return ""
    choices = body.get("choices", [])
    if not choices:
        return ""
    first_choice = choices[0]
    message = first_choice.get("message")
    if isinstance(message, dict):
        content = message.get("content", "")
    elif isinstance(first_choice.get("text"), str):
        content = first_choice.get("text", "")
    else:
        content = ""
    return content if isinstance(content, str) else str(content)
