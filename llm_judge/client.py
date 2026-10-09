"""OpenAI-compatible chat client with error-aware retry strategy.

Error classification & retry:
  - Auth/Billing (401/402/403) → raise immediately, no retry
  - Rate Limit (429) → exponential backoff: 1s→2s→4s→8s→...→64s (max)
  - Transient (connection/timeout/500) → short backoff retry
  - local_mode=True → all errors raise immediately
  - All retries exhausted → raise JudgeRateLimitError / JudgeAPIError
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

try:
    import openai
except ImportError:  # Allow --help/--dry-run without openai installed
    openai = None  # type: ignore[assignment]

from .text import strip_think_content

logger = logging.getLogger(__name__)


# ── Custom exceptions ──────────────────────────────────────────


class JudgeAuthError(RuntimeError):
    """Auth/billing/permission error (do not retry)."""

    pass


class JudgeRateLimitError(RuntimeError):
    """Rate limit exhausted after retries."""

    pass


class JudgeAPIError(RuntimeError):
    """Other unrecoverable API error."""

    pass


# ── Error classification helpers ───────────────────────────────

_ERROR_CATEGORIES: dict[str, type] = {}


def _classify_error(exc: Exception) -> type:
    if openai is None:
        return JudgeAPIError

    error_code = _try_extract_error_code(exc)
    status = _try_extract_status(exc)
    cache_key = f"{type(exc).__name__}:{status}:{error_code}"
    if cache_key in _ERROR_CATEGORIES:
        return _ERROR_CATEGORIES[cache_key]

    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        result = JudgeAuthError
    elif isinstance(exc, openai.BadRequestError):
        # 402 Payment Required sometimes wrapped in BadRequest
        result = JudgeAuthError if status == 402 else JudgeAPIError
    elif isinstance(exc, openai.RateLimitError):
        if error_code in {"insufficient_quota", "billing_not_active", "quota_exceeded"}:
            result = JudgeAuthError
        else:
            result = JudgeRateLimitError
    elif isinstance(exc, openai.APIConnectionError):
        result = JudgeRateLimitError  # network jitter
    elif isinstance(exc, (openai.APITimeoutError, openai.InternalServerError)):
        result = JudgeRateLimitError
    elif isinstance(exc, openai.APIStatusError):
        if status == 429 and error_code in {"insufficient_quota", "billing_not_active", "quota_exceeded"}:
            result = JudgeAuthError
        elif status in (429, 502, 503):
            result = JudgeRateLimitError
        elif status in (401, 402, 403):
            result = JudgeAuthError
        else:
            result = JudgeAPIError
    else:
        result = JudgeAPIError

    return _ERROR_CATEGORIES.setdefault(cache_key, result)


def _try_extract_status(exc: Exception) -> int | None:
    try:
        return exc.status_code  # type: ignore[union-attr]
    except AttributeError:
        pass
    try:
        return int(getattr(exc, "code", 0))  # type: ignore[arg-type]
    except (ValueError, TypeError, AttributeError):
        pass
    return None


def _try_extract_error_code(exc: Exception) -> str | None:
    try:
        body = getattr(exc, "body", None) or {}
        if isinstance(body, dict):
            return body.get("code")
    except AttributeError:
        pass
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    return _classify_error(exc) == JudgeRateLimitError


def _is_auth_error(exc: Exception) -> bool:
    return _classify_error(exc) == JudgeAuthError


def _build_error_message(category: type, exc: Exception, detail: str = "") -> str:
    base = {
        JudgeAuthError: "Judge API auth/billing error",
        JudgeRateLimitError: "Judge API rate limit exhausted",
        JudgeAPIError: "Judge API unrecoverable error",
    }.get(category, "Judge API unknown error")
    parts = [f"[{category.__name__}] {base}"]
    if detail:
        parts.append(detail)
    exc_info = str(exc)[:400]
    if exc_info:
        parts.append(f"Original error: {exc_info}")
    return " | ".join(parts)


# ── Exponential backoff ────────────────────────────────────────


def _exponential_backoff(attempt: int, first_backoff: float, max_backoff: float) -> float:
    wait = first_backoff * (2**attempt)
    return min(wait, max_backoff)


# ── OpenAI Compatible Client ───────────────────────────────────


class OpenAICompatibleClient:
    def __init__(
        self,
        api_base: str | None = None,
        base_url: str = "",
        api_key: str = "",
        timeout: int = 3000,
        local_mode: bool = False,
        rate_limit_first_backoff: float = 1.0,
        rate_limit_max_backoff: float = 64.0,
        rate_limit_max_retries: int = 8,
        transient_retries: int = 4,
    ) -> None:
        self.base_url = str(api_base or base_url or "").rstrip("/")
        self.api_key = str(api_key or "")
        self.timeout = int(timeout)
        self.local_mode = bool(local_mode)
        self.rate_limit_first_backoff = float(rate_limit_first_backoff)
        self.rate_limit_max_backoff = float(rate_limit_max_backoff)
        self.rate_limit_max_retries = int(rate_limit_max_retries)
        self.transient_retries = int(transient_retries)
        self._client_lock = threading.Lock()
        self._client = None

    def _require_client(self):
        if not self.base_url:
            raise ValueError("base_url is empty")
        if not self.api_key:
            raise ValueError("api_key is empty")
        with self._client_lock:
            if self._client is None:
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise JudgeAPIError("OpenAI SDK not installed. Please install: pip install openai") from exc

                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    max_retries=0,  # We control retries ourselves
                )
        return self._client

    def _execute_with_retry(self, api_call_fn, **resolve_kw) -> Any:
        """Unified error-aware retry executor."""
        if self.local_mode:
            try:
                return api_call_fn()
            except Exception as exc:
                raise JudgeAPIError(
                    _build_error_message(JudgeAPIError, exc, "local_mode=True raises immediately")
                ) from exc

        last_exc: Exception | None = None
        rate_limit_attempts = 0
        transient_attempts = 0

        while True:
            try:
                return api_call_fn()
            except Exception as exc:
                last_exc = exc

                if _is_auth_error(exc):
                    raise JudgeAuthError(_build_error_message(JudgeAuthError, exc)) from exc

                if _is_rate_limit_error(exc):
                    rate_limit_attempts += 1
                    if rate_limit_attempts > self.rate_limit_max_retries:
                        raise JudgeRateLimitError(
                            _build_error_message(
                                JudgeRateLimitError,
                                exc,
                                f"Rate limited after {self.rate_limit_max_retries} retries",
                            )
                        ) from exc
                    wait = _exponential_backoff(
                        rate_limit_attempts, self.rate_limit_first_backoff, self.rate_limit_max_backoff
                    )
                    logger.warning(
                        "Rate limit (attempt %d/%d), waiting %.0fs",
                        rate_limit_attempts,
                        self.rate_limit_max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                # Other transient errors (connection/timeout/500)
                transient_attempts += 1
                if transient_attempts > self.transient_retries:
                    raise JudgeAPIError(
                        _build_error_message(
                            JudgeAPIError,
                            last_exc,
                            f"Transient error after {self.transient_retries} retries",
                        )
                    ) from last_exc
                wait = min(2**transient_attempts, 60.0)
                logger.warning(
                    "Transient error (attempt %d/%d), waiting %.0fs: %s",
                    transient_attempts,
                    self.transient_retries,
                    wait,
                    str(exc)[:200],
                )
                time.sleep(wait)

    def chat_complete_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float | None = None,
        response_format: dict | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Chat completion returning plain text."""
        client = self._require_client()

        def _call():
            payload: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            }
            if response_format is not None:
                payload["response_format"] = response_format
            if top_p is not None:
                payload["top_p"] = float(top_p)
            if extra_body is not None:
                payload["extra_body"] = extra_body
            return client.chat.completions.create(**payload)

        response = self._execute_with_retry(_call)
        return self._extract_message_text(response)

    @staticmethod
    def _extract_message_text(response) -> str:
        """Extract text from OpenAI completion response."""
        try:
            choice = response.choices[0]
        except Exception as exc:
            raise ValueError(f"Invalid completion response: {response}") from exc

        def _choice_debug_info() -> str:
            try:
                finish_reason = getattr(choice, "finish_reason", None)
                message = getattr(choice, "message", None)
                content = getattr(message, "content", None) if message is not None else None
                return f"finish_reason={finish_reason!r}, content={content!r}"
            except Exception:
                return repr(choice)

        message = getattr(choice, "message", None)
        if message is not None:
            refusal = getattr(message, "refusal", None)
            if refusal is not None:
                raise JudgeAPIError(f"[{JudgeAPIError.__name__}] Model refused: {str(refusal)[:200]}")
            content = getattr(message, "content", None)
            if isinstance(content, str):
                text = strip_think_content(content).strip()
                if text:
                    return text
                raise ValueError(f"Empty assistant content: {_choice_debug_info()}")

        text = getattr(choice, "text", None)
        if isinstance(text, str):
            stripped = strip_think_content(text).strip()
            if stripped:
                return stripped
            raise ValueError(f"Empty assistant content: {_choice_debug_info()}")

        raise ValueError(f"Cannot extract text from completion response: {_choice_debug_info()}")


__all__ = [
    "OpenAICompatibleClient",
    "JudgeAuthError",
    "JudgeRateLimitError",
    "JudgeAPIError",
]
