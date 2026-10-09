"""Small, environment-only settings module shared by remote Judge clients."""

from __future__ import annotations

import os


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


JUDGE_API_BASE = _env("AIEDU_JUDGE_API_BASE")
JUDGE_API_KEY = _env("AIEDU_JUDGE_API_KEY")
JUDGE_MODEL = _env("AIEDU_JUDGE_MODEL")
JUDGE_REQUEST_TIMEOUT = int(_env("AIEDU_JUDGE_TIMEOUT", "300"))
JUDGE_RATE_LIMIT_FIRST_BACKOFF = float(_env("AIEDU_JUDGE_RATE_LIMIT_FIRST_BACKOFF", "1"))
JUDGE_RATE_LIMIT_MAX_BACKOFF = float(_env("AIEDU_JUDGE_RATE_LIMIT_MAX_BACKOFF", "64"))
JUDGE_RATE_LIMIT_MAX_RETRIES = int(_env("AIEDU_JUDGE_RATE_LIMIT_MAX_RETRIES", "8"))
JUDGE_TRANSIENT_RETRIES = int(_env("AIEDU_JUDGE_TRANSIENT_RETRIES", "4"))
JUDGE_LOCAL_MODE = _env("AIEDU_JUDGE_LOCAL_MODE", "0") != "0"
JUDGE_ALLOW_EMPTY_API_KEY = _env("AIEDU_JUDGE_ALLOW_EMPTY_API_KEY", "0") != "0"
