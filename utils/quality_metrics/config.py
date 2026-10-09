"""Runtime configuration for the Question quality parser and QDR evaluator."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ACTIVE_OUTPUT_DIR = PROJECT_ROOT / os.environ.get("AIEDU_QUALITY_OUTPUT_DIR", "results/question_quality")


@dataclass(frozen=True)
class GroupConfig:
    """One input group and its on-disk layout."""

    key: str
    display: str
    layout: str
    root: Path
    color: str = "#4E79A7"


@dataclass(frozen=True)
class APIConfig:
    base_url: str
    api_key: str
    judge_model: str
    embedding_model: str = ""


GROUPS: list[GroupConfig] = []


def get_groups(_preset: str = "default") -> list[GroupConfig]:
    return list(GROUPS)


def get_group(key: str, groups: Optional[Sequence[GroupConfig]] = None) -> GroupConfig:
    candidates = list(groups if groups is not None else GROUPS)
    for group in candidates:
        if group.key == key:
            return group
    raise KeyError(f"Unknown group {key!r}; available: {[g.key for g in candidates]}")


def set_output_dir(path: Path) -> None:
    global _ACTIVE_OUTPUT_DIR
    _ACTIVE_OUTPUT_DIR = Path(path)


def output_subdir(name: str) -> Path:
    path = _ACTIVE_OUTPUT_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_api_config():
    """Load remote Judge credentials without embedding a provider endpoint."""
    from utils.judge.settings import JUDGE_API_BASE, JUDGE_API_KEY, JUDGE_MODEL

    if not JUDGE_API_BASE or not JUDGE_API_KEY:
        raise RuntimeError(
            "Set AIEDU_JUDGE_API_BASE and AIEDU_JUDGE_API_KEY for an online QDR run"
        )
    return APIConfig(
        base_url=JUDGE_API_BASE,
        api_key=JUDGE_API_KEY,
        judge_model=JUDGE_MODEL or "judge-model",
    )
