"""Checkpoint management for resumable AnchorJudge evaluation."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path


class CheckpointManager:
    """Thread-safe checkpoint manager for anchor evaluation progress.

    Stores per-candidate, per-case vote results so that an interrupted
    evaluation can resume without re-running already-completed cases.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "version": 1,
            "task": "",
            "baseline": "",
            "candidates": {},
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)

    def is_candidate_done(self, candidate: str, case_ids: set[str]) -> bool:
        completed = set(
            self._data.get("candidates", {}).get(candidate, {}).get("cases", {}).keys()
        )
        return case_ids.issubset(completed)

    def get_completed_cases(self, candidate: str) -> set[str]:
        return set(
            self._data.get("candidates", {}).get(candidate, {}).get("cases", {}).keys()
        )

    def get_results(self, candidate: str) -> tuple[int, int, list[dict]]:
        """Return aggregated (baseline_wins, candidate_wins, reasons) for a candidate."""
        cand_data = self._data.get("candidates", {}).get(candidate, {})
        cases = cand_data.get("cases", {})
        total_baseline = sum(c["baseline_wins"] for c in cases.values())
        total_candidate = sum(c["candidate_wins"] for c in cases.values())
        all_reasons: list[dict] = []
        for case_data in cases.values():
            all_reasons.extend(case_data.get("reasons", []))
        return total_baseline, total_candidate, all_reasons

    def record_case(
        self,
        candidate: str,
        case_id: str,
        baseline_wins: int,
        candidate_wins: int,
        reasons: list[dict],
    ) -> None:
        with self._lock:
            if "candidates" not in self._data:
                self._data["candidates"] = {}
            if candidate not in self._data["candidates"]:
                self._data["candidates"][candidate] = {"cases": {}}
            self._data["candidates"][candidate]["cases"][case_id] = {
                "baseline_wins": baseline_wins,
                "candidate_wins": candidate_wins,
                "reasons": reasons,
            }
            self._save()

    def set_metadata(self, task: str, baseline: str) -> None:
        self._data["task"] = task
        self._data["baseline"] = baseline
        self._save()
