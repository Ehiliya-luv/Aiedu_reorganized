from __future__ import annotations

from pathlib import Path

from paper_reproduction.runner import verify_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_contains_only_four_public_figures() -> None:
    report = verify_manifest(ROOT / "paper_reproduction" / "manifest.yaml")
    assert [item["id"] for item in report["figures"]] == [
        "question-preference",
        "question-quality",
        "reasoning-consistency",
        "workflow-time",
    ]
