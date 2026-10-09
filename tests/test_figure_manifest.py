from __future__ import annotations

from pathlib import Path

from paper_reproduction.runner import _sha256, verify_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_contains_only_four_public_figures() -> None:
    report = verify_manifest(ROOT / "paper_reproduction" / "manifest.yaml")
    assert [item["id"] for item in report["figures"]] == [
        "question-preference",
        "question-quality",
        "reasoning-consistency",
        "workflow-time",
    ]


def test_csv_hash_is_stable_across_git_line_endings(tmp_path: Path) -> None:
    source = ROOT / "paper_reproduction" / "plot_data" / "question_quality.csv"
    crlf_copy = tmp_path / source.name
    crlf_copy.write_bytes(source.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert _sha256(crlf_copy) == _sha256(source)
