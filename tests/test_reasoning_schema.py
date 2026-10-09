from __future__ import annotations

from pathlib import Path

from tools.reasoning_hallucination.run_reasoning_hallucination_audit import discover_tasks
from tools.reasoning_hallucination.run_reasoning_hallucination_audit import _mock_counts, _normalize


def test_reasoning_fixture_counts_are_consistent() -> None:
    result = _normalize(_mock_counts("[supported] one\n[inferable] two\n[unsupported] three"))
    assert result["total_claims"] == 3
    assert result["case_consistency"] == 2 / 3


def test_reasoning_discovery_accepts_original_nested_layout(tmp_path: Path) -> None:
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / "case001.md").write_text("case", encoding="utf-8")
    outputs = tmp_path / "reasoning" / "baseline" / "model-a" / "case001"
    outputs.mkdir(parents=True)
    (outputs / "reasoning.md").write_text("[supported] claim", encoding="utf-8")

    tasks = discover_tasks(tmp_path / "reasoning", cases)

    assert [(task["model"], task["case_id"]) for task in tasks] == [("model-a", "case001")]
    assert tasks[0]["reasoning_file"].name == "reasoning.md"
