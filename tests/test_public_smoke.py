from __future__ import annotations

from pathlib import Path

from dpo.utils.dpo_data import load_dpo_samples
from llm_judge.pairwise import parse_judge_result
from llm_judge.stats import compute_anchor_score
from utils.quality_metrics.config import GroupConfig
from utils.quality_metrics.parser import parse_group
from utils.quality_metrics.qdr import summarize_qdr


ROOT = Path(__file__).resolve().parents[1]


def test_dpo_fixture_uses_required_schema() -> None:
    samples = load_dpo_samples(str(ROOT / "fixtures" / "dpo" / "dpo_pairs.jsonl"))
    assert len(samples) == 2
    assert all(sample.prompt and sample.chosen and sample.rejected for sample in samples)
    assert all(sample.chosen != sample.rejected for sample in samples)


def test_question_parser_preserves_existing_format() -> None:
    group = GroupConfig(
        key="demo",
        display="demo",
        layout="flat",
        root=ROOT / "fixtures" / "question_inputs" / "demo",
    )
    items = parse_group("demo", groups=[group])
    assert len(items) == 1
    assert items[0]["stem"].startswith("合成患者出现短暂发热")
    assert set(items[0]["options"]) == set("ABCDE")
    assert items[0]["answer"] == "A"
    assert items[0]["structural_issues"] == []


def test_qdr_summary_contract() -> None:
    item = {
        "group": "demo",
        "case_id": "case001",
        "qtype": "A1",
        "answer": "A",
        "solvers": {"s1": {"answer": "A"}},
        "analyzer": {"skipped": True},
    }
    summary = summarize_qdr([item])
    assert summary["demo"]["judged"] == 1
    assert summary["demo"]["defect"] == 0
    assert summary["demo"]["credibility_rate"] == 1.0


def test_judge_parser_and_score_are_deterministic() -> None:
    parsed = parse_judge_result('```json\n{"winner":"B","reason":"more complete"}\n```')
    assert parsed == {"winner": "B", "reason": "more complete"}
    score = compute_anchor_score(90, 210)
    assert score["score"] == 0.7
    assert score["total_votes"] == 300
