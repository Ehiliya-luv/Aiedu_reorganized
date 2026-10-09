from __future__ import annotations

from pathlib import Path

from utils.quality_metrics.config import GroupConfig
from utils.quality_metrics.parser import parse_group


ROOT = Path(__file__).resolve().parents[1]


def test_question_fixture_preserves_existing_markdown_format() -> None:
    group = GroupConfig("demo", "demo", "flat", ROOT / "fixtures" / "question_inputs" / "demo")
    rows = parse_group("demo", groups=[group])
    assert len(rows) == 1
    assert rows[0]["answer"] == "A"
    assert set(rows[0]["options"]) == set("ABCDE")
