from __future__ import annotations

from pathlib import Path

from dpo.utils.dpo_data import load_dpo_samples


ROOT = Path(__file__).resolve().parents[1]


def test_dpo_fixture_has_complete_preference_pairs() -> None:
    rows = load_dpo_samples(str(ROOT / "examples" / "dpo_pairs.synthetic.jsonl"))
    assert len(rows) == 2
    assert all(row.prompt and row.chosen and row.rejected for row in rows)
    assert all(row.chosen != row.rejected for row in rows)
