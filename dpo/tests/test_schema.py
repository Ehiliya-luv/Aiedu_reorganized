from __future__ import annotations

from pathlib import Path

from dpo.utils.dpo_data import load_dpo_samples


def test_synthetic_example_loads() -> None:
    root = Path(__file__).resolve().parents[2]
    assert len(load_dpo_samples(str(root / "examples" / "dpo_pairs.synthetic.jsonl"))) == 2
