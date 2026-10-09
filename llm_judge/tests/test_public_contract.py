from __future__ import annotations

from llm_judge.evaluate_paper_question import build_parser, verify_inputs
from pathlib import Path


def test_result_judge_parser_exposes_offline_modes() -> None:
    actions = {action.dest for action in build_parser()._actions}
    assert {"dry_run", "verify", "reproduce", "round", "run_id"} <= actions


def test_fixture_input_layout_is_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    report = verify_inputs(root / "fixtures" / "judge_outputs", ["question"], "baseline")
    assert report["tasks"]["question"]["cases_per_model"] == 1
