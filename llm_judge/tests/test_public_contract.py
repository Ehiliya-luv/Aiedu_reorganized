from __future__ import annotations

from llm_judge.evaluate_paper_question import build_parser, verify_inputs
from llm_judge.evaluate_question_reasoning import load_model_outputs
from pathlib import Path


def test_result_judge_parser_exposes_offline_modes() -> None:
    actions = {action.dest for action in build_parser()._actions}
    assert {"dry_run", "verify", "reproduce", "round", "run_id"} <= actions
    assert build_parser().parse_args([]).prompt_variant == "prompt_v3"


def test_fixture_input_layout_is_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    report = verify_inputs(root / "fixtures" / "judge_outputs", ["question"], "baseline")
    assert report["tasks"]["question"]["cases_per_model"] == 1


def test_question_loader_accepts_original_nested_layout(tmp_path: Path) -> None:
    for group, model, text in (
        ("baseline", "model-a", "baseline output"),
        ("lora-model", "model-b", "candidate output"),
    ):
        case_dir = tmp_path / group / model / "case001"
        case_dir.mkdir(parents=True)
        (case_dir / "question.md").write_text(text, encoding="utf-8")

    loaded = load_model_outputs(tmp_path, "question")

    assert loaded == {
        "model-a": {"case001": "baseline output"},
        "model-b": {"case001": "candidate output"},
    }
