"""Formal result-evaluation LLM Judge entry point.

This adapter keeps the existing Anchor Judge implementation in
``evaluate_question_reasoning.py`` and adds the paper-facing offline checks.
It never ships paper cases or model outputs; operators provide those paths at
run time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .evaluate_question_reasoning import dry_run, evaluate_task, load_model_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def verify_inputs(
    data_dir: Path,
    tasks: list[str],
    baseline: str,
    cases_dir: Path | None = None,
) -> dict:
    """Validate the input layout without reading private content into output."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {data_dir}")
    report: dict[str, object] = {
        "data_dir": str(data_dir),
        "baseline": baseline,
        "tasks": {},
    }
    for task in tasks:
        outputs = load_model_outputs(data_dir, task)
        if baseline not in outputs:
            raise ValueError(f"baseline {baseline!r} is missing for task {task!r}")
        candidates = [name for name in outputs if name != baseline]
        if not candidates:
            raise ValueError(f"no candidate model found for task {task!r}")
        baseline_cases = set(outputs[baseline])
        mismatches = {
            name: sorted(baseline_cases.symmetric_difference(outputs[name]))
            for name in candidates
            if set(outputs[name]) != baseline_cases
        }
        if mismatches:
            raise ValueError(f"case sets differ for task {task!r}: {mismatches}")
        report["tasks"][task] = {
            "models": sorted(outputs),
            "cases_per_model": len(baseline_cases),
        }

    if cases_dir is not None:
        if not cases_dir.is_dir():
            raise FileNotFoundError(f"cases directory does not exist: {cases_dir}")
        case_ids = {path.stem for path in cases_dir.glob("case*.*")}
        for task in tasks:
            expected = set(load_model_outputs(data_dir, task)[baseline])
            missing = sorted(expected - case_ids)
            if missing:
                raise ValueError(f"missing raw cases for {task!r}: {missing[:5]}")
        report["cases_dir"] = str(cases_dir)
    return report


def verify_frozen_figures(manifest: Path | None = None) -> dict:
    """Verify the four public paper-figure inputs and their renderer contract."""
    from paper_reproduction.runner import verify_manifest

    return verify_manifest(manifest or PROJECT_ROOT / "paper_reproduction" / "manifest.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate result outputs with the remote OpenAI-compatible LLM Judge"
    )
    parser.add_argument("--data-dir", default="fixtures/judge_outputs")
    parser.add_argument("--cases-dir", default=None)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--tasks", default="question")
    parser.add_argument("--output-dir", default="results/llm_judge")
    parser.add_argument("--judge-model", default=os.environ.get("AIEDU_JUDGE_MODEL", ""))
    parser.add_argument("--round", type=int, default=None, help="Optional numbered evaluation round")
    parser.add_argument("--run-id", default=None, help="Optional run name for private output")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Inspect inputs without API calls")
    parser.add_argument("--verify", action="store_true", help="Validate input layout and matching case sets")
    parser.add_argument("--reproduce", action="store_true", help="Verify frozen paper-figure inputs")
    parser.add_argument("--manifest", default="paper_reproduction/manifest.yaml")
    return parser


def _run_online(args: argparse.Namespace, tasks: list[str], data_dir: Path, cases_dir: Path | None) -> None:
    if not args.judge_model:
        raise RuntimeError("set AIEDU_JUDGE_MODEL or pass --judge-model for an online run")
    os.environ["AIEDU_JUDGE_MODEL"] = args.judge_model
    output_dir = _resolve_path(args.output_dir) or PROJECT_ROOT / "results" / "llm_judge"
    if args.run_id:
        output_dir = output_dir / "runs" / args.run_id
    if args.round is not None:
        output_dir = output_dir / f"round-{args.round}"
    for task in tasks:
        evaluate_task(
            task=task,
            data_dir=data_dir,
            baseline=args.baseline,
            output_dir=output_dir,
            repeats=args.repeats,
            workers=args.workers,
            n_boot=args.n_boot,
            limit_cases=args.limit_cases,
            resume=args.resume,
            cases_dir=cases_dir,
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    data_dir = _resolve_path(args.data_dir)
    cases_dir = _resolve_path(args.cases_dir)
    assert data_dir is not None

    try:
        if args.reproduce:
            print(json.dumps(verify_frozen_figures(_resolve_path(args.manifest)), ensure_ascii=False, indent=2))
            return 0
        if args.verify:
            report = verify_inputs(data_dir, tasks, args.baseline, cases_dir)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.dry_run:
            dry_run(data_dir, tasks, args.baseline, limit_cases=args.limit_cases)
            return 0
        _run_online(args, tasks, data_dir, cases_dir)
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
