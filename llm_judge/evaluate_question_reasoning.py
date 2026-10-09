"""Evaluate externally generated question or reasoning outputs with Anchor Judge.

Usage:
    python -m llm_judge.evaluate_question_reasoning --data-dir <package> --dry-run
    python -m llm_judge.evaluate_question_reasoning --data-dir <package> --tasks question,reasoning

The script reads model outputs from ``<data-dir>/by_model/`` and compares
candidate models against a baseline.  It never embeds a provider URL or key;
online runs receive those values from the environment.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from llm_judge import AnchorJudge, _resolve_judge_credentials
except ImportError:
    # Allow running directly inside package dir without pip-install
    _pkg_root = Path(__file__).parent
    sys.path.insert(0, str(_pkg_root.parent))
    from llm_judge import AnchorJudge, _resolve_judge_credentials


def load_model_outputs(data_dir: Path, task: str) -> dict:
    """Load all model outputs for a task from the by_model directory.

    Returns: {model_name: {case_id: content, ...}, ...}
    """
    by_model = {}
    by_model_dir = data_dir / "by_model"
    if not by_model_dir.exists():
        print(f"Error: '{by_model_dir}' not found. Is --data-dir correct?")
        return by_model

    for model_dir in sorted(by_model_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        task_dir = model_dir / task
        if not task_dir.exists():
            print(f"  [SKIP] {model_dir.name}: no '{task}' task data")
            continue
        outputs = {}
        for case_file in sorted(task_dir.glob("case*.md")):
            case_id = case_file.stem
            outputs[case_id] = case_file.read_text(encoding="utf-8")
        if outputs:
            by_model[model_dir.name] = outputs

    return by_model


def _map_task_name(task: str) -> str:
    """Map package task names to judge-recognized task names."""
    mapping = {
        "question": "考题生成",
        "reasoning": "临床思维",
    }
    return mapping.get(task, task)


def dry_run(data_dir: Path, tasks: list[str], baseline: str, limit_cases: int | None = None) -> None:
    """Parse data directory and print summary without calling API."""
    print(f"Data directory: {data_dir.resolve()}")
    print(f"Baseline model: {baseline}")
    print(f"Tasks: {', '.join(tasks)}")
    if limit_cases:
        print(f"Limit: {limit_cases} cases per task")
    print()

    for task in tasks:
        model_outputs = load_model_outputs(data_dir, task)
        if not model_outputs:
            print(f"[{task}] No models found!")
            continue

        # Apply limit for dry-run display
        if limit_cases:
            for model_name in model_outputs:
                case_ids = list(model_outputs[model_name].keys())[:limit_cases]
                model_outputs[model_name] = {k: model_outputs[model_name][k] for k in case_ids}

        models = list(model_outputs.keys())
        case_counts = {m: len(o) for m, o in model_outputs.items()}

        display_total = sum(case_counts.values())
        limit_note = f" (limited to first {limit_cases} cases)" if limit_cases else ""
        print(f"[{task}] {len(models)} models, {display_total} total cases{limit_note}:")
        for model in sorted(models):
            marker = " [BASELINE]" if model == baseline else ""
            print(f"  {model}: {case_counts[model]} cases{marker}")

        # Show candidates (non-baseline models)
        candidates = [m for m in models if m != baseline]
        if baseline not in models:
            print(f"  WARNING: Baseline '{baseline}' not found for this task!")
        else:
            print(f"  => {len(candidates)} candidates to compare against baseline")
        print()


def evaluate_task(task: str, data_dir: Path, baseline: str, output_dir: Path,
                  repeats: int, workers: int, n_boot: int,
                  limit_cases: int | None = None, resume: bool = False,
                  cases_dir: Path | None = None) -> dict:
    """Run AnchorJudge evaluation for one task."""
    model_outputs = load_model_outputs(data_dir, task)
    if baseline not in model_outputs:
        raise ValueError(f"Baseline '{baseline}' not found in data for task '{task}'")

    # Apply case limit if requested
    if limit_cases:
        baseline_cases = list(model_outputs[baseline].keys())[:limit_cases]
        model_outputs[baseline] = {k: model_outputs[baseline][k] for k in baseline_cases}

    # Prefer an explicit case directory.  The fallback keeps synthetic packages
    # usable when they contain only model outputs.
    raw_cases = {}
    if cases_dir is not None:
        for case_path in sorted(cases_dir.glob("case*.*")):
            if case_path.suffix.lower() not in {".md", ".txt", ".json"}:
                continue
            raw_cases[case_path.stem] = case_path.read_text(encoding="utf-8")
    for case_id, content in model_outputs[baseline].items():
        if case_id not in raw_cases:
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            raw_cases[case_id] = lines[0] if lines else case_id

    candidates = {k: v for k, v in model_outputs.items() if k != baseline}
    if not candidates:
        print(f"[{task}] No candidate models to evaluate (only baseline found)")
        return {}

    # Apply case limit to all candidates too (must match baseline cases)
    if limit_cases:
        for candidate_name in candidates:
            candidate_cases = list(candidates[candidate_name].keys())[:limit_cases]
            candidates[candidate_name] = {k: candidates[candidate_name][k] for k in candidate_cases}

    # Rebuild model_outputs dict with baseline first (AnchorJudge expects this)
    all_outputs = {baseline: model_outputs[baseline]}
    all_outputs.update(candidates)

    model_name = os.environ.get("AIEDU_JUDGE_MODEL", "").strip()
    if not model_name:
        raise RuntimeError("Set AIEDU_JUDGE_MODEL or pass --judge-model before an online run")
    api_base, api_key = _resolve_judge_credentials(model_name)
    judge = AnchorJudge(
        baseline=baseline,
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        max_tokens=8192,  # Long model outputs need larger token budget
    )

    judge_task = _map_task_name(task)
    task_output_dir = output_dir / task
    task_output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = str(task_output_dir / "checkpoint.json") if resume else None
    if resume:
        print(f"  Checkpoint: enabled (will resume from {task_output_dir / 'checkpoint.json'})")

    print(f"\n[{task}] Starting evaluation: {len(candidates)} candidates, "
          f"{len(raw_cases)} cases, {repeats} repeats each")
    print(f"  Judge task: {judge_task}")
    print(f"  Output: {task_output_dir.resolve()}")

    results = judge.compare_all(
        task=judge_task,
        raw_cases=raw_cases,
        model_outputs=all_outputs,
        repeats=repeats,
        workers=workers,
        output_dir=str(task_output_dir),
        visualize=True,
        n_boot=n_boot,
        checkpoint_path=checkpoint_path,
    )

    # Print summary
    print(f"\n[{task}] Results:")
    if results.get("stats_df") is not None:
        stats = results["stats_df"]
        for _, row in stats.iterrows():
            print(f"  {row['Model']}: Score={row['Score']:.1%} "
                  f"[CI: {row['Boot_CI_Lower']:.1%} - {row['Boot_CI_Upper']:.1%}]")

    if results.get("viz_paths"):
        for name, path in results["viz_paths"].items():
            print(f"  Plot: {path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated question or reasoning outputs with a remote OpenAI-compatible LLM Judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m llm_judge.evaluate_question_reasoning --data-dir <package> --dry-run
  python -m llm_judge.evaluate_question_reasoning --data-dir <package> --tasks question
  python -m llm_judge.evaluate_question_reasoning --data-dir <package> --tasks question,reasoning --repeats 5 --workers 8""",
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing by_model/<model>/<task>/case*.md")
    parser.add_argument("--cases-dir", default=None,
                        help="Optional directory containing raw case*.md/.txt/.json files")
    parser.add_argument("--baseline", default="baseline",
                        help="Baseline model directory name (default: baseline)")
    parser.add_argument("--judge-model", default=os.environ.get("AIEDU_JUDGE_MODEL", ""),
                        help="Remote Judge model; also accepted through AIEDU_JUDGE_MODEL")
    parser.add_argument("--tasks", default="question,reasoning",
                        help="Comma-separated task names (default: question,reasoning)")
    parser.add_argument("--output-dir", default="./results",
                        help="Output directory for results and plots (default: ./results)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of repeat votes per case (default: 3)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of concurrent worker threads (default: 4)")
    parser.add_argument("--n-boot", type=int, default=10000,
                        help="Bootstrap samples for CI (default: 10000)")
    parser.add_argument("--limit-cases", type=int, default=None,
                        help="Limit to first N cases per task (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse data directory without calling API")
    parser.add_argument("--resume", action="store_true",
                        help="Resume evaluation from checkpoint if exists")

    args = parser.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    data_dir = Path(args.data_dir)
    cases_dir = Path(args.cases_dir) if args.cases_dir else None
    output_dir = Path(args.output_dir)
    if args.judge_model:
        os.environ["AIEDU_JUDGE_MODEL"] = args.judge_model

    print("=" * 60)
    print("LLM Judge — Question & Reasoning Evaluation")
    print("=" * 60)

    if args.dry_run:
        dry_run(data_dir, tasks, args.baseline, limit_cases=args.limit_cases)
        print("\n[Dry-run complete — no API calls made]")
        return

    for task in tasks:
        try:
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
        except ValueError as e:
            print(f"\n[{task}] ERROR: {e}")
        except Exception as e:
            print(f"\n[{task}] UNEXPECTED ERROR: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print(f"Results saved to: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
