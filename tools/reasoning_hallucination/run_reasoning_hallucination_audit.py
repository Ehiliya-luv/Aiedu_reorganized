"""Evaluate reasoning consistency against externally supplied case text.

The public implementation stores only case identifiers, hashes and aggregate
counts.  Raw cases and model outputs are read from operator-provided paths and
are never copied into the repository.  ``--mock`` is deterministic and is used
by the synthetic fixture tests; an online run uses the same OpenAI-compatible
configuration as the result Judge.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS = ("supported", "inferable", "unsupported", "contradicted")
CASE_ID = re.compile(r"^case(\d+)$", re.IGNORECASE)


def _canonical_case_id(raw: str) -> str | None:
    match = CASE_ID.fullmatch(raw)
    if not match:
        return None
    return f"case{int(match.group(1)):03d}"


def _resolve(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def discover_tasks(data_dir: Path, cases_dir: Path) -> list[dict[str, Any]]:
    """Discover model/case pairs from public and original project layouts.

    The public fixture layout is ``by_model/<model>/reasoning/case001.md``.
    The project also uses ``<group>/<model>/case001/reasoning.md`` and a
    flat ``<model>/case001.md`` layout.  All three are accepted so an operator
    can point the evaluator at existing outputs without copying them.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"reasoning data directory does not exist: {data_dir}")
    if not cases_dir.is_dir():
        raise FileNotFoundError(f"case directory does not exist: {cases_dir}")
    case_files: dict[str, Path] = {}
    for path in cases_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".json"}:
            continue
        case_id = _canonical_case_id(path.stem)
        if case_id is not None:
            case_files[case_id] = path
    candidates: list[tuple[Path, dict[str, Path]]] = []

    # First handle the stable public layout.  If it is present and populated,
    # do not recursively rediscover the same files below it.
    root = data_dir / "by_model"
    if root.is_dir():
        for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            reasoning_dir = model_dir / "reasoning"
            files: dict[str, Path] = {}
            for path in reasoning_dir.glob("case*.*"):
                if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".json"}:
                    continue
                case_id = _canonical_case_id(path.stem)
                if case_id is not None:
                    files[case_id] = path
            if files:
                candidates.append((model_dir, files))

    if not candidates:
        # Original project layouts.  A candidate model directory is any
        # directory containing direct case files or case directories with a
        # reasoning.md child.  This deliberately ignores metadata files such
        # as case097_099_supplement_manifest.json.
        directories = [data_dir]
        directories.extend(path for path in data_dir.rglob("*") if path.is_dir())
        for model_dir in sorted(set(directories)):
            files: dict[str, Path] = {}
            for path in model_dir.glob("case*.*"):
                if path.is_file() and path.suffix.lower() in {".md", ".txt", ".json"}:
                    case_id = _canonical_case_id(path.stem)
                    if case_id is not None:
                        files[case_id] = path
            for case_dir in model_dir.glob("case*"):
                if not case_dir.is_dir():
                    continue
                case_id = _canonical_case_id(case_dir.name)
                if case_id is None:
                    continue
                nested = case_dir / "reasoning.md"
                if nested.is_file():
                    files[case_id] = nested
            if files:
                candidates.append((model_dir, files))

    if not candidates:
        expected = root if root.is_dir() else data_dir
        raise FileNotFoundError(f"no reasoning files found under {expected}")

    name_counts = Counter(directory.name for directory, _ in candidates)
    tasks: list[dict[str, Any]] = []
    for model_dir, files in candidates:
        model_name = model_dir.name
        if name_counts[model_name] > 1:
            model_name = f"{model_dir.parent.name}:{model_name}"
        for case_id, reasoning_file in sorted(files.items()):
            if case_id not in case_files:
                raise ValueError(f"missing raw case for {model_name}/{case_id}")
            tasks.append(
                {
                    "model": model_name,
                    "case_id": case_id,
                    "reasoning_file": reasoning_file,
                    "case_file": case_files[case_id],
                }
            )
    return tasks


def _mock_counts(reasoning: str) -> dict[str, Any]:
    """Count explicit fixture labels without attempting medical inference.

    A fixture line may start with ``[supported]``, ``[inferable]``,
    ``[unsupported]`` or ``[contradicted]``.  Unlabelled non-empty lines are
    treated as supported so a minimal synthetic reasoning file remains useful.
    """
    counts = {label: 0 for label in LABELS}
    examples = {"unsupported_examples": [], "contradicted_examples": []}
    lines = [line.strip() for line in reasoning.splitlines() if line.strip()]
    if not lines:
        return {
            "invalid_reasoning": True,
            "total_claims": 0,
            **counts,
            **examples,
            "notes": "empty reasoning fixture",
        }
    labelled = False
    for line in lines:
        match = re.match(r"^\[(supported|inferable|unsupported|contradicted)\]\s*(.*)$", line, re.I)
        if match:
            labelled = True
            label = match.group(1).lower()
            counts[label] += 1
            if label in {"unsupported", "contradicted"} and len(examples[f"{label}_examples"]) < 2:
                examples[f"{label}_examples"].append(match.group(2)[:80])
        elif not labelled:
            counts["supported"] += 1
    total = sum(counts.values())
    return {
        "invalid_reasoning": False,
        "total_claims": total,
        **counts,
        **examples,
        "notes": "deterministic synthetic fixture classification",
    }


def _normalize(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("judge response must be a JSON object")
    invalid = bool(payload.get("invalid_reasoning", False))
    values = {label: int(payload.get(label, 0)) for label in LABELS}
    total = int(payload.get("total_claims", sum(values.values())))
    if min(total, *values.values()) < 0 or sum(values.values()) != total:
        raise ValueError("reasoning count totals are inconsistent")
    if invalid and total != 0:
        raise ValueError("invalid reasoning must have zero claims")
    if not invalid and total == 0:
        raise ValueError("valid reasoning must contain at least one claim")
    consistency = (values["supported"] + values["inferable"]) / total if total else 0.0
    hallucination = (values["unsupported"] + values["contradicted"]) / total if total else 0.0
    return {
        "invalid_reasoning": invalid,
        "total_claims": total,
        **values,
        "case_consistency": consistency,
        "hallucination_rate": hallucination,
        "unsupported_examples": list(payload.get("unsupported_examples") or [])[:2],
        "contradicted_examples": list(payload.get("contradicted_examples") or [])[:2],
        "notes": str(payload.get("notes") or "")[:160],
    }


def _online_counts(case_text: str, reasoning_text: str, model: str) -> dict[str, Any]:
    from llm_judge import OpenAICompatibleClient, _resolve_judge_credentials

    api_base, api_key = _resolve_judge_credentials(model)
    if not api_base or not api_key:
        raise RuntimeError("set the Judge API environment variables before an online run")
    client = OpenAICompatibleClient(base_url=api_base, api_key=api_key, timeout=300)
    prompt = (
        "Classify medical claims in the reasoning against the case. Return JSON only. "
        "Use supported, inferable, unsupported and contradicted; counts must sum to total_claims.\n\n"
        f"CASE:\n{case_text}\n\nREASONING:\n{reasoning_text}\n\n"
        '{"invalid_reasoning":false,"total_claims":0,"supported":0,"inferable":0,'
        '"unsupported":0,"contradicted":0,"unsupported_examples":[],"contradicted_examples":[],"notes":""}'
    )
    content = client.chat_complete_text(
        model=model,
        system_prompt="You are a concise clinical reasoning consistency evaluator.",
        user_prompt=prompt,
        temperature=0.0,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    return _normalize(json.loads(content))


def _evaluate_one(task: dict[str, Any], *, mock: bool, judge_model: str) -> dict[str, Any]:
    case_text = _load_text(task["case_file"])
    reasoning_text = _load_text(task["reasoning_file"])
    counts = _mock_counts(reasoning_text) if mock else _online_counts(case_text, reasoning_text, judge_model)
    normalized = _normalize(counts)
    # Keep the persisted audit privacy-preserving: examples and notes can
    # quote source text, so only hashes and numeric classifications are saved.
    normalized.pop("unsupported_examples", None)
    normalized.pop("contradicted_examples", None)
    normalized.pop("notes", None)
    return {
        "model": task["model"],
        "case_id": task["case_id"],
        "case_sha256": _sha256(task["case_file"]),
        "reasoning_sha256": _sha256(task["reasoning_file"]),
        **normalized,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        summary = result.setdefault(
            record["model"],
            {"cases": 0, "total_claims": 0, **{label: 0 for label in LABELS}},
        )
        summary["cases"] += 1
        for key in ("total_claims", *LABELS):
            summary[key] += int(record[key])
    for summary in result.values():
        total = summary["total_claims"]
        summary["case_consistency"] = (
            (summary["supported"] + summary["inferable"]) / total if total else 0.0
        )
        summary["hallucination_rate"] = (
            (summary["unsupported"] + summary["contradicted"]) / total if total else 0.0
        )
    return result


def write_outputs(records: list[dict[str, Any]], output_dir: Path, baseline: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda row: (row["model"], row["case_id"]))
    with (output_dir / "items.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize(records)
    (output_dir / "summary.json").write_text(
        json.dumps({"baseline": baseline, "models": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = ["model", "cases", "total_claims", *LABELS, "case_consistency", "hallucination_rate"]
    with (output_dir / "reasoning_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, values in sorted(summary.items()):
            writer.writerow({"model": model, **values})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reasoning consistency and hallucination-rate evaluation")
    parser.add_argument("--data-dir", default="fixtures/reasoning_outputs")
    parser.add_argument("--cases-dir", default="fixtures/reasoning_cases")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--output-dir", default="results/reasoning_quality")
    parser.add_argument("--judge-model", default=os.environ.get("AIEDU_JUDGE_MODEL", ""))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use deterministic fixture labels; no API call")
    parser.add_argument("--dry-run", action="store_true", help="Inspect discovered tasks only")
    parser.add_argument("--verify", action="store_true", help="Validate input layout only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tasks = discover_tasks(_resolve(args.data_dir), _resolve(args.cases_dir))
        if args.limit_cases > 0:
            case_ids = sorted({task["case_id"] for task in tasks})[: args.limit_cases]
            tasks = [task for task in tasks if task["case_id"] in case_ids]
        models = sorted({task["model"] for task in tasks})
        if args.baseline not in models:
            raise ValueError(f"baseline {args.baseline!r} not found; available: {models}")
        print(f"models={models}; tasks={len(tasks)}; baseline={args.baseline}")
        if args.dry_run or args.verify:
            return 0
        if not args.mock and not args.judge_model:
            raise RuntimeError("pass --mock for offline fixtures or set --judge-model for an online run")

        output_dir = _resolve(args.output_dir)
        existing: dict[tuple[str, str], dict[str, Any]] = {}
        items_path = output_dir / "items.jsonl"
        if args.resume and items_path.exists():
            for line in items_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    existing[(row["model"], row["case_id"])] = row
        pending = [task for task in tasks if (task["model"], task["case_id"]) not in existing]
        if args.mock:
            for task in pending:
                existing[(task["model"], task["case_id"])] = _evaluate_one(task, mock=True, judge_model="")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = [executor.submit(_evaluate_one, task, mock=False, judge_model=args.judge_model) for task in pending]
                for task, future in zip(pending, futures):
                    existing[(task["model"], task["case_id"])] = future.result()
        write_outputs(list(existing.values()), output_dir, args.baseline)
        print(f"wrote {len(existing)} private evaluation records to {output_dir}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
