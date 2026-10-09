"""Question parsing and QDR evaluation entry point.

The command keeps the original repository's familiar ``parse`` and ``qdr``
subcommands while accepting paths at runtime.  No data directory is assumed
to be public or checked into Git.

Examples:
    python compute_quality_metrics.py parse \
        --group-dir demo=fixtures/question_inputs/demo \
        --output-dir scratch/demo_quality
    python compute_quality_metrics.py qdr \
        --parsed-dir scratch/demo_quality/parsed \
        --output-dir results/question_quality
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from utils.quality_metrics.config import GroupConfig, load_api_config, set_output_dir
from utils.quality_metrics.parser import parse_group, write_jsonl
from utils.quality_metrics.qdr import SOLVER_MODELS, evaluate_qdr, summarize_qdr


PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _detect_layout(root: Path) -> str:
    if any(root.glob("case*.md")):
        return "flat"
    if any((p / "question.md").exists() for p in root.glob("case*")):
        return "nested_question"
    if any((p / "per_stage" / "questions.md").exists() for p in root.glob("case*")):
        return "per_stage"
    raise ValueError(
        f"Cannot detect question layout under {root}; expected case*.md, "
        "case*/question.md, or case*/per_stage/questions.md"
    )


def _parse_group_spec(spec: str) -> GroupConfig:
    if "=" not in spec:
        raise ValueError(f"--group-dir must be NAME=PATH, got {spec!r}")
    key, raw_path = spec.split("=", 1)
    key = key.strip()
    root = _resolve_path(raw_path.strip())
    if not key or not root.is_dir():
        raise ValueError(f"Invalid group directory: {spec!r}")
    return GroupConfig(key=key, display=key, layout=_detect_layout(root), root=root)


def _limit_cases(items: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return items
    seen: dict[str, set[str]] = {}
    result: list[dict] = []
    for item in items:
        group_seen = seen.setdefault(item["group"], set())
        if item["case_id"] in group_seen or len(group_seen) < limit:
            group_seen.add(item["case_id"])
            result.append(item)
    return result


def _write_summary(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    types = [
        "gold_wrong",
        "multiple_valid",
        "structural_issue",
        "ambiguous",
        "impossible",
        "not_defect",
    ]
    lines = [
        "group,total,judged,suspect,defect,qdr,credibility_rate,suspect_rate,"
        + ",".join(types)
    ]
    for group, values in summary.items():
        lines.append(
            ",".join(
                [
                    group,
                    str(values["total"]),
                    str(values["judged"]),
                    str(values["suspect"]),
                    str(values["defect"]),
                    f"{values['qdr']:.6f}",
                    f"{values['credibility_rate']:.6f}",
                    f"{values['suspect_rate']:.6f}",
                    *(str(values["type_counts"].get(t, 0)) for t in types),
                ]
            )
        )
    (output_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_parse(args: argparse.Namespace) -> None:
    groups = [_parse_group_spec(spec) for spec in args.group_dir]
    output_dir = _resolve_path(args.output_dir)
    parsed_dir = output_dir / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for group in groups:
        items = _limit_cases(parse_group(group.key, groups=groups), args.limit)
        path = parsed_dir / f"{group.key}.jsonl"
        total += write_jsonl(items, path)
        print(f"[parse] {group.key}: {len(items)} items -> {path}")
    print(f"[parse] total items: {total}")


def _read_parsed(parsed_dir: Path) -> list[dict]:
    items: list[dict] = []
    for path in sorted(parsed_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON: {path}:{line_no}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected an object: {path}:{line_no}")
                items.append(record)
    if not items:
        raise ValueError(f"No parsed JSONL files found under {parsed_dir}")
    return items


def _mock_records(items: Iterable[dict]) -> list[dict]:
    """Produce deterministic offline records for public fixture validation."""
    records = []
    for item in items:
        if item.get("structural_issues"):
            records.append({**item, "analyzer": {"auto_flagged": "structural_issue", "is_defect": True, "defect_type": "structural_issue"}, "solvers": {}})
            continue
        records.append(
            {
                **item,
                "solvers": {model: {"answer": item["answer"], "reason": "fixture"} for model in SOLVER_MODELS},
                "n_disagree": 0,
                "solver_consensus": item["answer"],
                "analyzer": {"skipped": True, "n_disagree": 0},
            }
        )
    return records


def cmd_qdr(args: argparse.Namespace) -> None:
    parsed_dir = _resolve_path(args.parsed_dir)
    output_dir = _resolve_path(args.output_dir)
    set_output_dir(output_dir)
    items = _limit_cases(_read_parsed(parsed_dir), args.limit)
    if args.mock:
        records = _mock_records(items)
        print("[qdr] offline fixture mode; no remote calls")
    else:
        api = load_api_config()
        solver_models = [m.strip() for m in args.solvers.split(",") if m.strip()] or None
        records = evaluate_qdr(
            items,
            api=api,
            workers=args.workers,
            solver_models=solver_models,
            disagree_threshold=args.disagree_threshold,
        )
    qdr_dir = output_dir / "qdr"
    qdr_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, qdr_dir / "items.jsonl")
    summary = summarize_qdr(records)
    _write_summary(summary, qdr_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Question parsing and QDR evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    parse = sub.add_parser("parse", help="Parse case Markdown into JSONL")
    parse.add_argument("--group-dir", action="append", required=True, metavar="NAME=PATH")
    parse.add_argument("--output-dir", default="scratch/question_quality")
    parse.add_argument("--limit", type=int, default=0)
    parse.set_defaults(func=cmd_parse)

    qdr = sub.add_parser("qdr", help="Run QDR with remote Judge or offline fixture mode")
    qdr.add_argument("--parsed-dir", required=True)
    qdr.add_argument("--output-dir", default="results/question_quality")
    qdr.add_argument("--workers", type=int, default=8)
    qdr.add_argument("--limit", type=int, default=0)
    qdr.add_argument("--solvers", default="")
    qdr.add_argument("--disagree-threshold", type=int, default=2)
    qdr.add_argument("--mock", action="store_true", help="Use deterministic local fixture decisions")
    qdr.set_defaults(func=cmd_qdr)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.func(arguments)
