"""Run the complete public, offline smoke test on synthetic inputs."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch" / "fixture_validation"


def run(*args: str) -> str:
    command = [sys.executable, *args]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")
    return result.stdout


def scan_public_tree() -> None:
    forbidden = [
        re.compile(r"(?i)-----begin [a-z ]*private key-----"),
        re.compile(r"(?i)(?:api[_-]?key|authorization|token)\s*[:=]\s*['\"][^'\"]{12,}"),
        re.compile(r"(?i)sk-[A-Za-z0-9]{12,}"),
        re.compile(r"(?i)[A-Z]:\\Users\\"),
    ]
    ignored_parts = {".git", ".venv", "__pycache__", ".pytest_cache", "scratch"}
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ignored_parts.intersection(path.parts):
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".toml", ".yml", ".yaml", ".example"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in forbidden:
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT)} matches {pattern.pattern}")
    if findings:
        raise AssertionError("Potential private material found:\n" + "\n".join(findings))


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from dpo.utils.dpo_data import load_dpo_samples

    dpo_samples = load_dpo_samples(str(ROOT / "fixtures" / "dpo" / "dpo_pairs.jsonl"))
    if len(dpo_samples) != 2:
        raise AssertionError(f"Unexpected DPO fixture count: {len(dpo_samples)}")
    parse_output = SCRATCH / "quality"
    run(
        str(ROOT / "question_quality_main.py"),
        "parse",
        "--group-dir",
        "demo=fixtures/question_inputs/demo",
        "--output-dir",
        str(parse_output),
    )
    parsed = parse_output / "parsed" / "demo.jsonl"
    if not parsed.exists() or not parsed.read_text(encoding="utf-8").strip():
        raise AssertionError("Question parser produced no JSONL")

    qdr_output = SCRATCH / "qdr"
    qdr_stdout = run(
        str(ROOT / "question_quality_main.py"),
        "qdr",
        "--parsed-dir",
        str(parse_output / "parsed"),
        "--output-dir",
        str(qdr_output),
        "--mock",
    )
    summary_path = qdr_output / "qdr" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("demo", {}).get("judged") != 1:
        raise AssertionError(f"Unexpected fixture summary: {summary}")

    judge_result = run(
        str(ROOT / "question_llm_judge_main.py"),
        "--data-dir",
        "fixtures/judge_outputs",
        "--baseline",
        "baseline",
        "--tasks",
        "question",
        "--dry-run",
    )
    if "candidate" not in judge_result or "BASELINE" not in judge_result:
        raise AssertionError("Judge dry-run did not discover the fixture models")

    reasoning_output = SCRATCH / "reasoning"
    run(
        str(ROOT / "reasoning_quality_main.py"),
        "--mock",
        "--data-dir",
        "fixtures/reasoning_outputs",
        "--cases-dir",
        "fixtures/reasoning_cases",
        "--output-dir",
        str(reasoning_output),
    )
    reasoning_summary = json.loads((reasoning_output / "summary.json").read_text(encoding="utf-8"))
    if sorted(reasoning_summary.get("models", {})) != ["baseline", "candidate"]:
        raise AssertionError("Reasoning fixture summary is incomplete")

    run(
        str(ROOT / "paper_reproduction_main.py"),
        "--only",
        "question-quality",
        "--output-dir",
        str(SCRATCH / "figures"),
    )
    expected = [SCRATCH / "figures" / f"question_quality.{ext}" for ext in ("pdf", "png", "svg")]
    if not all(path.exists() and path.stat().st_size > 0 for path in expected):
        raise AssertionError("Question quality figures were not generated")

    scan_public_tree()
    report = {
        "status": "passed",
        "dpo_fixture_records": len(dpo_samples),
        "question_items": 1,
        "judge_mode": "dry-run",
        "qdr_mode": "mock",
        "reasoning_mode": "mock",
        "figures": [str(path.relative_to(ROOT)) for path in expected],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
