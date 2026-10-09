"""Validate and summarize a reasoning evaluator output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path(args.output_dir)
    items = root / "items.jsonl"
    summary = root / "summary.json"
    if not items.is_file() or not summary.is_file():
        raise SystemExit("output directory must contain items.jsonl and summary.json")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    print(json.dumps({"items": sum(1 for _ in items.open(encoding="utf-8")), "models": sorted(payload.get("models", {}))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
