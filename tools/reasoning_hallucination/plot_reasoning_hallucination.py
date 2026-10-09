"""Plot a private reasoning-summary CSV produced by the evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.summary).sort_values("case_consistency")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(frame["model"], frame["case_consistency"] * 100, color="#00A6BF")
    ax.set_xlabel("Reasoning consistency (%)")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
