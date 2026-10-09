"""Minimal Question-quality visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_question_quality(csv_path: str | Path, output_path: str | Path) -> Path:
    frame = pd.read_csv(csv_path)
    columns = ["answer_key_accuracy", "option_set_adequacy", "stem_clarity"]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing Question quality columns: {missing}")
    fig, ax = plt.subplots(figsize=(9, 5))
    frame.set_index("model")[columns].plot(kind="bar", ax=ax, ylim=(0, 1))
    ax.set_ylabel("Rate")
    ax.set_xlabel("Model")
    ax.legend(title="Question quality")
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
