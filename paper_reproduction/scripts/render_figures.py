"""Render paper-facing figures from frozen aggregate CSVs.

The inputs contain model-level or task-level aggregates only.  This renderer
never reads cases, model outputs, Judge reasons, or private timing records.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_LABELS = {
    "answer_key_accuracy": "Answer-key accuracy",
    "option_set_adequacy": "Option-set validity",
    "stem_clarity": "Stem clarity",
}
COLORS = ["#00A6BF", "#70787D", "#4C8A5A", "#D1645F", "#D59A4A", "#6F4E84", "#7FB68A", "#D68985", "#4F8DB6"]


def _save(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.{ext}" for ext in ("pdf", "png", "svg")]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def render_question_quality(data_dir: Path, output_dir: Path) -> list[Path]:
    frame = pd.read_csv(data_dir / "question_quality.csv")
    missing = set(METRIC_LABELS) - set(frame.columns)
    if missing:
        raise ValueError(f"question_quality.csv is missing columns: {sorted(missing)}")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=False)
    for ax, (column, label) in zip(axes, METRIC_LABELS.items()):
        view = frame.sort_values(column, ascending=False)
        x = np.arange(len(view))
        ax.bar(x, view[column] * 100, color=[COLORS[i % len(COLORS)] for i in range(len(view))])
        ax.set_title(label, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(view["model"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Percent")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.suptitle("Question quality metrics", fontweight="bold")
    fig.subplots_adjust(top=0.84, bottom=0.34, left=0.05, right=0.99, wspace=0.28)
    return _save(fig, output_dir, "question_quality")


def render_preference(data_dir: Path, output_dir: Path) -> list[Path]:
    frame = pd.read_csv(data_dir / "question_preference.csv").sort_values("preference_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    y = np.arange(len(frame))
    ax.barh(y, frame["preference_rate"] * 100, color="#00A6BF")
    ax.errorbar(
        frame["preference_rate"] * 100,
        y,
        xerr=[(frame["preference_rate"] - frame["ci_low"]) * 100, (frame["ci_high"] - frame["preference_rate"]) * 100],
        fmt="none",
        ecolor="#25292D",
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(frame["model"])
    ax.set_xlabel("Candidate preference rate (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.set_title("Result evaluation by remote LLM Judge", fontweight="bold")
    fig.tight_layout()
    return _save(fig, output_dir, "question_preference")


def render_reasoning(data_dir: Path, output_dir: Path) -> list[Path]:
    frame = pd.read_csv(data_dir / "reasoning_summary.csv")
    frame = frame.sort_values("case_consistency", ascending=True)
    labels = {
        "api_baichuan_m2_norag_norefl": "Baichuan-M2",
        "base_lora1800_norag_norefl": "ShuTong",
        "lora2000_norag_norefl": "ShuTong-DPO-2000",
        "dashscope_deepseek_v3_2": "DeepSeek-V3.2",
        "dashscope_qwen3_14b": "Qwen3-14B",
        "huatuo3_32b": "Huatuo3-32B",
        "huatuo_o1_7b": "Huatuo-O1-7B",
        "qwen25_32b": "Qwen2.5-32B",
        "qwen3_32b": "Qwen3-32B",
    }
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    y = np.arange(len(frame))
    values = frame["case_consistency"].to_numpy()
    ax.barh(y, values * 100, color="#00A6BF")
    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(value, value) for value in frame["group"]])
    ax.set_xlabel("Case consistency (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.set_title("Reasoning consistency", fontweight="bold")
    fig.tight_layout()
    return _save(fig, output_dir, "reasoning_consistency")


def render_workflow_time(data_dir: Path, output_dir: Path) -> list[Path]:
    frame = pd.read_csv(data_dir / "workflow_time_summary.csv")
    labels = frame["task"].astype(str).tolist()
    x = np.arange(len(frame))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, statistic in zip(axes, ("median", "mean")):
        ax.bar(x - width / 2, frame[f"system_{statistic}_min"], width, label="System", color="#00A6BF")
        ax.bar(x + width / 2, frame[f"clinician_{statistic}_min"], width, label="Clinician estimate", color="#70787D")
        ax.set_title(statistic.title())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Minutes")
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.legend(frameon=False)
    fig.suptitle("Workflow time comparison", fontweight="bold")
    fig.tight_layout()
    return _save(fig, output_dir, "workflow_time")


def render_all(data_dir: Path, output_dir: Path, selected: set[str]) -> list[Path]:
    functions = {
        "question-quality": render_question_quality,
        "preference": render_preference,
        "reasoning": render_reasoning,
        "workflow-time": render_workflow_time,
    }
    unknown = selected - set(functions)
    if unknown:
        raise ValueError(f"Unknown figure names: {sorted(unknown)}")
    paths: list[Path] = []
    for name, function in functions.items():
        if not selected or name in selected:
            paths.extend(function(data_dir, output_dir))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="paper_reproduction/plot_data")
    parser.add_argument("--output-dir", default="results/paper_figures")
    parser.add_argument("--only", default="", help="Comma-separated: question-quality,preference,reasoning,workflow-time")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    selected = {item.strip() for item in args.only.split(",") if item.strip()}
    for path in render_all(data_dir, output_dir, selected):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
