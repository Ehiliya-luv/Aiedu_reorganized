"""Simple visualization for anchor-based comparison results.

Lightweight matplotlib plots for:
  1. Win matrix heatmap
  2. Win rate bar chart
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _get_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _get_linear_segmented_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap


_TASK_DISPLAY_NAMES = {
    "考题生成": "Question Generation",
    "临床思维": "Clinical Reasoning",
    "病历标准化": "Medical Record Standardization",
    "病历综合评分": "Medical Record Scoring",
}


def _task_display(task: str) -> str:
    """Map Chinese task names to English for clean rendering."""
    return _TASK_DISPLAY_NAMES.get(task, task)


def _setup_font():
    """Configure matplotlib for English rendering only."""
    import matplotlib

    # Force English rendering — CJK fonts unavailable on headless servers
    matplotlib.rcParams["axes.unicode_minus"] = False


def _simplify_model_name(name: str) -> str:
    """Strip provider prefix to keep only the base model name.

    Examples:
        dashscope_dashscope-deepseek-v3.2 -> deepseek-v3.2
        provider_claude-haiku -> claude-haiku
        local_baichuan-m2-lora            -> baichuan-m2-lora
        qwen_qwen3-32b                    -> qwen3-32b
        baichuan_baichuan-m2              -> baichuan-m2
    """
    parts = name.split("_", 1)
    if len(parts) < 2:
        return name
    provider, model = parts
    if model.startswith(f"{provider}-"):
        stripped = model[len(provider) + 1 :]
        if len(stripped) >= 3:
            model = stripped
    return model


def plot_win_matrix(
    win_matrix,
    output_path: str,
    title: str = "Win Matrix",
) -> str:
    """Plot win matrix heatmap.

    Args:
        win_matrix: pd.DataFrame with win counts
        output_path: Path to save PNG
        title: Plot title

    Returns:
        Saved file path
    """
    import numpy as np

    plt = _get_plt()
    linear_segmented_colormap = _get_linear_segmented_colormap()
    _setup_font()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    df = win_matrix.copy()
    models = [_simplify_model_name(m) for m in df.index.tolist()]
    n = len(models)

    # Normalize for color (0-1)
    max_val = df.values.max()
    if max_val == 0:
        max_val = 1

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1)))

    # Custom colormap: white (0) -> blue (win)
    cmap = linear_segmented_colormap.from_list("win_cmap", ["#F8F9FA", "#5E72C8"], N=256)

    im = ax.imshow(df.values, cmap=cmap, vmin=0, vmax=max_val, aspect="auto")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(models, fontsize=10)

    # Annotate cells with values
    for i in range(n):
        for j in range(n):
            val = df.iloc[i, j]
            if i == j:
                text = "-"
                color = "#999999"
            else:
                text = str(int(val))
                color = "white" if val > max_val * 0.5 else "#333333"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Defender", fontsize=11)
    ax.set_ylabel("Attacker", fontsize=11)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Win Count", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info("[可视化] 胜负矩阵热力图已保存：%s", output_path)
    return output_path


def plot_win_rate(
    summary: dict,
    output_path: str,
    title: str = "Win Rate vs Baseline",
    baseline_name: str | None = None,
    count_label: str = "votes",
) -> str:
    """Plot win rate bar chart for each candidate vs baseline.

    Args:
        summary: {candidate_name: {"win_rate": float, "candidate_wins": int, "baseline_wins": int, "total_votes": int}}
        output_path: Path to save PNG
        title: Plot title
        baseline_name: Name of the baseline model to show as a reference bar
        count_label: Label for the count annotation, e.g. "votes" or "cases"

    Returns:
        Saved file path
    """
    import numpy as np

    plt = _get_plt()
    _setup_font()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    raw_candidates = list(summary.keys())
    candidates = [_simplify_model_name(c) for c in raw_candidates]
    win_rates = [summary[c]["win_rate"] * 100 for c in raw_candidates]
    total_votes = [summary[c]["total_votes"] for c in raw_candidates]

    show_baseline = baseline_name is not None
    n = len(candidates) + (1 if show_baseline else 0)
    fig, ax = plt.subplots(figsize=(max(7, n * 1.5), 5.5))

    x = np.arange(n)
    bar_width = min(0.7, max(0.4, 0.7 * n / max(n, 4)))

    # Color by win rate: red (<50%) -> gray (50%) -> blue (>50%)
    colors = []
    for rate in win_rates:
        if rate > 50:
            colors.append("#5E72C8")  # Blue
        elif rate < 50:
            colors.append("#D47663")  # Red
        else:
            colors.append("#999999")  # Gray

    # Insert baseline bar at the end
    if show_baseline:
        colors.append("#999999")
        win_rates = win_rates + [50.0]
        total_votes = total_votes + [0]
        simple_baseline = _simplify_model_name(baseline_name)
        candidates = candidates + [f"{simple_baseline}\n(Baseline)"]

    bars = ax.bar(x, win_rates, color=colors, width=bar_width, edgecolor="white", linewidth=1.2, zorder=3)

    # Annotate bars
    for bar, rate, votes in zip(bars, win_rates, total_votes):
        height = bar.get_height()
        label = f"{rate:.1f}%"
        if votes > 0:
            label += f"\n({int(votes)} {count_label})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.5,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    # 50% reference line
    ax.axhline(y=50, color="#666666", linestyle="--", linewidth=1.5, alpha=0.6, zorder=1)
    if not show_baseline:
        ax.text(n - 0.5, 52, "50% (tie)", ha="right", va="bottom", fontsize=9, color="#666666", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(candidates, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Win Rate (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, max(100, max(win_rates) + 15))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info("[可视化] 胜率柱状图已保存：%s", output_path)
    return output_path


def plot_ranking_with_ci(
    stats_df,
    output_path: str,
    title: str = "Model Ranking",
    score_col: str = "Score",
    lower_col: str = "Boot_CI_Lower",
    upper_col: str = "Boot_CI_Upper",
    baseline_name: str | None = None,
) -> str:
    """Plot ranking with confidence intervals (vertical dot + errorbar).

    Args:
        stats_df: DataFrame with Score and CI columns
        output_path: Path to save PNG
        title: Plot title
        score_col: Column name for score (0-1)
        lower_col: Column name for CI lower bound
        upper_col: Column name for CI upper bound
        baseline_name: Name of the baseline model to show as a reference point

    Returns:
        Saved file path
    """
    import numpy as np

    plt = _get_plt()
    _setup_font()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    df = stats_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df["Model"] = df["Model"].apply(_simplify_model_name)
    n = len(df)
    show_baseline = baseline_name is not None
    total_n = n + (1 if show_baseline else 0)

    fig, ax = plt.subplots(figsize=(max(8, total_n * 1.3), 6))

    x = np.arange(total_n)
    scores = df[score_col].values * 100  # Convert to percentage
    ci_lower = df[lower_col].values * 100
    ci_upper = df[upper_col].values * 100

    yerr_lower = np.maximum(scores - ci_lower, 0)
    yerr_upper = np.maximum(ci_upper - scores, 0)

    # Plot candidates
    ax.errorbar(
        x[:n],
        scores,
        yerr=[yerr_lower, yerr_upper],
        fmt="o",
        markersize=10,
        color="#2E86AB",
        elinewidth=2.5,
        capsize=8,
        capthick=2,
        alpha=0.8,
        zorder=3,
    )

    # Annotate score values
    for pos, score in zip(x[:n], scores):
        ax.text(
            pos + 0.15,
            score,
            f"{score:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Plot baseline reference point at the end
    if show_baseline:
        ax.plot(
            x[-1],
            50.0,
            marker="s",
            markersize=12,
            color="#999999",
            zorder=3,
        )
        ax.text(
            x[-1] + 0.15,
            50.0,
            "50.0%",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#666666",
        )

    # 50% reference line
    ax.axhline(y=50, color="#666666", linestyle="--", linewidth=1.5, alpha=0.5, zorder=1)

    ax.set_xticks(x)
    model_labels = [m if len(m) <= 18 else m[:15] + "..." for m in df["Model"]]
    if show_baseline:
        simple_baseline = _simplify_model_name(baseline_name)
        model_labels.append(f"{simple_baseline}\n(Baseline)")
    ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=10)

    ax.set_ylabel("Win Rate vs Baseline (%)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    y_max = max(100, max(ci_upper) + 10)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info("[可视化] 排名图（带 CI）已保存：%s", output_path)
    return output_path


def visualize_anchor_results(
    win_matrix,
    summary: dict,
    output_dir: str,
    task: str = "",
    stats_df=None,
    baseline_name: str | None = None,
    score_col: str = "Score",
    lower_col: str = "Boot_CI_Lower",
    upper_col: str = "Boot_CI_Upper",
    ranking_name: str = "ranking_ci",
    count_label: str = "votes",
) -> dict:
    """Generate all plots for anchor comparison results.

    Args:
        win_matrix: Win counts matrix
        summary: Per-candidate summary dict
        output_dir: Directory to save plots
        task: Task name prefix
        stats_df: Optional DataFrame with score and CI columns for ranking plot
        baseline_name: Name of the baseline model to show in plots
        score_col: Column name for score (0-1)
        lower_col: Column name for CI lower bound
        upper_col: Column name for CI upper bound
        ranking_name: File-name suffix for the ranking plot, without extension
        count_label: Label for win-rate bar counts, e.g. "votes" or "cases"

    Returns:
        {"win_matrix_path": str, "win_rate_path": str, "ranking_path": str (optional)}
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{task}_" if task else ""
    display_task = _task_display(task)

    matrix_path = os.path.join(output_dir, f"{prefix}win_matrix.png")
    rate_path = os.path.join(output_dir, f"{prefix}win_rate.png")

    plot_win_matrix(win_matrix, matrix_path, title=f"{display_task} - Win Matrix" if display_task else "Win Matrix")
    plot_win_rate(
        summary,
        rate_path,
        title=f"{display_task} - Win Rate vs Baseline" if display_task else "Win Rate vs Baseline",
        baseline_name=baseline_name,
        count_label=count_label,
    )

    result = {
        "win_matrix_path": matrix_path,
        "win_rate_path": rate_path,
    }

    # Ranking plot with CI
    if stats_df is not None and not stats_df.empty:
        ranking_path = os.path.join(output_dir, f"{prefix}{ranking_name}.png")
        plot_ranking_with_ci(
            stats_df,
            ranking_path,
            title=f"{display_task} - Model Ranking with 95% CI" if display_task else "Model Ranking with 95% CI",
            score_col=score_col,
            lower_col=lower_col,
            upper_col=upper_col,
            baseline_name=baseline_name,
        )
        result["ranking_path"] = ranking_path

    return result


__all__ = [
    "plot_win_matrix",
    "plot_win_rate",
    "plot_ranking_with_ci",
    "visualize_anchor_results",
]
