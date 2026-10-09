"""Statistics for anchor-based comparison: Anchor Score + Bootstrap CI.

Lightweight alternative to full Bradley-Terry (which needs all pairwise data).
For anchor mode, we only have candidate-vs-baseline comparisons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_anchor_score(
    baseline_wins: int,
    candidate_wins: int,
) -> dict:
    """Compute Anchor Score and basic statistics.

    Args:
        baseline_wins: Wins for baseline model
        candidate_wins: Wins for candidate model

    Returns:
        {
            "score": float,          # candidate win rate (0-1)
            "baseline_wins": int,
            "candidate_wins": int,
            "total_votes": int,
            "se": float,             # standard error (binomial)
            "ci_lower": float,       # 95% CI lower bound (Wilson)
            "ci_upper": float,       # 95% CI upper bound (Wilson)
        }
    """
    total = baseline_wins + candidate_wins
    if total == 0:
        return {
            "score": 0.0,
            "baseline_wins": baseline_wins,
            "candidate_wins": candidate_wins,
            "total_votes": 0,
            "se": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }

    p = candidate_wins / total

    # Standard error (binomial proportion)
    se = np.sqrt(p * (1 - p) / total) if total > 0 else 0.0

    # Wilson confidence interval (more accurate for small samples)
    z = 1.96  # 95% CI
    if total > 0:
        denominator = 1 + z**2 / total
        centre = (p + z**2 / (2 * total)) / denominator
        half_width = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
        ci_lower = max(0.0, centre - half_width)
        ci_upper = min(1.0, centre + half_width)
    else:
        ci_lower = 0.0
        ci_upper = 0.0

    return {
        "score": float(p),
        "baseline_wins": baseline_wins,
        "candidate_wins": candidate_wins,
        "total_votes": total,
        "se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def bootstrap_anchor_ci(
    baseline_wins: int,
    candidate_wins: int,
    n_boot: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for anchor score.

    Args:
        baseline_wins: Baseline win count
        candidate_wins: Candidate win count
        n_boot: Number of bootstrap samples
        seed: Random seed

    Returns:
        (ci_lower, ci_upper) — 95% percentile bootstrap CI
    """
    total = baseline_wins + candidate_wins
    if total == 0:
        return 0.0, 0.0

    rng = np.random.RandomState(seed)
    samples = []

    for _ in range(n_boot):
        # Resample votes with replacement
        boot_candidate = rng.binomial(total, candidate_wins / total)
        boot_score = boot_candidate / total if total > 0 else 0.0
        samples.append(boot_score)

    ci_lower = float(np.percentile(samples, 2.5))
    ci_upper = float(np.percentile(samples, 97.5))
    return ci_lower, ci_upper


def _bootstrap_binary_mean(
    successes: int,
    total: int,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict:
    """Bootstrap a binary success rate and return only bootstrap-derived values."""
    if total <= 0:
        return {"score": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    rng = np.random.RandomState(seed)
    samples = rng.binomial(total, successes / total, size=n_boot) / total
    return {
        "score": float(np.mean(samples)),
        "ci_lower": float(np.percentile(samples, 2.5)),
        "ci_upper": float(np.percentile(samples, 97.5)),
    }


def bootstrap_vote_stats(
    baseline_wins: int,
    candidate_wins: int,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict:
    """Bootstrap candidate vote share from aggregate vote counts."""
    total = int(baseline_wins) + int(candidate_wins)
    return _bootstrap_binary_mean(int(candidate_wins), total, n_boot=n_boot, seed=seed)


def bootstrap_case_majority_stats(
    case_results: list[int],
    n_boot: int = 10000,
    seed: int = 42,
) -> dict:
    """Bootstrap candidate case-majority share from per-case winners.

    Args:
        case_results: 1 for candidate case win, 0 for baseline case win.
                      Tied or invalid cases should be filtered before calling.
    """
    total = len(case_results)
    if total == 0:
        return {"score": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    values = np.asarray(case_results, dtype=float)
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, total, size=(n_boot, total))
    samples = values[indices].mean(axis=1)
    return {
        "score": float(np.mean(samples)),
        "ci_lower": float(np.percentile(samples, 2.5)),
        "ci_upper": float(np.percentile(samples, 97.5)),
    }


def compute_checkpoint_bootstrap_stats(
    checkpoint: dict,
    *,
    n_boot: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute vote-level and case-majority bootstrap stats from a checkpoint.

    The output intentionally avoids raw probability columns such as
    candidate_wins / total_votes. Reported score columns are bootstrap means.
    """
    records = []
    for offset, (candidate, payload) in enumerate(checkpoint.get("candidates", {}).items()):
        baseline_votes = 0
        candidate_votes = 0
        baseline_case_wins = 0
        candidate_case_wins = 0
        tied_cases = 0
        invalid_cases = 0
        case_results: list[int] = []

        cases = payload.get("cases", {})
        for case_payload in cases.values():
            baseline_wins = int(case_payload.get("baseline_wins", 0))
            candidate_wins = int(case_payload.get("candidate_wins", 0))
            total_votes = baseline_wins + candidate_wins
            baseline_votes += baseline_wins
            candidate_votes += candidate_wins

            if total_votes <= 0:
                invalid_cases += 1
                continue
            if candidate_wins > baseline_wins:
                candidate_case_wins += 1
                case_results.append(1)
            elif baseline_wins > candidate_wins:
                baseline_case_wins += 1
                case_results.append(0)
            else:
                tied_cases += 1

        vote_stats = bootstrap_vote_stats(
            baseline_votes,
            candidate_votes,
            n_boot=n_boot,
            seed=seed + offset,
        )
        case_stats = bootstrap_case_majority_stats(
            case_results,
            n_boot=n_boot,
            seed=seed + 1009 + offset,
        )

        records.append(
            {
                "Model": candidate,
                "Case_Bootstrap_Score": case_stats["score"],
                "Case_CI_Lower": case_stats["ci_lower"],
                "Case_CI_Upper": case_stats["ci_upper"],
                "Vote_Bootstrap_Score": vote_stats["score"],
                "Vote_CI_Lower": vote_stats["ci_lower"],
                "Vote_CI_Upper": vote_stats["ci_upper"],
                "Baseline_Case_Wins": baseline_case_wins,
                "Candidate_Case_Wins": candidate_case_wins,
                "Tied_Cases": tied_cases,
                "Invalid_Cases": invalid_cases,
                "Valid_Cases": len(case_results),
                "Total_Cases": len(cases),
                "Baseline_Wins": baseline_votes,
                "Candidate_Wins": candidate_votes,
                "Total_Votes": baseline_votes + candidate_votes,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values(
        ["Case_Bootstrap_Score", "Vote_Bootstrap_Score"],
        ascending=False,
    ).reset_index(drop=True)


def checkpoint_to_case_majority_matrix(
    checkpoint: dict,
    baseline: str,
) -> tuple[pd.DataFrame, dict]:
    """Build a win matrix and summary using per-case majority winners."""
    candidates = list(checkpoint.get("candidates", {}).keys())
    models = [baseline] + candidates
    win_matrix = pd.DataFrame(0, index=models, columns=models)
    summary = {}

    for candidate, payload in checkpoint.get("candidates", {}).items():
        baseline_case_wins = 0
        candidate_case_wins = 0
        tied_cases = 0
        invalid_cases = 0

        for case_payload in payload.get("cases", {}).values():
            baseline_wins = int(case_payload.get("baseline_wins", 0))
            candidate_wins = int(case_payload.get("candidate_wins", 0))
            total_votes = baseline_wins + candidate_wins
            if total_votes <= 0:
                invalid_cases += 1
            elif candidate_wins > baseline_wins:
                candidate_case_wins += 1
            elif baseline_wins > candidate_wins:
                baseline_case_wins += 1
            else:
                tied_cases += 1

        valid_cases = baseline_case_wins + candidate_case_wins
        win_matrix.loc[baseline, candidate] = baseline_case_wins
        win_matrix.loc[candidate, baseline] = candidate_case_wins
        summary[candidate] = {
            "baseline_wins": baseline_case_wins,
            "candidate_wins": candidate_case_wins,
            "total_votes": valid_cases,
            "tied_cases": tied_cases,
            "invalid_cases": invalid_cases,
            "win_rate": candidate_case_wins / valid_cases if valid_cases else 0.0,
        }

    return win_matrix, summary


def compute_all_stats(
    win_matrix: pd.DataFrame,
    baseline: str,
    n_boot: int = 10000,
) -> pd.DataFrame:
    """Compute Anchor Score statistics for all candidates vs baseline.

    Returns:
        DataFrame with columns:
            Model, Score, SE, CI_Lower, CI_Upper, Boot_CI_Lower, Boot_CI_Upper,
            Baseline_Wins, Candidate_Wins, Total_Votes
    """
    models = [m for m in win_matrix.index if m != baseline]
    records = []

    for candidate in models:
        baseline_wins = int(win_matrix.loc[baseline, candidate])
        candidate_wins = int(win_matrix.loc[candidate, baseline])

        stats = compute_anchor_score(baseline_wins, candidate_wins)
        boot_lower, boot_upper = bootstrap_anchor_ci(baseline_wins, candidate_wins, n_boot=n_boot)

        records.append(
            {
                "Model": candidate,
                "Score": stats["score"],
                "SE": stats["se"],
                "CI_Lower": stats["ci_lower"],
                "CI_Upper": stats["ci_upper"],
                "Boot_CI_Lower": boot_lower,
                "Boot_CI_Upper": boot_upper,
                "Baseline_Wins": baseline_wins,
                "Candidate_Wins": candidate_wins,
                "Total_Votes": stats["total_votes"],
            }
        )

    df = pd.DataFrame(records)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df


__all__ = [
    "compute_anchor_score",
    "bootstrap_anchor_ci",
    "bootstrap_vote_stats",
    "bootstrap_case_majority_stats",
    "compute_checkpoint_bootstrap_stats",
    "checkpoint_to_case_majority_matrix",
    "compute_all_stats",
]
