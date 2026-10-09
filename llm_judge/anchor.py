"""Anchor-based LLM Judge: compare all candidates against a single baseline.

Compare N models using only (N-1) pairs: each candidate vs baseline.
Supports multi-threading, repeated votes per case, and reason collection.

Extracted and simplified from utils/com_exp/judge_api.py.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .client import JudgeAuthError
from .pairwise import PairwiseJudge

logger = logging.getLogger(__name__)


def _compare_single_case(
    judge: PairwiseJudge,
    task: str,
    case_id: str,
    raw_case: str,
    baseline_output: str,
    candidate_output: str,
    repeats: int,
    candidate_name: str,
) -> tuple[str, int, int, list[dict]]:
    """Compare one candidate vs baseline on a single case.

    Returns:
        (case_id, wins_baseline, wins_candidate, reasons)
        reasons = [{"winner": "A"/"B", "reason": "..."}, ...]
    """
    wins_baseline, wins_candidate, reasons = judge.compare_with_repeats(
        task=task,
        raw_case=raw_case,
        output_a=baseline_output,
        output_b=candidate_output,
        repeats=repeats,
        debias=True,
    )
    return case_id, wins_baseline, wins_candidate, reasons


class AnchorJudge:
    """Anchor-based judge: all candidates compared against one baseline.

    Usage:
        judge = AnchorJudge(
            baseline="base_model",
            model="deepseek/deepseek-v3.2",
            api_base=os.environ["AIEDU_JUDGE_API_BASE"],
            api_key=os.environ["AIEDU_JUDGE_API_KEY"],
        )

        results = judge.compare_all(
            task="病历标准化",
            raw_cases={"case1": "患者男...", "case2": "患者女..."},
            model_outputs={
                "base_model": {"case1": "...", "case2": "..."},
                "candidate_a": {"case1": "...", "case2": "..."},
                "candidate_b": {"case1": "...", "case2": "..."},
            },
            repeats=3,
            workers=4,
            output_dir="./results",
            visualize=True,
        )

        # results contains:
        #   "win_matrix": pd.DataFrame
        #   "summary": dict — per-candidate stats
        #   "stats_df": pd.DataFrame — Anchor Score + CI
        #   "reasons": dict — collected reasons per candidate
        #   "viz_paths": dict — paths to saved plots
    """

    def __init__(
        self,
        baseline: str,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        top_p: float | None = 1.0,
        max_tokens: int = 2048,
        timeout: int = 3000,
        local_mode: bool = False,
        rate_limit_max_retries: int = 8,
        transient_retries: int = 4,
        prompt_variant: str = "current",
    ) -> None:
        # Resolve provider if not explicitly provided
        if not api_base or not api_key:
            from llm_judge import _resolve_judge_credentials
            resolved_base, resolved_key = _resolve_judge_credentials(model)
            api_base = api_base or resolved_base
            api_key = api_key or resolved_key
        self.baseline = baseline
        self.judge = PairwiseJudge(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            local_mode=local_mode,
            rate_limit_max_retries=rate_limit_max_retries,
            transient_retries=transient_retries,
            prompt_variant=prompt_variant,
        )

    def compare_pair(
        self,
        task: str,
        raw_cases: dict[str, str],
        model_outputs: dict[str, dict[str, str]],
        candidate: str,
        *,
        repeats: int = 3,
        workers: int = 4,
        case_ids: list[str] | None = None,
        on_case_complete: callable | None = None,
    ) -> tuple[int, int, list[dict]]:
        """Compare one candidate against baseline across all common cases.

        Args:
            case_ids: Optional subset of case IDs to evaluate. If None,
                      all common cases between baseline and candidate are used.
            on_case_complete: Optional callback invoked after each case finishes.
                              Signature: (case_id, baseline_wins, candidate_wins, reasons)

        Returns:
            (total_baseline_wins, total_candidate_wins, all_reasons)
        """
        baseline_outputs = model_outputs[self.baseline]
        candidate_outputs = model_outputs[candidate]

        if case_ids is not None:
            common_cases = sorted(
                c
                for c in case_ids
                if c in baseline_outputs and c in candidate_outputs and c in raw_cases
            )
        else:
            common_cases = sorted(set(baseline_outputs.keys()) & set(candidate_outputs.keys()))
        if not common_cases:
            logger.warning("[AnchorJudge] No common cases between %s and %s", self.baseline, candidate)
            return 0, 0, []

        total_baseline = 0
        total_candidate = 0
        all_reasons = []

        args_list = [
            (
                self.judge,
                task,
                case_id,
                raw_cases[case_id],
                baseline_outputs[case_id],
                candidate_outputs[case_id],
                repeats,
                candidate,
            )
            for case_id in common_cases
            if case_id in raw_cases
        ]

        logger.info(
            "[AnchorJudge] %s vs %s: %d cases × %d repeats = %d votes",
            candidate,
            self.baseline,
            len(args_list),
            repeats,
            len(args_list) * repeats,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_compare_single_case, *args) for args in args_list]

            iterator = concurrent.futures.as_completed(futures)
            if tqdm is not None:
                iterator = tqdm(
                    iterator,
                    total=len(futures),
                    desc=f"{candidate[:30]:30}",
                    unit="case",
                    leave=True,
                    dynamic_ncols=True,
                )

            completed = 0
            for future in iterator:
                try:
                    case_id, wins_baseline, wins_candidate, reasons = future.result()
                    total_baseline += wins_baseline
                    total_candidate += wins_candidate
                    all_reasons.extend(reasons)
                    completed += 1
                    if on_case_complete:
                        on_case_complete(case_id, wins_baseline, wins_candidate, reasons)
                    if completed % 10 == 0 or completed == len(args_list):
                        logger.info(
                            "[AnchorJudge] %s: %d/%d cases done, current score %s=%d %s=%d",
                            candidate,
                            completed,
                            len(args_list),
                            self.baseline,
                            total_baseline,
                            candidate,
                            total_candidate,
                        )
                except (JudgeAuthError, KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    logger.warning("[AnchorJudge] Case failed: %s: %s", type(exc).__name__, exc)
                    continue

        logger.info(
            "[AnchorJudge] %s vs %s final: %s=%d %s=%d",
            candidate,
            self.baseline,
            self.baseline,
            total_baseline,
            candidate,
            total_candidate,
        )
        return total_baseline, total_candidate, all_reasons

    def compare_all(
        self,
        task: str,
        raw_cases: dict[str, str],
        model_outputs: dict[str, dict[str, str]],
        *,
        repeats: int = 3,
        workers: int = 4,
        output_dir: str | None = None,
        visualize: bool = True,
        n_boot: int = 10000,
        checkpoint_path: str | None = None,
    ) -> dict:
        """Compare all candidates against baseline.

        Args:
            task:         Task name
            raw_cases:    {case_id: raw_text}
            model_outputs: {model_name: {case_id: output_text}}
            repeats:      Votes per case (default 3)
            workers:      ThreadPool max workers (default 4)
            output_dir:   Directory to save plots (default: no plots)
            visualize:    Whether to generate plots (default True)
            n_boot:       Bootstrap samples for CI (default 10000)
            checkpoint_path: Path to JSON checkpoint for resume support.

        Returns:
            dict with:
                "win_matrix": pd.DataFrame — win counts
                "summary":    dict — per-candidate summary
                "stats_df":   pd.DataFrame — Anchor Score + CI
                "reasons":    dict — collected reasons per candidate
                "viz_paths":  dict — paths to saved plots
        """
        if self.baseline not in model_outputs:
            raise ValueError(f"Baseline model '{self.baseline}' not found in model_outputs")

        models = list(model_outputs.keys())
        candidates = [m for m in models if m != self.baseline]

        if not candidates:
            raise ValueError("No candidate models found (only baseline)")

        # Initialize win matrix
        win_matrix = pd.DataFrame(0, index=models, columns=models)

        summary = {}
        all_reasons = {}

        ckpt = None
        if checkpoint_path:
            from .checkpoint import CheckpointManager

            ckpt = CheckpointManager(checkpoint_path)
            ckpt.set_metadata(task, self.baseline)
            logger.info("[AnchorJudge] Checkpoint enabled: %s", checkpoint_path)

        baseline_outputs = model_outputs[self.baseline]

        # Compare each candidate vs baseline
        for candidate in candidates:
            baseline_wins, candidate_wins, reasons = 0, 0, []

            if ckpt:
                completed_cases = ckpt.get_completed_cases(candidate)
                common_cases = sorted(
                    set(baseline_outputs.keys()) & set(model_outputs[candidate].keys())
                )
                remaining_cases = [c for c in common_cases if c not in completed_cases]

                if not common_cases:
                    logger.warning(
                        "[AnchorJudge] No common cases between %s and %s", self.baseline, candidate
                    )
                    baseline_wins, candidate_wins, reasons = 0, 0, []
                elif not remaining_cases:
                    logger.info(
                        "[AnchorJudge] %s: all %d cases found in checkpoint, skipping",
                        candidate,
                        len(common_cases),
                    )
                    baseline_wins, candidate_wins, reasons = ckpt.get_results(candidate)
                else:
                    if completed_cases:
                        logger.info(
                            "[AnchorJudge] %s: resuming %d/%d cases from checkpoint",
                            candidate,
                            len(remaining_cases),
                            len(common_cases),
                        )
                    self.compare_pair(
                        task=task,
                        raw_cases=raw_cases,
                        model_outputs=model_outputs,
                        candidate=candidate,
                        repeats=repeats,
                        workers=workers,
                        case_ids=remaining_cases,
                        on_case_complete=lambda cid, bw, cw, r: ckpt.record_case(
                            candidate, cid, bw, cw, r
                        ),
                    )
                    baseline_wins, candidate_wins, reasons = ckpt.get_results(candidate)
            else:
                baseline_wins, candidate_wins, reasons = self.compare_pair(
                    task=task,
                    raw_cases=raw_cases,
                    model_outputs=model_outputs,
                    candidate=candidate,
                    repeats=repeats,
                    workers=workers,
                )

            win_matrix.loc[self.baseline, candidate] = baseline_wins
            win_matrix.loc[candidate, self.baseline] = candidate_wins

            total_votes = baseline_wins + candidate_wins
            summary[candidate] = {
                "baseline_wins": baseline_wins,
                "candidate_wins": candidate_wins,
                "total_votes": total_votes,
                "win_rate": candidate_wins / total_votes if total_votes > 0 else 0,
            }
            all_reasons[candidate] = reasons

        # Compute statistics
        try:
            from .stats import compute_all_stats

            stats_df = compute_all_stats(win_matrix, self.baseline, n_boot=n_boot)
        except (JudgeAuthError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            logger.warning("[AnchorJudge] Stats computation failed: %s", exc)
            stats_df = None

        result = {
            "win_matrix": win_matrix,
            "summary": summary,
            "stats_df": stats_df,
            "reasons": all_reasons,
        }

        # Visualization
        if visualize and output_dir:
            try:
                from .viz import visualize_anchor_results

                viz_paths = visualize_anchor_results(
                    win_matrix=win_matrix,
                    summary=summary,
                    output_dir=output_dir,
                    task=task,
                    stats_df=stats_df,
                    baseline_name=self.baseline,
                )
                result["viz_paths"] = viz_paths
            except ImportError as exc:
                logger.warning("[AnchorJudge] Visualization skipped: %s", exc)
                result["viz_paths"] = {}
        else:
            result["viz_paths"] = {}

        return result


__all__ = ["AnchorJudge"]
