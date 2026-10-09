"""LLM Judge: pairwise LLM-as-a-Judge over medical tasks."""

import os

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:
    import warnings
    warnings.warn(
        "[llm_judge] python-dotenv not installed; .env will NOT be loaded. "
        "Run: pip install python-dotenv",
        stacklevel=2,
    )

_MAINLAND_MODELS = ("qwen", "baichuan", "huatuo", "glm", "minimax", "kimi", "longcat", "mimo", "hunyuan")


def _resolve_judge_credentials(model_name: str):
    """Return (api_base, api_key) based on model name prefix routing.

    - deepseek-* → DEEPSEEK_API_KEY / DEEPSEEK_API_BASE
    - qwen/baichuan/huatuo/glm/minimax/kimi/longcat/mimo/hunyuan → DASHSCOPE_API_KEY / DASHSCOPE_API_BASE
    - other models → AIEDU_JUDGE_API_KEY / AIEDU_JUDGE_API_BASE

    No provider URL is embedded in the repository.  The caller supplies the
    URL and key through the environment or directly to ``AnchorJudge``.
    """
    model_lower = model_name.lower()
    if model_lower.startswith("deepseek"):
        return (
            os.environ.get("DEEPSEEK_API_BASE", ""),
            os.environ.get("DEEPSEEK_API_KEY", ""),
        )
    if any(model_lower.startswith(prefix) for prefix in _MAINLAND_MODELS):
        return (
            os.environ.get("DASHSCOPE_API_BASE", ""),
            os.environ.get("DASHSCOPE_API_KEY", ""),
        )
    return (
        os.environ.get("AIEDU_JUDGE_API_BASE", ""),
        os.environ.get("AIEDU_JUDGE_API_KEY", ""),
    )


from .client import JudgeAPIError, JudgeAuthError, JudgeRateLimitError, OpenAICompatibleClient
from .pairwise import PairwiseJudge, call_judge, parse_judge_result
from .anchor import AnchorJudge
from .prompts import get_judge_prompt, resolve_task_name
from .text import extract_completion_text, strip_think_content
from .stats import bootstrap_anchor_ci, compute_all_stats
from .viz import plot_ranking_with_ci, plot_win_matrix, plot_win_rate, visualize_anchor_results

__all__ = [
    "PairwiseJudge",
    "AnchorJudge",
    "OpenAICompatibleClient",
    "_resolve_judge_credentials",
    "JudgeAPIError",
    "JudgeAuthError",
    "JudgeRateLimitError",
    "call_judge",
    "parse_judge_result",
    "get_judge_prompt",
    "resolve_task_name",
    "strip_think_content",
    "extract_completion_text",
    "bootstrap_anchor_ci",
    "compute_all_stats",
    "visualize_anchor_results",
    "plot_win_matrix",
    "plot_win_rate",
    "plot_ranking_with_ci",
]
