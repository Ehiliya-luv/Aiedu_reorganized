"""Pairwise LLM Judge: single-case comparison (A/B winner + reason).

Extracted and simplified from utils/com_exp/judge_api.py.
No checkpoint/resume, no batch evaluation — just one comparison at a time.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time

from .client import JudgeAPIError, JudgeAuthError, JudgeRateLimitError, OpenAICompatibleClient
from .prompts import get_judge_prompt
from .text import strip_think_content

logger = logging.getLogger(__name__)

FATAL_JUDGE_ERRORS = (JudgeAuthError, JudgeAPIError)

PAIRWISE_JUDGE_RESPONSE_FORMAT = {"type": "json_object"}


def _format_raw_response(content: str) -> str:
    if content == "":
        return "<EMPTY STRING>"
    return repr(content)


def parse_judge_result(content: str) -> dict | None:
    """Parse {"winner": "A"/"B", "reason": "..."} from LLM response.

    Returns:
        {"winner": "A"/"B", "reason": "..."} or None (unparseable)
    """
    if not content:
        return None

    raw = strip_think_content(content).strip()
    # Strip markdown code fences (```json ... ``` or bare ``` ... ```)
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)
    raw = raw.strip()

    # 1) Strict JSON parse (preferred path)
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            winner = payload.get("winner")
            if winner in ("A", "B"):
                reason = payload.get("reason", "")
                if not isinstance(reason, str):
                    reason = str(reason)
                return {"winner": winner, "reason": reason.strip()}
    except json.JSONDecodeError:
        pass

    # 2) Regex fallback — survives unescaped inner quotes in `reason`.
    #    The model occasionally writes `"reason": "...如A4题考查"核心..."的逻辑..."`
    #    which breaks json.loads but the winner letter is still recoverable.
    winner_match = re.search(r'"winner"\s*:\s*"([AB])"', raw)
    if not winner_match:
        logger.warning("[Judge] Cannot parse result, raw content: %s", _format_raw_response(raw))
        return None
    winner = winner_match.group(1)

    reason_match = re.search(r'"reason"\s*:\s*"(.*)"\s*[,}]\s*$', raw, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else ""

    logger.info("[Judge] Salvaged winner=%s via regex (JSON malformed)", winner)
    return {"winner": winner, "reason": reason}


def call_judge(
    client: OpenAICompatibleClient,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    top_p: float | None = None,
    max_tokens: int = 2048,
    max_retries: int = 3,  # only for JSON parse failures (transport retries live in client.py)
    response_format: dict | None = PAIRWISE_JUDGE_RESPONSE_FORMAT,
    extra_body: dict | None = None,
) -> dict | None:
    """Call Judge API and parse result.

    Returns:
        {"winner": "A"/"B", "reason": "..."} or None
    """
    for attempt in range(max_retries):
        try:
            content = client.chat_complete_text(
                model=model,
                system_prompt="",
                user_prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                response_format=response_format,
                extra_body=extra_body,
            )

            content = strip_think_content(content)
            result = parse_judge_result(content)
            if result is not None:
                return result

            logger.warning("[Judge] Attempt %d: unexpected result, got: %s", attempt + 1, _format_raw_response(content))
        except FATAL_JUDGE_ERRORS:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except JudgeRateLimitError as e:
            logger.warning("[Judge] Rate limit exhausted: %s: %s", type(e).__name__, e)
            return None
        except Exception as e:
            logger.warning("[Judge] Attempt %d failed: %s: %s", attempt + 1, type(e).__name__, e)

        if attempt < max_retries - 1:
            time.sleep(2**attempt)

    logger.warning("[Judge] All retries failed, returning None")
    return None


class PairwiseJudge:
    """High-level pairwise judge: compare two outputs for one task.

    Usage:
        judge = PairwiseJudge(...)
        result = judge.compare(task="...", raw_case="...", output_a="...", output_b="...")
        # result = {"winner": "A", "reason": "A 的结构更完整..."}
    """

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
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
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.prompt_variant = prompt_variant
        self.client = OpenAICompatibleClient(
            base_url=api_base,
            api_key=api_key,
            timeout=timeout,
            local_mode=local_mode,
            rate_limit_max_retries=rate_limit_max_retries,
            transient_retries=transient_retries,
        )

    def compare(
        self,
        task: str,
        raw_case: str,
        output_a: str,
        output_b: str,
        *,
        debias: bool = True,
        seed: int | None = None,
        extra_body: dict | None = None,
    ) -> dict | None:
        """Compare two outputs and return winner + reason.

        Returns:
            {"winner": "A"/"B", "reason": "..."} or None
        """
        rng = random.Random(seed) if seed is not None else random
        if debias and rng.random() > 0.5:
            prompt = get_judge_prompt(task, raw_case, output_b, output_a, prompt_variant=self.prompt_variant)
            result = call_judge(
                self.client,
                self.model,
                prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                extra_body=extra_body,
            )
            if result is None:
                return None
            # Flip winner back
            winner = result["winner"]
            reason = result["reason"]
            # Flip "模型 A"/"模型 B" labels to match flipped winner
            reason = reason.replace("模型 A", "__TEMP__").replace("模型 B", "模型 A").replace("__TEMP__", "模型 B")
            return {
                "winner": "B" if winner == "A" else "A",
                "reason": reason,
            }
        else:
            prompt = get_judge_prompt(task, raw_case, output_a, output_b, prompt_variant=self.prompt_variant)
            return call_judge(
                self.client,
                self.model,
                prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                extra_body=extra_body,
            )

    def compare_with_repeats(
        self,
        task: str,
        raw_case: str,
        output_a: str,
        output_b: str,
        *,
        repeats: int = 3,
        debias: bool = True,
        extra_body: dict | None = None,
    ) -> tuple[int, int, list[dict]]:
        """Compare with multiple votes and return win counts + all reasons.

        Returns:
            (wins_a, wins_b, reasons_list)
            reasons_list = [{"winner": "A"/"B", "reason": "..."}, ...]
        """
        wins_a = 0
        wins_b = 0
        reasons = []

        for _i in range(repeats):
            result = self.compare(
                task=task,
                raw_case=raw_case,
                output_a=output_a,
                output_b=output_b,
                debias=debias,
                extra_body=extra_body,
            )
            if result is None:
                continue

            reasons.append(result)
            if result["winner"] == "A":
                wins_a += 1
            elif result["winner"] == "B":
                wins_b += 1

        if not reasons:
            logger.warning(
                "compare_with_repeats: all %d repeat(s) returned None — " "comparison failed",
                repeats,
            )

        return wins_a, wins_b, reasons

    def compare_batch(
        self,
        task: str,
        cases: list[dict],
        *,
        debias: bool = True,
        extra_body: dict | None = None,
    ) -> list[dict | None]:
        """Compare multiple cases sequentially.

        Args:
            task:   Task name
            cases:  List of {"raw_case": str, "output_a": str, "output_b": str}

        Returns:
            List of {"winner": "A"/"B", "reason": "..."}/None results
        """
        results = []
        for i, case in enumerate(cases):
            winner = self.compare(
                task=task,
                raw_case=case["raw_case"],
                output_a=case["output_a"],
                output_b=case["output_b"],
                debias=debias,
                extra_body=extra_body,
            )
            results.append(winner)
            if (i + 1) % 10 == 0:
                logger.info("[Judge] Progress: %d/%d", i + 1, len(cases))
        return results


__all__ = ["PairwiseJudge", "call_judge", "parse_judge_result"]
