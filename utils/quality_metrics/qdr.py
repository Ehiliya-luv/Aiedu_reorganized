# -*- coding: utf-8 -*-
"""题目错误率（Question Defect Rate, QDR）。

两阶段流程：
1. Solver: 3 个异构强模型独立作答（看 stem+options，不给 answer/explanation）
2. Analyzer: 仅对"多数 solver 与 gold 不一致 / 内部分散"的可疑题二次分析，
   判定是否为题目缺陷 + 错误类型。

QDR = 被 analyzer 判为缺陷的子题数 / 已判定子题数
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.judge.api import JudgeAuthError, NoThinkingError, OpenAICompatibleClient
from .common import JudgeCache, _safe_parse
from .config import APIConfig, output_subdir


SOLVER_MODELS = [
    "gpt-5.5",                  # OpenAI
    "claude-sonnet-4-6",        # Anthropic（已有缓存）
    "gemini-3.1-pro-preview",   # Google
]

# Analyzer 用独立强模型，不在 SOLVER_MODELS 集合内，避免自我背书偏差
ANALYZER_MODEL = "claude-opus-4-8"

DISAGREE_THRESHOLD = 2   # ≥ 2 个 solver 与 gold 不一致 → 进入 analyzer
PROMPT_VERSION = "qdr-v2"  # v2: 增加max_tokens, 修正option_overlap定义, analyzer换独立模型
# Analyzer has a separate contract so semantic rubric changes never invalidate
# expensive solver votes or silently reuse judgments produced by another rubric.
LEGACY_ANALYZER_VERSION = "analyzer-v2"
LEGACY_ANALYZER_PROMPT_VERSION = "qdr-v2"
ANALYZER_VERSION = "analyzer-v5"
ANALYZER_NORMALIZER_VERSION = "qdr-analyzer-normalizer-v5"
LEGACY_ANALYZER_NORMALIZER_VERSION = "qdr-analyzer-normalizer-v2-reconstructed"
ANALYZER_PROFILE_V5 = "v5"
ANALYZER_PROFILE_LEGACY_V2 = "legacy-v2"
ANALYZER_PROFILES = (ANALYZER_PROFILE_V5, ANALYZER_PROFILE_LEGACY_V2)
DEFAULT_ANALYZER_PROFILE = ANALYZER_PROFILE_V5

# QDR 默认限制为 768 个输出 token。GPT-5.5 使用 low；DeepSeek-V4-Pro
# 关闭 thinking 并单独使用 1024-token 上限，其他模型关闭 thinking。
DEFAULT_MAX_TOKENS = 768
DEEPSEEK_REASONING_MAX_TOKENS = 1024
SOLVER_ABSTAIN_ANSWER = "-"
SOLVER_VALID_ANSWERS = {"A", "B", "C", "D", "E", SOLVER_ABSTAIN_ANSWER}
SOLVER_TOKEN_LADDER = [DEFAULT_MAX_TOKENS]
ANALYZER_MAX_TOKENS = 2048
QDR_REQUEST_TIMEOUT_SECONDS = 120
QDR_TRANSIENT_RETRIES = 0
MODEL_REASONING_EFFORT = {
    "gpt-5.5": "low",
}
NO_THINKING_EXTRA_BODY = {
    "thinking": {"type": "disabled"},
    "enable_thinking": False,
}
# 统一使用 Responses reasoning.effort；每次正式运行前验证 Kouri 实际执行值。
RESPONSES_NO_THINKING_MODELS = {
    "gpt-5.5",
    "claude-sonnet-4-6",
    "claude-opus-4-8",
}

# Solver shortening does not change the voting rule or solver cache contract.
# The analyzer rubric/taxonomy is semantic V5 and is isolated independently by
# ANALYZER_VERSION plus a full request-prompt fingerprint.
CONCISE_PROMPT_POLICY_VERSION = "qdr-single-item-quality-v5"
CONCISE_PROMPT_ENABLED = True
SOLVER_REASON_MAX_CHARS = 24
ANALYZER_TEXT_MAX_CHARS = 24
ANALYZER_PROMPT_PROFILE = "full-rubric-short-json-v5"


DEFECT_TYPES = [
    "gold_wrong",      # gold 答案错了
    "multiple_valid",  # 多个选项都成立
    "structural_issue", # 题型结构异常：无选项、缺选项、重复标签、简答题等
    "ambiguous",       # 题干含糊/信息不足
    "impossible",      # 不可解（缺关键信息/选项均错）
    "not_defect",      # 题没问题
]

# This is the schema recoverable from the repository's original analyzer-v2
# implementation. The historical cache later reused the same key while adding
# ``structural_issue`` and changing prompts, so those rows are explicitly
# treated as key-only historical evidence rather than prompt-attested V2 rows.
LEGACY_RECONSTRUCTED_DEFECT_TYPES = [
    "gold_wrong",
    "multiple_valid",
    "ambiguous",
    "impossible",
    "not_defect",
]
LEGACY_HISTORICAL_RESULT_TYPES = list(DEFECT_TYPES)


def normalize_analyzer_profile(profile: str | None) -> str:
    value = profile or DEFAULT_ANALYZER_PROFILE
    if value not in ANALYZER_PROFILES:
        raise ValueError(
            f"unsupported analyzer profile {value!r}; expected one of {ANALYZER_PROFILES}"
        )
    return value


def concise_prompt_policy_metadata(*, applied: bool = True) -> Dict[str, Any]:
    """Return auditable metadata for newly written QDR cache rows."""
    return {
        "version": CONCISE_PROMPT_POLICY_VERSION,
        "mode": "single_item",
        "applied": bool(applied),
        "solver_reason_max_chars": SOLVER_REASON_MAX_CHARS,
        "analyzer_text_max_chars": ANALYZER_TEXT_MAX_CHARS,
        "analyzer_prompt_profile": ANALYZER_PROMPT_PROFILE,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }


def solver_cache_record(
    result: Dict[str, Any], model: str, **extra: Any
) -> Dict[str, Any]:
    """Build a materializer-compatible solver cache row with prompt provenance."""
    return {
        "solver": result,
        "model": model,
        "prompt_policy": concise_prompt_policy_metadata(),
        **extra,
    }


def analyzer_cache_record(
    result: Dict[str, Any],
    *,
    analyzer_profile: str = DEFAULT_ANALYZER_PROFILE,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a materializer-compatible analyzer cache row with provenance."""
    profile = normalize_analyzer_profile(analyzer_profile)
    return {
        "analyze": result,
        "prompt_policy": concise_prompt_policy_metadata(),
        "analyzer_profile": profile,
        "analyzer_contract": analyzer_contract_metadata(profile),
        **extra,
    }


def _clip_text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def reasoning_effort_for_model(model: str) -> str:
    return MODEL_REASONING_EFFORT.get(model, "none")


def max_tokens_for_model(model: str, requested: int) -> int:
    return DEEPSEEK_REASONING_MAX_TOKENS if model == "deepseek-v4-pro" else requested


def make_qdr_client(api: APIConfig) -> OpenAICompatibleClient:
    """Create a QDR client with a bounded per-request wait for paid batches."""
    return OpenAICompatibleClient(
        base_url=api.base_url,
        api_key=api.api_key,
        timeout=QDR_REQUEST_TIMEOUT_SECONDS,
        transient_retries=QDR_TRANSIENT_RETRIES,
    )


def _complete_reasoning_policy_json(
    client: OpenAICompatibleClient,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> str:
    """按模型策略请求 reasoning effort，并拒绝供应商静默改写策略。"""
    effort = reasoning_effort_for_model(model)
    common = {
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "max_tokens": max_tokens_for_model(model, max_tokens),
        "response_format": {"type": "json_object"},
    }
    if model in RESPONSES_NO_THINKING_MODELS:
        return client.responses_complete_text(
            **common,
            temperature=None,
            reasoning={"effort": effort},
            require_no_reasoning=effort == "none",
            required_reasoning_effort=effort if effort != "none" else None,
        )
    return client.chat_complete_text(
        **common,
        temperature=0.0,
        reasoning_effort=effort,
        extra_body=NO_THINKING_EXTRA_BODY if effort == "none" else None,
        allow_reasoning_fallback=False,
        # Kouri may preserve a reasoning_content field even when its official
        # disable switches are applied. Ignore that field and parse only the
        # final content instead of treating field presence as policy failure.
        require_no_reasoning=False,
    )


def validate_reasoning_models(
    client: OpenAICompatibleClient,
    models: List[str],
    max_tokens: int = 64,
    max_attempts: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """用极短请求验证 reasoning 策略，并容忍供应商偶发的畸形响应。"""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    results: Dict[str, Dict[str, Any]] = {}
    for model in dict.fromkeys(models):
        content = ""
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                content = _complete_reasoning_policy_json(
                    client,
                    model=model,
                    system_prompt="Return only strict JSON. Do not explain.",
                    user_prompt='只返回 JSON：{"ok":true}',
                    max_tokens=max_tokens,
                )
                obj = _safe_parse(content)
                if isinstance(obj, dict) and obj.get("ok") is True:
                    break
                last_error = RuntimeError(
                    f"no-thinking preflight 返回无效 JSON: content={content[:200]!r}"
                )
            except (JudgeAuthError, NoThinkingError) as exc:
                raise type(exc)(f"model={model}: {exc}") from exc
            except Exception as exc:
                last_error = exc
            if attempt + 1 < max_attempts:
                time.sleep(attempt + 1)
        else:
            assert last_error is not None
            raise type(last_error)(f"model={model}: {last_error}") from last_error
        results[model] = {
            "ok": True,
            "api": "responses" if model in RESPONSES_NO_THINKING_MODELS else "chat",
            "reasoning_effort": reasoning_effort_for_model(model),
            "max_tokens": max_tokens,
        }
    return results


# ── Solver ──────────────────────────────────────────────────


_SOLVER_SYS = (
    "你是一位临床执业医师考试的应试者。读完题干和选项后，"
    "选择**一个**最符合临床医学共识的最佳答案。只输出严格 JSON。"
)


def _build_solver_prompt(stem: str, options: Dict[str, str]) -> str:
    options_block = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return (
        "请按临床医学知识作答以下选择题：\n\n"
        f"【题干】\n{stem}\n\n【选项】\n{options_block}\n\n"
        "只输出严格 JSON：\n"
        '{\n'
        '  "answer": "A"|"B"|"C"|"D"|"E",\n'
        f'  "reason": "不超过{SOLVER_REASON_MAX_CHARS}个汉字的临床依据"\n'
        "}\n"
        f"严格限制：reason 不得超过 {SOLVER_REASON_MAX_CHARS} 个汉字。"
    )


def _stem_for_solver(item: Dict[str, Any]) -> str:
    parts = []
    if item.get("shared_stem"):
        parts.append(f"【共享题干】{item['shared_stem']}")
    parts.append(f"【题目】{item['stem']}")
    return "\n".join(parts)


def solver_cache_key(item: Dict[str, Any], model: str) -> str:
    payload = json.dumps({
        "v": PROMPT_VERSION,
        "model": model,
        "group": item["group"],
        "case_id": item["case_id"],
        "qtype": item["qtype"],
        "sub_idx": item["sub_idx"],
        "stem": item["stem"][:300],
        "shared_stem": item.get("shared_stem", "")[:300],
        "options": dict(sorted(item["options"].items())),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _normalize_solver_obj(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    ans = obj.get("answer")
    if ans not in SOLVER_VALID_ANSWERS:
        return None
    return {
        "answer": ans,
        "reason": _clip_text(obj.get("reason", ""), SOLVER_REASON_MAX_CHARS),
    }


def _solve_one(
    client: OpenAICompatibleClient,
    model: str,
    item: Dict[str, Any],
    max_attempts: int = 2,
) -> Dict[str, Any]:
    """调用单个 solver 模型作答一道题，保证尽量拿到有效票。

    策略：
    1. 默认关闭 thinking，避免无用 reasoning token
    2. max_tokens 固定为 DEFAULT_MAX_TOKENS；失败仅重试，不自动扩容
    3. 只接受正常 content，不从 reasoning 内容中提取答案
    其他参数（top_p 等）用 API 默认值，只固定 temperature=0 保证可复现。
    """
    user_prompt = _build_solver_prompt(_stem_for_solver(item), item["options"])
    last_err: Optional[str] = None

    def _try(max_tokens: int) -> Optional[Dict[str, Any]]:
        nonlocal last_err
        try:
            content = _complete_reasoning_policy_json(
                client,
                model=model,
                system_prompt=_SOLVER_SYS,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
        except (JudgeAuthError, NoThinkingError):
            raise
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            return None
        obj = _safe_parse(content)
        if obj is None:
            last_err = "parse_failed"
            return None
        norm = _normalize_solver_obj(obj)
        if norm is None:
            last_err = "schema_invalid"
            return None
        norm["judge_model"] = model
        effort = reasoning_effort_for_model(model)
        norm["reasoning_effort"] = effort
        norm["thinking_disabled"] = effort == "none"
        norm["max_tokens_used"] = max_tokens_for_model(model, max_tokens)
        norm["prompt_policy_version"] = CONCISE_PROMPT_POLICY_VERSION
        return norm

    # 固定 768 token；失败时保持相同配置重试。
    for max_tokens in SOLVER_TOKEN_LADDER:
        for _ in range(max_attempts):
            res = _try(max_tokens)
            if res is not None:
                return res

    # A clean stop with no final answer, or a parsed JSON refusal outside A-E,
    # is a solver abstention. Transport/auth/timeouts remain errors and retry.
    if last_err == "schema_invalid" or (
        last_err
        and "empty assistant content" in last_err
        and "finish_reason='stop'" in last_err
    ):
        effort = reasoning_effort_for_model(model)
        return {
            "answer": SOLVER_ABSTAIN_ANSWER,
            "reason": "无法从选项作答",
            "judge_model": model,
            "reasoning_effort": effort,
            "thinking_disabled": effort == "none",
            "max_tokens_used": max_tokens_for_model(model, DEFAULT_MAX_TOKENS),
            "prompt_policy_version": CONCISE_PROMPT_POLICY_VERSION,
            "solver_abstention": True,
        }

    return {"error": last_err or "unknown_failure"}


# ── Analyzer ────────────────────────────────────────────────


_ANALYZER_SYS = (
    "你是资深临床医学考题质控员。严格按两步判断：\n"
    "第一步，完全忽略 gold 和 solver，仅凭临床知识独立选择一个最佳答案；\n"
    "第二步，再比较独立答案、gold、题干和选项，判断题目是否有缺陷。\n"
    "多个模型意见一致不等于正确；solver 偏离 gold 本身也不是缺陷证据。只有临床依据明确时才否定 gold，"
    "not_defect 应是最常见结论。\n"
    "必须按以下优先级只选一类：\n"
    "1.structural_issue：仅限呈现结构错误，如选项缺失/重复标签/无选项/简答题伪装选择题，"
    "或另一道题的题干、选项被拼入本题；医学内容错误或信息不足绝不归此类。\n"
    "2.impossible：缺少解题所必需的核心信息，或所有给定选项均不能回答题干。"
    "若题目使用‘该患者/本例’等指代却没有对应患者事实，不得从选项、gold 或 solver 反推背景，必须归此类。\n"
    "3.multiple_valid：在题干给定的同一组事实下，至少两个选项同等成立。\n"
    "4.ambiguous：题干缺少范围、稳定性、时点、治疗目标等区分条件，"
    "不同合理补充或读法分别导向不同答案；它与同一事实下多项同时成立不同。\n"
    "5.gold_wrong：题干和选项可唯一作答、gold 在临床上不可辩护，且恰有一个唯一替代答案；"
    "若 gold 仍可辩护，或只是另一选项看起来更优先，绝不得归 gold_wrong。\n"
    "not_defect：gold 在临床上可辩护且题目能唯一作答，即使 solver 集体答错也选此类。\n"
    "症状轻重、处置优先级或措辞粒度差异不单独构成缺陷。"
    "structural_issue/impossible 无法独立作答时独立答案可为 null。只输出严格 JSON，不输出额外文字。"
)

# Exact system prompt recoverable from the original analyzer-v2 source. It is
# retained as a frozen compatibility profile; it is never selected implicitly.
_LEGACY_ANALYZER_SYS = (
    "你是资深临床医学专家，负责医学考题质控。给定一道选择题、官方 gold 答案，以及多个强 LLM 的作答。\n\n"
    "**关键工作流程（必须严格遵守）：**\n"
    "第一步：先抛开 gold 和 solver 的答案，仅凭你自己的临床医学知识独立解这道题，"
    "确定你认为的正确答案（independent_answer）及临床依据。\n"
    "第二步：再把你的独立答案与 gold 比对，判断**题目本身是否有缺陷**。\n\n"
    "**重要警示：多个强 LLM 给出相同答案，不代表该答案正确**——它们常共享同一知识盲区或"
    "被题干表述带偏而集体犯错。solver 与 gold 不一致，首先要怀疑的是 solver 集体错了，"
    "而非默认题目有问题。只有你独立分析后确信 gold 在临床上站不住，才判为缺陷。\n\n"
    "**判定类型：**\n"
    "1. gold_wrong：你独立得出的答案与 gold 不同，且你有充分临床依据认为 gold 错、应是另一选项。\n"
    "2. multiple_valid：存在两个及以上同等成立、无法分出优劣的正确选项。\n"
    "3. ambiguous：题干关键信息缺失或表述含糊，**任何医生都无法**确定唯一答案"
    "（注意：你自己能确定答案就不算 ambiguous）。\n"
    "4. impossible：题目无法解答（缺核心信息或所有选项都错）。\n"
    "5. not_defect：题目没问题。你的独立答案与 gold 一致，或虽然 solver 们偏离 gold，"
    "但你认同 gold 是对的、是 solver 集体推理偏差。**这应当是最常见的判定。**\n\n"
    "注意：不同程度的症状（如『头痛』vs『剧烈头痛』）是不同临床表现，不构成缺陷。\n"
    "只输出严格 JSON。"
)


def _build_analyzer_prompt(
    stem: str,
    options: Dict[str, str],
    gold: str,
    solver_results: List[tuple[str, Dict[str, Any]]],
) -> str:
    options_block = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    solver_block = "\n".join(
        f"- {model}: {r.get('answer','?')}；{_clip_text(r.get('reason', ''), SOLVER_REASON_MAX_CHARS)}"
        for model, r in solver_results
    )
    type_list = ", ".join(DEFECT_TYPES)
    return (
        f"【题干】{stem}\n【选项】\n{options_block}\n"
        f"【gold】{gold}\n【solver参考】\n{solver_block}\n"
        "按先独立作答、后比较 gold 的顺序，仅返回 JSON；短键含义："
        "a=独立答案，t=缺陷类型，c=仅当 gold_wrong 时填写正确答案，否则 null，r=核心判据。"
        f'{{"a":"A-E之一或null","t":"{type_list}之一",'
        f'"c":"A-E之一或null","r":"<={ANALYZER_TEXT_MAX_CHARS}汉字"}}'
    )


def _build_legacy_analyzer_prompt(
    stem: str,
    options: Dict[str, str],
    gold: str,
    solver_results: List[tuple[str, Dict[str, Any]]],
) -> str:
    """Reconstruct the full analyzer-v2 user prompt byte for byte."""
    options_block = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    solver_block = "\n".join(
        f"- {model}: 选 {r.get('answer','?')}  理由：{r.get('reason','')}"
        for model, r in solver_results
    )
    type_list = ", ".join(LEGACY_RECONSTRUCTED_DEFECT_TYPES)
    return (
        "请审阅这道考题：\n\n"
        f"【题干】\n{stem}\n\n【选项】\n{options_block}\n\n"
        "——以下为参考信息，请先独立解题后再参考——\n"
        f"【官方 gold 答案】{gold}. {options.get(gold, '')}\n\n"
        f"【强 LLM 各自作答】\n{solver_block}\n\n"
        "请按工作流程（先独立解题，再比对 gold）输出严格 JSON：\n"
        '{\n'
        '  "independent_answer": "A"|"B"|"C"|"D"|"E",  // 你抛开 gold/solver 独立解出的答案\n'
        '  "independent_reason": "你独立解题的临床依据（一两句）",\n'
        f'  "defect_type": one of [{type_list}],\n'
        '  "is_defect": true|false,   // type ∈ {gold_wrong, multiple_valid, ambiguous, impossible} → true；not_defect → false\n'
        '  "correct_answer_if_gold_wrong": "A"|"B"|"C"|"D"|"E"|null,\n'
        '  "evidence": "引用题干或选项中的具体文字，说明判定依据",\n'
        '  "reason": "一两句话总结：为何判为该类型"\n'
        "}"
    )


def analyzer_system_prompt(profile: str = DEFAULT_ANALYZER_PROFILE) -> str:
    profile = normalize_analyzer_profile(profile)
    return _ANALYZER_SYS if profile == ANALYZER_PROFILE_V5 else _LEGACY_ANALYZER_SYS


def build_analyzer_prompt_for_profile(
    item: Dict[str, Any],
    solver_results: List[tuple[str, Dict[str, Any]]],
    profile: str = DEFAULT_ANALYZER_PROFILE,
) -> str:
    profile = normalize_analyzer_profile(profile)
    builder = (
        _build_analyzer_prompt
        if profile == ANALYZER_PROFILE_V5
        else _build_legacy_analyzer_prompt
    )
    return builder(
        _stem_for_solver(item), item["options"], item["answer"], solver_results
    )


def analyzer_contract_metadata(
    profile: str = DEFAULT_ANALYZER_PROFILE,
) -> Dict[str, Any]:
    profile = normalize_analyzer_profile(profile)
    if profile == ANALYZER_PROFILE_V5:
        version = ANALYZER_VERSION
        normalizer_version = ANALYZER_NORMALIZER_VERSION
        prompt_profile = ANALYZER_PROMPT_PROFILE
        defect_types = list(DEFECT_TYPES)
        attestation = "full_prompt_contract"
    else:
        version = LEGACY_ANALYZER_VERSION
        normalizer_version = LEGACY_ANALYZER_NORMALIZER_VERSION
        prompt_profile = "historical-v2-key-reconstructed-prompt"
        defect_types = list(LEGACY_RECONSTRUCTED_DEFECT_TYPES)
        attestation = "historical_v2_key_unattested_prompt"
    payload = {
        "profile": profile,
        "analyzer_version": version,
        "prompt_version": (
            PROMPT_VERSION
            if profile == ANALYZER_PROFILE_V5
            else LEGACY_ANALYZER_PROMPT_VERSION
        ),
        "analyzer_model": ANALYZER_MODEL,
        "normalizer_version": normalizer_version,
        "system_prompt": analyzer_system_prompt(profile),
        "defect_types": defect_types,
        "accepted_historical_result_types": (
            list(LEGACY_HISTORICAL_RESULT_TYPES)
            if profile == ANALYZER_PROFILE_LEGACY_V2
            else defect_types
        ),
        "prompt_profile": prompt_profile,
        "historical_attestation": attestation,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return {
        "profile": profile,
        "version": version,
        "model": ANALYZER_MODEL,
        "normalizer_version": normalizer_version,
        "prompt_profile": prompt_profile,
        "historical_attestation": attestation,
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def analyzer_cache_record_has_current_contract(record: Optional[Dict[str, Any]]) -> bool:
    return analyzer_cache_record_has_contract(record, ANALYZER_PROFILE_V5)


def analyzer_cache_record_has_contract(
    record: Optional[Dict[str, Any]],
    profile: str = DEFAULT_ANALYZER_PROFILE,
) -> bool:
    profile = normalize_analyzer_profile(profile)
    if not isinstance(record, dict):
        return False
    result = record.get("analyze") or {}
    if not result or "error" in result:
        return False
    allowed_types = (
        DEFECT_TYPES
        if profile == ANALYZER_PROFILE_V5
        else LEGACY_HISTORICAL_RESULT_TYPES
    )
    if result.get("defect_type") not in allowed_types:
        return False
    if not isinstance(result.get("is_defect"), bool):
        return False
    contract = record.get("analyzer_contract")
    if profile == ANALYZER_PROFILE_LEGACY_V2 and contract is None:
        # Historical rows predate contract metadata. This acceptance path is
        # reachable only when the caller explicitly selects ``legacy-v2``.
        return True
    return (
        record.get("analyzer_profile", profile) == profile
        and contract == analyzer_contract_metadata(profile)
    )


def analyzer_cache_attestation(
    record: Optional[Dict[str, Any]],
    profile: str = DEFAULT_ANALYZER_PROFILE,
) -> str | None:
    if not analyzer_cache_record_has_contract(record, profile):
        return None
    profile = normalize_analyzer_profile(profile)
    if profile == ANALYZER_PROFILE_LEGACY_V2 and not record.get("analyzer_contract"):
        return "historical_key_only"
    return "contract_metadata"


def analyzer_request_signature(
    item: Dict[str, Any],
    solver_results: List[tuple[str, Dict[str, Any]]],
    profile: str = DEFAULT_ANALYZER_PROFILE,
) -> str:
    profile = normalize_analyzer_profile(profile)
    user_prompt = build_analyzer_prompt_for_profile(item, solver_results, profile)
    payload = json.dumps(
        {
            "contract": analyzer_contract_metadata(profile),
            "system_prompt": analyzer_system_prompt(profile),
            "user_prompt": user_prompt,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def legacy_analyzer_cache_key(
    item: Dict[str, Any], solver_results: List[tuple[str, Dict[str, Any]]]
) -> str:
    """Return the historical V2 key for audit/calibration only."""
    payload = json.dumps({
        "v": LEGACY_ANALYZER_PROMPT_VERSION,
        "av": LEGACY_ANALYZER_VERSION,
        "group": item["group"],
        "case_id": item["case_id"],
        "qtype": item["qtype"],
        "sub_idx": item["sub_idx"],
        "stem": item["stem"][:300],
        "shared_stem": item.get("shared_stem", "")[:300],
        "options": dict(sorted(item["options"].items())),
        "gold": item["answer"],
        "solvers": [(m, r.get("answer")) for m, r in sorted(solver_results)],
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def analyzer_cache_key(
    item: Dict[str, Any],
    solver_results: List[tuple[str, Dict[str, Any]]],
    profile: str = DEFAULT_ANALYZER_PROFILE,
) -> str:
    profile = normalize_analyzer_profile(profile)
    if profile == ANALYZER_PROFILE_LEGACY_V2:
        return legacy_analyzer_cache_key(item, solver_results)
    payload = json.dumps({
        "v": PROMPT_VERSION,
        "av": ANALYZER_VERSION,
        "contract_sha256": analyzer_contract_metadata(profile)["sha256"],
        "request_sha256": analyzer_request_signature(item, solver_results, profile),
        "group": item["group"],
        "case_id": item["case_id"],
        "qtype": item["qtype"],
        "sub_idx": item["sub_idx"],
        "stem": item["stem"][:300],
        "shared_stem": item.get("shared_stem", "")[:300],
        "options": dict(sorted(item["options"].items())),
        "gold": item["answer"],
        "solvers": [(m, r.get("answer")) for m, r in sorted(solver_results)],
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _normalize_analyzer_obj(
    obj: Dict[str, Any],
    analyzer_profile: str = DEFAULT_ANALYZER_PROFILE,
) -> Optional[Dict[str, Any]]:
    profile = normalize_analyzer_profile(analyzer_profile)
    if not isinstance(obj, dict):
        return None
    dt = obj["defect_type"] if "defect_type" in obj else obj.get("t")
    allowed_types = (
        DEFECT_TYPES
        if profile == ANALYZER_PROFILE_V5
        else LEGACY_RECONSTRUCTED_DEFECT_TYPES
    )
    if dt not in allowed_types:
        return None
    is_def = dt != "not_defect"
    correct = (
        obj["correct_answer_if_gold_wrong"]
        if "correct_answer_if_gold_wrong" in obj
        else obj.get("c")
    )
    if correct not in {"A", "B", "C", "D", "E", None}:
        correct = None
    indep = obj["independent_answer"] if "independent_answer" in obj else obj.get("a")
    if indep not in {"A", "B", "C", "D", "E"}:
        indep = None
    compact_reason = obj.get("r", "")
    independent_reason = (
        obj["independent_reason"]
        if "independent_reason" in obj
        else obj.get("ir", compact_reason)
    )
    evidence = obj["evidence"] if "evidence" in obj else obj.get("e", compact_reason)
    reason = obj["reason"] if "reason" in obj else obj.get("r", "")
    return {
        "defect_type": dt,
        "is_defect": is_def,
        "correct_answer_if_gold_wrong": correct,
        "independent_answer": indep,
        "independent_reason": _clip_text(
            independent_reason,
            ANALYZER_TEXT_MAX_CHARS if profile == ANALYZER_PROFILE_V5 else 300,
        ),
        "evidence": _clip_text(
            evidence, ANALYZER_TEXT_MAX_CHARS if profile == ANALYZER_PROFILE_V5 else 300
        ),
        "reason": _clip_text(
            reason, ANALYZER_TEXT_MAX_CHARS if profile == ANALYZER_PROFILE_V5 else 300
        ),
    }


def _analyze_one(
    client: OpenAICompatibleClient,
    item: Dict[str, Any],
    solver_results: List[tuple[str, Dict[str, Any]]],
    max_attempts: int = 3,
    analyzer_profile: str = DEFAULT_ANALYZER_PROFILE,
) -> Dict[str, Any]:
    """调用 analyzer 模型判定题目是否有缺陷。

    API 参数设计：
    - temperature=0.0: 确保判定结果可复现
    - max_tokens=ANALYZER_MAX_TOKENS: 统一限制为 768
    - 按模型使用经预检验证的 no-thinking 接口与参数
    - response_format=json_object: 结构化输出
    - 其他参数使用 API 默认值
    """
    profile = normalize_analyzer_profile(analyzer_profile)
    user_prompt = build_analyzer_prompt_for_profile(item, solver_results, profile)
    system_prompt = analyzer_system_prompt(profile)
    last_err: Optional[str] = None
    for _ in range(max_attempts):
        try:
            content = _complete_reasoning_policy_json(
                client,
                model=ANALYZER_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=ANALYZER_MAX_TOKENS,
            )
        except (JudgeAuthError, NoThinkingError):
            raise
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            continue
        obj = _safe_parse(content)
        if obj is None:
            last_err = "parse_failed"
            continue
        norm = _normalize_analyzer_obj(obj, profile)
        if norm is None:
            last_err = "schema_invalid"
            continue
        norm["judge_model"] = ANALYZER_MODEL
        effort = reasoning_effort_for_model(ANALYZER_MODEL)
        norm["reasoning_effort"] = effort
        norm["thinking_disabled"] = effort == "none"
        norm["max_tokens_used"] = ANALYZER_MAX_TOKENS
        norm["prompt_policy_version"] = CONCISE_PROMPT_POLICY_VERSION
        norm["analyzer_profile"] = profile
        norm["analyzer_version"] = analyzer_contract_metadata(profile)["version"]
        norm["analyzer_contract_sha256"] = analyzer_contract_metadata(profile)["sha256"]
        norm["analyzer_request_sha256"] = analyzer_request_signature(
            item, solver_results, profile
        )
        return norm
    return {"error": last_err or "unknown_failure"}


# ── 顶层入口 ────────────────────────────────────────────────


def _is_multi_select(item: Dict[str, Any]) -> bool:
    """是否为多选题。

    多选题仍纳入 QDR 分母，但不跑单选 solver/analyzer，直接记为
    multiple_valid 缺陷。这保留题目质量缺陷信息，同时避免无意义 API 消耗。
    """
    answer = str(item.get("answer") or "")
    return (
        "," in answer
        or "，" in answer
        or "、" in answer
        or "多选" in (item.get("stem") or "")
        or "多选" in (item.get("shared_stem") or "")
    )


def _multi_select_defect_record(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **item,
        "solvers": {},
        "n_disagree": 0,
        "solver_consensus": None,
        "analyzer": {
            "defect_type": "multiple_valid",
            "is_defect": True,
            "correct_answer_if_gold_wrong": None,
            "independent_answer": None,
            "evidence": "题目或答案为多选形式，不符合单选最佳答案题口径。",
            "reason": "多选题作为题目格式缺陷直接计入 QDR，不调用 solver/analyzer。",
            "auto_flagged": "multi_select",
        },
    }


def _has_structural_issue(item: Dict[str, Any]) -> bool:
    return bool(item.get("structural_issues") or item.get("option_issues"))


def _structural_issue_defect_record(item: Dict[str, Any]) -> Dict[str, Any]:
    issues = item.get("structural_issues") or item.get("option_issues") or []
    return {
        **item,
        "solvers": {},
        "n_disagree": 0,
        "solver_consensus": None,
        "analyzer": {
            "defect_type": "structural_issue",
            "is_defect": True,
            "correct_answer_if_gold_wrong": None,
            "independent_answer": None,
            "evidence": json.dumps(issues, ensure_ascii=False)[:300],
            "reason": "题目存在无选项、缺选项、重复标签、简答题等结构问题，直接计入 QDR 缺陷。",
            "auto_flagged": "structural_issue",
        },
    }


def _disagreement_count(solver_results: List[tuple[str, Dict[str, Any]]], gold: str) -> int:
    """有几个 solver 与 gold 不一致（仅成功作答的计入分母）。"""
    cnt = 0
    for _, r in solver_results:
        if "error" in r:
            continue
        if r.get("answer") and r.get("answer") != gold:
            cnt += 1
    return cnt


def _solver_consensus(solver_results: List[tuple[str, Dict[str, Any]]]) -> Optional[str]:
    """solver 内部最多票数选项；分散返回 None。"""
    counts = Counter(
        r["answer"] for _, r in solver_results
        if "error" not in r and r.get("answer")
    )
    if not counts:
        return None
    most_common = counts.most_common()
    if len(most_common) >= 2 and most_common[0][1] == most_common[1][1]:
        return None
    return most_common[0][0]


def evaluate_qdr(
    items: List[Dict[str, Any]],
    api: APIConfig,
    workers: int = 16,
    progress_every: int = 100,
    solver_models: Optional[List[str]] = None,
    disagree_threshold: int = DISAGREE_THRESHOLD,
    require_complete_votes: bool = True,
    analyzer_profile: str = DEFAULT_ANALYZER_PROFILE,
) -> List[Dict[str, Any]]:
    """主流程：solve → 标记可疑 → analyze → 落盘每子题一行。

    require_complete_votes：是否强制每题拿满 len(solver_models) 张有效票。
    True（默认）时，solver 阶段后若仍有题缺票则抛错中止——
    票数不齐会让 n_disagree/触发逻辑失真，必须先补齐。
    """
    solver_models = solver_models or SOLVER_MODELS
    analyzer_profile = normalize_analyzer_profile(analyzer_profile)
    out_dir = output_subdir("qdr")

    structural_records = [_structural_issue_defect_record(it) for it in items if _has_structural_issue(it)]
    items = [it for it in items if not _has_structural_issue(it)]
    multi_records = [_multi_select_defect_record(it) for it in items if _is_multi_select(it)]
    items = [it for it in items if not _is_multi_select(it)]
    n_multi = len(multi_records)
    n_structural = len(structural_records)
    if n_structural:
        print(f"[QDR] 结构问题 {n_structural} 道直接记为 structural_issue 缺陷")
    if n_multi:
        print(f"[QDR] 多选题 {n_multi} 道直接记为 multiple_valid 缺陷；单选待评估 {len(items)} 道")

    solver_cache = JudgeCache.load(out_dir / "solver_cache.jsonl")
    analyze_cache = JudgeCache.load(out_dir / "analyze_cache.jsonl")
    print(f"[QDR] 加载缓存：solver={len(solver_cache._data)} analyze={len(analyze_cache._data)}")

    client = make_qdr_client(api)

    # ── Step 1: 跑 N 个 solver × 所有 items ──
    solver_todo: List[tuple[int, str, str, Dict[str, Any]]] = []  # (item_idx, model, key, item)
    solver_results: Dict[int, Dict[str, Dict[str, Any]]] = {i: {} for i in range(len(items))}

    for i, item in enumerate(items):
        for model in solver_models:
            key = solver_cache_key(item, model)
            cached = solver_cache.get(key)
            # error 缓存不算数：重新请求，交给阶梯+关 reasoning 逻辑补救
            if cached is not None and "error" not in (cached.get("solver") or {}):
                solver_results[i][model] = cached.get("solver", {})
            else:
                solver_todo.append((i, model, key, item))

    print(f"[QDR] solver 待请求：{len(solver_todo)} / {len(items) * len(solver_models)}")

    if solver_todo:
        completed = 0
        lock = threading.Lock()

        def _solve_task(args):
            nonlocal completed
            i, model, key, item = args
            res = _solve_one(client, model, item)
            solver_cache.append(key, solver_cache_record(res, model))
            with lock:
                completed += 1
                done = completed
            if done % progress_every == 0 or done == len(solver_todo):
                print(f"[QDR] solver 进度 {done}/{len(solver_todo)}")
            return i, model, res

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for i, model, res in ex.map(_solve_task, solver_todo):
                solver_results[i][model] = res

    # ── Step 1.5: 强制校验票数齐全 ──
    # 票数不齐会让 n_disagree / 触发逻辑失真，必须在进入 analyzer 前确保每题满票。
    incomplete: List[tuple[Dict[str, Any], List[str]]] = []
    for i, item in enumerate(items):
        sr = solver_results[i]
        bad = [m for m in solver_models
               if m not in sr or "error" in (sr.get(m) or {})]
        if bad:
            incomplete.append((item, bad))
    if incomplete:
        sample = "; ".join(
            f"{it['group']}/{it['case_id']}/{it['qtype']}#{it['sub_idx']}→缺{bad}"
            for it, bad in incomplete[:5]
        )
        msg = (
            f"[QDR] 票数不齐：{len(incomplete)}/{len(items)} 题未拿满 "
            f"{len(solver_models)} 张有效票（前5例：{sample}）。"
        )
        if require_complete_votes:
            raise RuntimeError(
                msg + " 已用 token 阶梯+关 reasoning 仍失败，请检查模型可用性或单独补票后重跑。"
            )
        print("[警告] " + msg + " require_complete_votes=False，继续但结果可能失真。")
    else:
        print(f"[QDR] 票数校验通过：{len(items)} 题均拿满 {len(solver_models)} 票")

    # ── Step 2: 标记需要 analyzer 的可疑题 ──
    analyze_todo: List[tuple[int, str, Dict[str, Any], List[tuple[str, Dict[str, Any]]]]] = []
    analyze_results: Dict[int, Dict[str, Any]] = {}

    for i, item in enumerate(items):
        sr = solver_results[i]
        sr_list = sorted(sr.items())  # 排序保证 cache 稳定
        gold = item["answer"]
        n_disagree = _disagreement_count(sr_list, gold)
        # 触发条件：≥ threshold 个 solver 偏离 gold。
        # （此前的 consensus is None 条件冗余：3 票齐全时无共识必然 ≥2 偏离，已被覆盖）
        trigger = n_disagree >= disagree_threshold
        if not trigger:
            analyze_results[i] = {"skipped": True, "n_disagree": n_disagree}
            continue

        key = analyzer_cache_key(item, sr_list, analyzer_profile)
        cached = analyze_cache.get(key)
        if analyzer_cache_record_has_contract(cached, analyzer_profile):
            analyze_results[i] = cached.get("analyze", {})
        else:
            analyze_todo.append((i, key, item, sr_list))

    print(f"[QDR] analyzer 待请求：{len(analyze_todo)} / 总可疑数 {len(analyze_todo) + sum(1 for v in analyze_results.values() if 'skipped' not in v)}")

    if analyze_todo:
        completed = 0
        lock = threading.Lock()

        def _analyze_task(args):
            nonlocal completed
            i, key, item, sr_list = args
            res = _analyze_one(
                client, item, sr_list, analyzer_profile=analyzer_profile
            )
            analyze_cache.append(
                key,
                analyzer_cache_record(res, analyzer_profile=analyzer_profile),
            )
            with lock:
                completed += 1
                done = completed
            if done % 25 == 0 or done == len(analyze_todo):
                print(f"[QDR] analyzer 进度 {done}/{len(analyze_todo)}")
            return i, res

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for i, res in ex.map(_analyze_task, analyze_todo):
                analyze_results[i] = res

    # ── Step 3: 组合输出 ──
    out: List[Dict[str, Any]] = list(structural_records) + list(multi_records)
    for i, item in enumerate(items):
        sr = solver_results[i]
        sr_list = sorted(sr.items())
        n_disagree = _disagreement_count(sr_list, item["answer"])
        consensus = _solver_consensus(sr_list)
        analyzer = analyze_results.get(i, {"skipped": True, "n_disagree": n_disagree})
        out.append({
            **item,
            "solvers": {m: r for m, r in sr_list},
            "n_disagree": n_disagree,
            "solver_consensus": consensus,
            "analyzer": analyzer,
        })
    return out


def is_defect(record: Dict[str, Any]) -> bool:
    a = record.get("analyzer") or {}
    if a.get("skipped"):
        return False
    if "error" in a:
        return False
    return bool(a.get("is_defect"))


def summarize_qdr(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_group: Dict[str, Dict[str, Any]] = {}
    for r in records:
        g = r["group"]
        d = by_group.setdefault(g, {
            "total": 0,
            "judged": 0,             # 至少一个 solver 成功，或由确定性规则自动判定
            "suspect": 0,            # 触发了 analyzer
            "defect": 0,
            "type_counts": {t: 0 for t in DEFECT_TYPES},
            "by_qtype": {},
        })
        d["total"] += 1
        a = r.get("analyzer") or {}
        auto_flagged = a.get("auto_flagged") in {"multi_select", "structural_issue"}
        # 至少一个 solver 成功；自动标记题是确定性格式缺陷，不需要 solver。
        any_ok = auto_flagged or any("error" not in s for s in r.get("solvers", {}).values())
        if not any_ok:
            continue
        d["judged"] += 1
        triggered = not a.get("skipped")
        if triggered:
            d["suspect"] += 1
            # 按 is_defect 归一化类型：is_defect=False 一律记 not_defect，
            # 避免 analyzer 残留旧 defect_type 标签导致堆叠图与 defect 数对不上
            if is_defect(r):
                t = a.get("defect_type")
                if t not in DEFECT_TYPES or t == "not_defect":
                    t = "gold_wrong"  # 兜底：标缺陷却无合法类型
            else:
                t = "not_defect"
            d["type_counts"][t] += 1
        defect_flag = is_defect(r)
        if defect_flag:
            d["defect"] += 1
        qd = d["by_qtype"].setdefault(r["qtype"], {"total": 0, "defect": 0})
        qd["total"] += 1
        qd["defect"] += int(defect_flag)

    for d in by_group.values():
        d["qdr"] = d["defect"] / d["judged"] if d["judged"] else 0.0
        # 可信率 = 1 - QDR：对外得分，越高代表数据集题目越可信
        d["credibility_rate"] = (d["judged"] - d["defect"]) / d["judged"] if d["judged"] else 0.0
        d["suspect_rate"] = d["suspect"] / d["judged"] if d["judged"] else 0.0
        for qd in d["by_qtype"].values():
            qd["rate"] = qd["defect"] / qd["total"] if qd["total"] else 0.0
    return by_group
