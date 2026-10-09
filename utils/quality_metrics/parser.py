# -*- coding: utf-8 -*-
"""md 题目解析器：把每份 case 的 md 拆成结构化子题。

输出 schema（每行一个 dict）:
{
  "group": "base|lora|framework",
  "case_id": "case001",
  "qtype": "A1|A2|A3|A4|B1",
  "group_idx": int,    # 题型组序号（每 case 0..4）
  "sub_idx": int,      # 题型组内子题序号（A1/A2 恒为 0；A3/A4/B1 0..N-1）
  "shared_stem": str,  # A3/A4/B1 共享题干；A1/A2 为空
  "stem": str,         # 子题独立题干（A1/A2 = 题干；A3/A4 = 子问题；B1 = 配伍问题）
  "options": {"A": "...", "B": "...", ...},
  "answer": "A",
  "structural_issues": [],  # 可用但结构异常时记录，由 QDR 统一处理
  "explanation": str,
}

A1/A2 是单题；A3/A4/B1 共享题干 + 多个子问题，但选项不同：
- A3/A4: 每个问题各自有 A-E 选项
- B1: 5 个备选（A-E），后面每个问题都从这同一组备选里选
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import GroupConfig, get_group


# ── md 文件定位 ─────────────────────────────────────────────────


def _flat_files(root: Path) -> List[tuple[str, Path]]:
    out: List[tuple[str, Path]] = []
    for p in sorted(root.glob("case*.md")):
        if p.name.endswith("_meta.json"):
            continue
        m = re.match(r"^case(\d+)\.md$", p.name)
        if not m:
            continue
        case_id = f"case{int(m.group(1)):03d}"
        out.append((case_id, p))
    return out


def _per_stage_files(root: Path) -> List[tuple[str, Path]]:
    out: List[tuple[str, Path]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        m = re.match(r"^case(\d+)$", sub.name)
        if not m:
            continue
        path = sub / "per_stage" / "questions.md"
        if not path.exists():
            continue
        case_id = f"case{int(m.group(1)):03d}"
        out.append((case_id, path))
    return sorted(out, key=lambda x: x[0])


def _nested_question_files(root: Path) -> List[tuple[str, Path]]:
    """新布局：root/caseN/question.md（caseN 非零填充，如 case1/case31）。"""
    out: List[tuple[str, Path]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        m = re.match(r"^case(\d+)$", sub.name)
        if not m:
            continue
        path = sub / "question.md"
        if not path.exists():
            continue
        case_id = f"case{int(m.group(1)):03d}"
        out.append((case_id, path))
    return sorted(out, key=lambda x: x[0])


def discover_files(group: GroupConfig) -> List[tuple[str, Path]]:
    if group.layout == "flat":
        return _flat_files(group.root)
    if group.layout == "per_stage":
        return _per_stage_files(group.root)
    if group.layout == "nested_question":
        return _nested_question_files(group.root)
    raise ValueError(f"未知 layout: {group.layout}")


# ── md 文本解析 ────────────────────────────────────────────────

_QTYPES = ["A1", "A2", "A3", "A4", "B1"]
_EXPECTED_OPTION_KEYS = set("ABCDE")
_OPTION_LABEL_KEYS = set("ABCDEF")

# 题型分隔：## A1型题（...）；忽略中文括号差异
_SECTION_RE = re.compile(
    r"^##\s*(A1|A2|A3|A4|B1)型题",
    re.MULTILINE,
)

_OPTION_RE = re.compile(
    r"^\s*[-*]?\s*([A-F])[\.\．、)）：:]\s*(.+?)\s*$",
    re.MULTILINE,
)

_ANSWER_RE = re.compile(
    r"(?:\*\*\s*)?答\s*案?\s*(?:\*\*)?\s*[:：]\s*([A-E])",
)

_ANSWER_LINE_RE = re.compile(
    r"(?:\*\*\s*)?答\s*案?\s*(?:\*\*)?\s*[:：]\s*([^\n\r]+)",
)

_OPTION_LINE_START_RE = re.compile(
    r"^[ \t]*[-*]?[ \t]*[A-F][\.\．、)）：:]",
    re.MULTILINE,
)

_SUBQUESTION_START_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?问题[ \t]*\d+"
    r"(?:(?:\*\*)?[ \t]*[:：]|(?:\*\*)?[ \t]*$)[ \t]*",
    re.MULTILINE,
)

_NUMBERED_STEM_START_RE = re.compile(
    r"^[ \t]*(?:[-*][ \t]*)?\d+[\.．、)][ \t]*(?:\*\*)?题干(?:\*\*)?[ \t]*[:：][ \t]*",
    re.MULTILINE,
)

_NUMBERED_QUESTION_START_RE = re.compile(
    r"^[ \t]*(?:[-*][ \t]*)?\d+[\.．、)][ \t]*(?![ \t]*(?:\*\*)?题干(?:\*\*)?[ \t]*[:：])"
    r"(?=\S)",
    re.MULTILINE,
)

_CHINESE_NUM = "一二三四五六七八九十"

_HASHED_CHINESE_QUESTION_START_RE = re.compile(
    rf"^[ \t]*#{{1,6}}[ \t]*(?:\*\*)?(?:问题|题目)[{_CHINESE_NUM}0-9]+(?:\*\*)?[ \t]*[:：]?[ \t]*",
    re.MULTILINE,
)

_B1_GENERIC_QUESTION_FIELD_START_RE = re.compile(
    r"^[ \t]*\*\*问题\*\*[ \t]*[:：][ \t]*",
    re.MULTILINE,
)

_BOLDED_NUMBERED_QUESTION_START_RE = re.compile(
    r"^[ \t]*(?:\*\*)?(?:第[ \t]*\d+[ \t]*题|题[ \t]*\d+|题干[ \t]*\d+|题目[ \t]*\d+)"
    r"(?:\*\*)?[ \t]*[:：][ \t]*",
    re.MULTILINE,
)

_BOLDED_STEM_FIELD_NUMBERED_START_RE = re.compile(
    r"^[ \t]*(?:\*\*)?题干(?:\*\*)?[ \t]*[:：][ \t]*\d+[\.．、)][ \t]*",
    re.MULTILINE,
)

_BOLDED_STEM_FIELD_START_RE = re.compile(
    r"^[ \t]*(?:\*\*)?题干(?:\*\*)?[ \t]*[:：][ \t]*",
    re.MULTILINE,
)

_INLINE_QUESTION_START_RE = re.compile(
    r"(?m)(?:^[ \t]*|(?<=[。；;])[ \t]*)(?:\*\*)?问题[ \t]*\d+"
    r"(?:(?:\*\*)?[ \t]*[:：]|(?:\*\*)?[ \t]*$)[ \t]*",
)

_ANSWER_MARKER_RE = re.compile(r"(?:\*\*\s*)?答\s*案?\s*(?:\*\*)?\s*[:：]")
_EXPLANATION_MARKER_RE = re.compile(r"(?:\*\*\s*)?解析(?:\*\*)?\s*[:：]?")
_INLINE_ANSWER_STOP_RE = re.compile(
    r"(?:\s*(?:\*\*)?解析(?:\*\*)?\s*[:：]|"
    r"\s*(?:\*\*)?理由(?:\*\*)?\s*[:：]|"
    r"\s*(?:\*\*)?说明(?:\*\*)?\s*[:：]).*$"
)


def _has_numbered_subquestion_marker(block: str, idx: int) -> bool:
    """判断当前子题块是否明确含有对应的 `问题N` 标记。"""
    pat = re.compile(
        rf"^\s*(?:#{{1,6}}[ \t]*)?(?:\*\*)?问题[ \t]*{idx + 1}"
        rf"(?:(?:\*\*)?[ \t]*[:：]|(?:\*\*)?[ \t]*$)[ \t]*",
        re.MULTILINE,
    )
    return bool(
        pat.search(block)
        or _NUMBERED_STEM_START_RE.search(block)
        or _NUMBERED_QUESTION_START_RE.search(block)
        or _HASHED_CHINESE_QUESTION_START_RE.search(block)
        or _BOLDED_NUMBERED_QUESTION_START_RE.search(block)
        or _BOLDED_STEM_FIELD_NUMBERED_START_RE.search(block)
        or _BOLDED_STEM_FIELD_START_RE.search(block)
        or _INLINE_QUESTION_START_RE.search(block)
    )


def _split_sections(md_text: str) -> List[tuple[str, str, int, int, int, int]]:
    """按题型切分，并保留模型重复输出的同名题型段。"""
    matches = list(_SECTION_RE.finditer(md_text))
    sections: List[tuple[str, str, int, int, int, int]] = []
    qtype_ordinals = {qtype: 0 for qtype in _QTYPES}
    for i, m in enumerate(matches):
        qtype = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        sections.append((qtype, md_text[start:end], i, qtype_ordinals[qtype], m.start(), end))
        qtype_ordinals[qtype] += 1
    return sections


def _extract_field(text: str, name: str) -> str:
    """匹配 `**xx**：...`，在选项/答案/解析/下一字段前停止。"""
    pat = re.compile(
        rf"\*\*{re.escape(name)}\*\*\s*[:：]?\s*",
        re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        return ""
    rest = text[m.end():]
    candidates = []
    for term_re in (
        _OPTION_LINE_START_RE,
        re.compile(r"\n\s*\*\*[^*\n]{1,24}\*\*\s*[:：]?"),
        re.compile(r"\n\s*\n"),
        _ANSWER_MARKER_RE,
        _EXPLANATION_MARKER_RE,
        _INLINE_QUESTION_START_RE,
    ):
        tm = term_re.search(rest)
        if tm:
            candidates.append(tm.start())
    end = min(candidates) if candidates else len(rest)
    return _clean_question_text(rest[:end])


def _clean_question_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[（(]\s*$", "", text).strip()
    text = re.sub(
        r"[（(]?\s*(?:\*\*\s*)?答\s*案?\s*(?:\*\*)?\s*[:：]\s*"
        r"[A-F](?:[,，、]\s*[A-F])*\s*[）)]?\s*$",
        "",
        text,
    ).strip()
    return text


def _extract_question_text(block: str, idx: int) -> str:
    """提取 `**问题N**：...` 的题干，停在第一个选项行/`**答案`/空行/下一加粗。

    A3/A4 经常把选项紧跟题干同段，没有空行，所以用 option-line 作为额外终止点。
    """
    pat = re.compile(
        rf"^\s*(?:#{{1,6}}[ \t]*)?(?:\*\*)?问题[ \t]*{idx + 1}"
        rf"(?:(?:\*\*)?[ \t]*[:：]|(?:\*\*)?[ \t]*$)[ \t]*",
        re.MULTILINE,
    )
    m = pat.search(block)
    if not m:
        m = _NUMBERED_STEM_START_RE.search(block)
    if not m:
        m = _HASHED_CHINESE_QUESTION_START_RE.search(block)
    if not m:
        m = _BOLDED_NUMBERED_QUESTION_START_RE.search(block)
    if not m:
        m = _BOLDED_STEM_FIELD_NUMBERED_START_RE.search(block)
    if not m:
        m = _BOLDED_STEM_FIELD_START_RE.search(block)
    if not m:
        m = _INLINE_QUESTION_START_RE.search(block)
    if not m:
        m = _NUMBERED_QUESTION_START_RE.search(block)
    if not m:
        # 退化匹配：第一个问题小标题，兼容 **问题1** / ### 问题1 / 问题1：
        m = _SUBQUESTION_START_RE.search(block)
        if not m:
            return ""
    rest = block[m.end():].lstrip()
    rest = re.sub(r"^\s*[-*]?\s*(?:\*\*)?题干(?:\*\*)?\s*[:：]\s*", "", rest)
    # 终止符：下一个 \n\s*[-*]? [A-E][. 、 等]、或 \n\s*\*\*、或 \n\n
    candidates = []
    for term_re in (
        _OPTION_LINE_START_RE,
        _ANSWER_MARKER_RE,
        _EXPLANATION_MARKER_RE,
        re.compile(r"\n\s*\*\*答案"),
        re.compile(r"\n\s*\*\*解析"),
        re.compile(r"\n\n"),
    ):
        tm = term_re.search(rest)
        if tm:
            candidates.append(tm.start())
    end = min(candidates) if candidates else len(rest)
    return _clean_question_text(rest[:end])


def _parse_options(block: str) -> Dict[str, str]:
    """从一段 md 文本里提取选项，忽略后续的答案/解析字段。"""
    return _parse_options_with_issues(block)[0]


def _parse_options_with_issues(block: str) -> tuple[Dict[str, str], List[Dict[str, Any]]]:
    """提取选项并保留结构问题。

    选择题输出存在仍可解析、但会影响质量判定的选项瑕疵：
    - 只给 A-D 或 A/B 等部分选项；
    - 有 5 个选项但标签写成 A/B/C/D/D 之类。后者可确定缺失标签时修复标签。
    这些情况对外统一记录为 structural_issue，供 QDR 自动判为题目缺陷。
    """
    end_candidates = []
    for marker_re in (
        re.compile(r"\*\*答案?\*\*"),
        re.compile(r"\*\*解析\*\*"),
        re.compile(r"^答案?\s*[:：]", re.MULTILINE),
        re.compile(r"^解析\s*[:：]", re.MULTILINE),
    ):
        m = marker_re.search(block)
        if m:
            end_candidates.append(m.start())
    end = min(end_candidates) if end_candidates else len(block)
    snippet = block[:end]

    # 用选项标记位置切分，兼容『每行一个』与『同行空格分隔』两种排版。
    # 负向后顾：选项字母前不能是字母/汉字，避免误匹配 "维生素B" 之类。
    marks = list(re.finditer(r"(?<![A-Za-z一-鿿])([A-F])[\.\．、)）：:]\s*", snippet))
    raw_options: List[tuple[str, str]] = []
    raw_labels: List[str] = []
    for i, mk in enumerate(marks):
        label = mk.group(1)
        start = mk.end()
        stop = marks[i + 1].start() if i + 1 < len(marks) else len(snippet)
        value = snippet[start:stop].strip()
        if not value or value.startswith("**"):
            continue
        raw_labels.append(label)
        raw_options.append((label, value))

    issues: List[Dict[str, Any]] = []
    labels = [label for label, _value in raw_options]
    present = set(labels)
    missing = sorted(_EXPECTED_OPTION_KEYS - present)
    unexpected = sorted(present - _EXPECTED_OPTION_KEYS)
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})

    repair_index: Optional[int] = None
    repair_label: Optional[str] = None
    if (
        len(raw_options) == len(_EXPECTED_OPTION_KEYS)
        and len(present) == len(_EXPECTED_OPTION_KEYS) - 1
        and len(missing) == 1
        and len(duplicate_labels) == 1
    ):
        duplicate = duplicate_labels[0]
        seen_duplicate = False
        for idx, (label, _value) in enumerate(raw_options):
            if label != duplicate:
                continue
            if not seen_duplicate:
                seen_duplicate = True
                continue
            repair_index = idx
            repair_label = missing[0]
            break
        issues.append({
            "type": "structural_issue",
            "kind": "option_issue",
            "raw_labels": labels,
            "repair": {"duplicate_label": duplicate, "repaired_label": repair_label},
        })
    elif present and (present != _EXPECTED_OPTION_KEYS or unexpected):
        issues.append({
            "type": "structural_issue",
            "kind": "option_issue",
            "raw_labels": labels,
            "present": sorted(present),
            "missing": missing,
            "unexpected": unexpected,
        })
    elif duplicate_labels:
        issues.append({
            "type": "structural_issue",
            "kind": "option_issue",
            "raw_labels": labels,
            "duplicates": duplicate_labels,
        })
    options: Dict[str, str] = {}
    for idx, (letter, value) in enumerate(raw_options):
        fixed_letter = repair_label if repair_index == idx and repair_label else letter
        if fixed_letter not in options:
            options[fixed_letter] = value
    return options, issues


def _has_complete_options(options: Dict[str, str]) -> bool:
    return set(options.keys()) == _EXPECTED_OPTION_KEYS


def _has_usable_options(options: Dict[str, str]) -> bool:
    return len(options) >= 2


def _add_required_field_issues(
    issues: List[Dict[str, Any]],
    *,
    stem: str,
    answer: str,
    options: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Record missing required fields without discarding the explicit question."""
    out = list(issues)
    if not stem:
        out.append({
            "type": "structural_issue",
            "kind": "missing_stem",
        })
    if not answer:
        out.append({
            "type": "structural_issue",
            "kind": "missing_answer",
        })
    if not _has_usable_options(options):
        out.append({
            "type": "structural_issue",
            "kind": "no_options",
            "raw_labels": sorted(options),
        })
    return out


def _has_answer_line(block: str) -> bool:
    return bool(_ANSWER_LINE_RE.search(block))


def _extract_answer(block: str) -> Optional[str]:
    """提取答案；单选返回 A-F，多选返回逗号连接的 A-F（如 A,C）。"""
    m = _ANSWER_LINE_RE.search(block)
    if not m:
        return None
    return _normalize_answer_value(m.group(1))


def _normalize_answer_value(value: str) -> Optional[str]:
    value = _INLINE_ANSWER_STOP_RE.sub("", value or "").strip()
    value = re.sub(r"[\s*]+", "", value)
    value = value.replace("或", ",").replace("/", ",").replace("、", ",").replace("，", ",")
    m_value = re.match(r"^([A-F](?:,[A-F])*)(?:[）)\]】。.!！?？;；:：(（].*)?$", value)
    if not m_value:
        m_compact = re.match(r"^([A-F]{2,6})(?:[）)\]】。.!！?？;；:：]*)$", value)
        if m_compact:
            letters = list(m_compact.group(1))
            if len(set(letters)) == len(letters):
                return ",".join(letters)
        return None
    letters = re.findall(r"[A-F]", m_value.group(1))
    if len(set(letters)) != len(letters):
        return None
    return ",".join(letters)


def _extract_answer_text(block: str) -> Optional[str]:
    """提取非标准答案文本；用于简答题/无选项题保留可用信息。"""
    m = _ANSWER_LINE_RE.search(block)
    if not m:
        return None
    value = _INLINE_ANSWER_STOP_RE.sub("", m.group(1)).strip().strip("*").strip()
    value = re.sub(r"[）)\]】。.!！?？;；:：]+$", "", value).strip()
    if not value:
        return None
    return value[:500]


def _answer_issue(answer: str, options: Dict[str, str]) -> Optional[Dict[str, Any]]:
    answer = str(answer or "").strip()
    if not answer:
        return None
    labels = [part.strip() for part in re.split(r"[,，、/]|或", answer) if part.strip()]
    if labels and all(re.fullmatch(r"[A-Z]", part) for part in labels):
        option_keys = set(options)
        if not all(label in option_keys for label in labels):
            return {
                "type": "structural_issue",
                "kind": "answer_option_mismatch",
                "answer": answer,
                "options": sorted(option_keys),
            }
        if any(label not in _EXPECTED_OPTION_KEYS for label in labels):
            return {
                "type": "structural_issue",
                "kind": "nonstandard_answer_label",
                "answer": answer,
                "options": sorted(option_keys),
            }
        return None
    return {
        "type": "structural_issue",
        "kind": "non_option_answer",
        "answer": answer[:200],
    }


def _extract_bare_answer_before_explanation(block: str) -> Optional[str]:
    """兼容 B1 子题中答案独立成行、下一段为解析的输出。"""
    lines = block.splitlines()
    hits: List[str] = []
    for i, line in enumerate(lines):
        value = line.strip()
        norm = _normalize_answer_value(value)
        if not norm:
            continue
        next_nonempty = ""
        for nxt in lines[i + 1:]:
            if nxt.strip():
                next_nonempty = nxt.strip()
                break
        if _EXPLANATION_MARKER_RE.match(next_nonempty):
            hits.append(norm)
    return ",".join(hits) if hits else None


def _extract_trailing_question_answer(block: str, idx: int) -> Optional[str]:
    """兼容 B1 子题题干末尾直接写单字母答案的情况。"""
    question_text = _extract_question_text(block, idx)
    if not question_text:
        return None
    m = re.search(r"(?:^|[\s：:，,。])([A-F](?:\s*[,，、/]\s*[A-F])*)\s*$", question_text)
    return m.group(1) if m else None


def _strip_trailing_question_answer(text: str) -> str:
    return re.sub(r"(?:[\s：:，,。]+)[A-F](?:\s*[,，、/]\s*[A-F])*\s*$", "", text).strip()


def _extract_explanation(text: str) -> str:
    """解析整段，遇到下一个 `**问题`/`**题干`/`---` 或文末停。"""
    pat = re.compile(
        r"(?:\*\*解析\*\*|(?:^|\n)解析)\s*[:：]?\s*([\s\S]*?)(?=\n---|\n\s*\*\*问题|\n\s*\*\*题干|\Z)",
    )
    m = pat.search(text)
    return m.group(1).strip() if m else ""


def _split_subquestions(text: str) -> List[str]:
    """A3/A4/B1：按 `**问题N**` 或 `1. **题干**` 切子题块。"""
    match_infos = sorted(
        [(m, False, "question") for m in _SUBQUESTION_START_RE.finditer(text)]
        + [(m, False, "numbered_stem") for m in _NUMBERED_STEM_START_RE.finditer(text)]
        + [(m, False, "question") for m in _HASHED_CHINESE_QUESTION_START_RE.finditer(text)]
        + [(m, False, "question") for m in _BOLDED_NUMBERED_QUESTION_START_RE.finditer(text)]
        + [(m, False, "numbered_stem") for m in _BOLDED_STEM_FIELD_NUMBERED_START_RE.finditer(text)]
        + [(m, False, "stem_field") for m in _BOLDED_STEM_FIELD_START_RE.finditer(text)]
        + [(m, True, "inline_question") for m in _INLINE_QUESTION_START_RE.finditer(text)]
        + [(m, True, "broad_number") for m in _NUMBERED_QUESTION_START_RE.finditer(text)],
        key=lambda item: item[0].start(),
    )
    accepted: List[tuple[Any, str]] = []
    for i, (m, is_broad, kind) in enumerate(match_infos):
        if accepted and m.start() == accepted[-1][0].start():
            # The more specific expression is inserted first above.
            continue
        if (
            kind == "stem_field"
            and accepted
            and accepted[-1][1] in {"question", "numbered_stem"}
            and not text[accepted[-1][0].end():m.start()].strip()
        ):
            # `**问题1**` followed by `**题干**` is one question, not two.
            continue
        if not is_broad:
            accepted.append((m, kind))
            continue
        next_start = match_infos[i + 1][0].start() if i + 1 < len(match_infos) else len(text)
        snippet = text[m.start():next_start]
        # Inline `问题N` references are common in explanations, so they still
        # require an answer. A line-leading numbered block is explicit enough
        # when it has an answer, usable options, or question punctuation.
        options, _issues = _parse_options_with_issues(snippet)
        numbered_stem = _extract_question_text(snippet, 0) if kind == "broad_number" else ""
        if (
            _extract_answer(snippet)
            or (
                kind == "broad_number"
                and (_has_usable_options(options) or numbered_stem.rstrip().endswith(("?", "？")))
            )
        ):
            accepted.append((m, kind))
    matches = [m for m, _kind in accepted]
    if not matches:
        return []
    blocks: List[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append(text[start:end])
    return blocks


def _has_explicit_subquestion_marker(block: str) -> bool:
    """Return whether a block starts with a deliberate question/stem marker.

    Bare numbered prose such as ``1. 诊断依据`` is intentionally excluded: it
    is also common inside explanations and must not become a missing-answer item.
    """
    if re.match(r"^[ \t]*\*\*[ \t]*问题[ \t]*\d+", block):
        return True
    explicit_patterns = (
        _NUMBERED_STEM_START_RE,
        _HASHED_CHINESE_QUESTION_START_RE,
        _BOLDED_NUMBERED_QUESTION_START_RE,
        _BOLDED_STEM_FIELD_NUMBERED_START_RE,
        _BOLDED_STEM_FIELD_START_RE,
    )
    return any(pattern.match(block) for pattern in explicit_patterns)


def _looks_like_single_question_block(body: str) -> bool:
    options, _issues = _parse_options_with_issues(body)
    return bool(_ANSWER_LINE_RE.search(body) and _has_usable_options(options))


def _parse_a_single(qtype: str, body: str) -> List[Dict[str, Any]]:
    """A1/A2 单题。"""
    shared_stem = _extract_field(body, "共享题干") if qtype == "A2" else ""
    stem = _extract_field(body, "题干")
    if not stem and qtype == "A2":
        stem = _extract_field(body, "问题")
    answer = _extract_answer(body) or _extract_answer_text(body) or ""
    options, structural_issues = _parse_options_with_issues(body)
    structural_issues = _add_required_field_issues(
        structural_issues,
        stem=stem,
        answer=answer,
        options=options,
    )
    answer_issue = _answer_issue(answer, options)
    if answer_issue:
        structural_issues = structural_issues + [answer_issue]
    return [{
        "qtype": qtype,
        "sub_idx": 0,
        "shared_stem": shared_stem,
        "stem": stem,
        "options": options,
        "answer": answer,
        "structural_issues": structural_issues,
        "explanation": _extract_explanation(body),
    }]


def _parse_a_group(qtype: str, body: str) -> List[Dict[str, Any]]:
    """A3/A4：共享题干 + 多个子题，子题各自有选项。"""
    shared_stem = _extract_field(body, "共享题干") or _extract_field(body, "题干")
    blocks = _split_subquestions(body)
    if not blocks and _looks_like_single_question_block(body):
        blocks = [body]
    out: List[Dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        sub_idx = len(out)
        question_text = _extract_question_text(block, sub_idx)
        answer = _extract_answer(block) or _extract_answer_text(block) or ""
        options, structural_issues = _parse_options_with_issues(block)
        if block is body:
            structural_issues = structural_issues + [{
                "type": "structural_issue",
                "kind": "group_without_subquestions",
            }]
        if not question_text:
            if block is body:
                question_text = shared_stem
        structural_issues = _add_required_field_issues(
            structural_issues,
            stem=question_text,
            answer=answer,
            options=options,
        )
        answer_issue = _answer_issue(answer, options)
        if answer_issue:
            structural_issues = structural_issues + [answer_issue]
        out.append({
            "qtype": qtype,
            "sub_idx": sub_idx,
            "shared_stem": shared_stem,
            "stem": question_text,
            "options": options,
            "answer": answer,
            "structural_issues": structural_issues,
            "explanation": _extract_explanation(block),
        })
    return out


def _extract_b1_alternative_text(body: str) -> str:
    marker = re.search(r"\*\*备选答案\*\*\s*[:：]?\s*", body)
    if not marker:
        return body

    rest = body[marker.end():]
    stops: List[int] = []
    blocks = _split_subquestions(rest)
    if blocks:
        first_block_start = rest.find(blocks[0])
        if first_block_start >= 0:
            stops.append(first_block_start)
    for stop_re in (
        re.compile(r"^[ \t]*---[ \t]*$", re.MULTILINE),
        _B1_GENERIC_QUESTION_FIELD_START_RE,
    ):
        match = stop_re.search(rest)
        if match:
            stops.append(match.start())
    return rest[:min(stops)] if stops else rest


def _parse_b1(body: str) -> List[Dict[str, Any]]:
    """B1：备选答案 A-E + 多个 `**问题N**`，子题共享备选。"""
    alt_text = _extract_b1_alternative_text(body)
    options, structural_issues = _parse_options_with_issues(alt_text + "\n**答案**")  # 强制截断

    blocks = _split_subquestions(body)
    out: List[Dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        sub_idx = len(out)
        question_text = _extract_question_text(block, sub_idx)
        answer = (
            _extract_answer(block)
            or _extract_bare_answer_before_explanation(block)
            or _extract_trailing_question_answer(block, sub_idx)
            or _extract_answer_text(block)
            or ""
        )
        norm = _normalize_answer_value(answer)
        if norm:
            answer = norm
        question_text = _strip_trailing_question_answer(question_text)
        if not answer and question_text and block_index + 1 < len(blocks):
            next_block = blocks[block_index + 1]
            next_question = _strip_trailing_question_answer(
                _extract_question_text(next_block, sub_idx) or ""
            )
            next_answer = _extract_answer(next_block) or _extract_answer_text(next_block)
            if next_answer and next_question == question_text:
                # Some outputs repeat a heading before the complete question body.
                # This is one represented question, not an omitted-answer question.
                continue
        item_issues = _add_required_field_issues(
            structural_issues,
            stem=question_text,
            answer=answer,
            options=options,
        )
        answer_issue = _answer_issue(answer, options)
        if answer_issue:
            item_issues.append(answer_issue)
        out.append({
            "qtype": "B1",
            "sub_idx": sub_idx,
            "shared_stem": "",     # B1 没有真正的共享题干，只有备选
            "stem": question_text,
            "options": options,
            "answer": answer,
            "structural_issues": item_issues,
            "explanation": _extract_explanation(block),
        })
    return out


def parse_case(md_text: str) -> List[Dict[str, Any]]:
    """从一份 case 的 md 解析出所有子题。"""
    # 规整字面转义换行：部分 case 把选项用字面 "\n"（反斜杠+n 两字符）写在同一物理行，
    # 导致行锚定的选项正则只识别行首的 A，B-E 被吞进 A 的内容里。先还原成真换行。
    md_text = md_text.replace("\\r\\n", "\n").replace("\\n", "\n")
    sections = _split_sections(md_text)
    items: List[Dict[str, Any]] = []
    qtype_to_group_idx = {q: i for i, q in enumerate(_QTYPES)}
    next_sub_idx = {qtype: 0 for qtype in _QTYPES}
    for qtype, body, section_index, section_ordinal, section_start, section_end in sections:
        if qtype in {"A1", "A2"}:
            sub_items = _parse_a_single(qtype, body)
        elif qtype in {"A3", "A4"}:
            sub_items = _parse_a_group(qtype, body)
        elif qtype == "B1":
            sub_items = _parse_b1(body)
        else:
            continue
        for s in sub_items:
            s["group_idx"] = qtype_to_group_idx[qtype]
            s["sub_idx"] = next_sub_idx[qtype]
            s["source_section_index"] = section_index
            s["source_section_ordinal"] = section_ordinal
            s["source_section_start"] = section_start
            s["source_section_end"] = section_end
            next_sub_idx[qtype] += 1
        items.extend(sub_items)
    return items


# ── 入口 ───────────────────────────────────────────────────────


def parse_group(group_key: str, groups: Optional[List[GroupConfig]] = None) -> List[Dict[str, Any]]:
    group = get_group(group_key, groups=groups)
    if group.layout == "preparsed_jsonl":
        out = []
        for item in read_jsonl(group.root):
            rec = dict(item)
            rec["group"] = group_key
            out.append(rec)
        return out
    files = discover_files(group)
    out: List[Dict[str, Any]] = []
    for case_id, path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise UnicodeError(f"题目文件 UTF-8 解码失败: {path}") from exc
        for item in parse_case(text):
            item["group"] = group_key
            item["case_id"] = case_id
            out.append(item)
    return out


def write_jsonl(items: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    n = 0
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return n


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
