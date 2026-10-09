"""Pairwise comparison prompts for medical tasks.

Each prompt returns a str that asks the Judge to answer with JSON {"winner": "A"/"B"}.
"""

from typing import Callable

# Known task name -> prompt function registry
PROMPT_REGISTRY: dict[str, Callable[[str, str, str], str]] = {}
PROMPT_VARIANTS = ("current", "legacy_question", "prompt_v3", "prompt_v4")

STRICT_OUTPUT_REQUIREMENT = """【输出要求】
你必须只输出且只能输出下面两种 JSON 中的任意一个，不能包含任何解释、前后缀、代码块、Markdown、空行或其他字符：
{"winner": "A", "reason": "A 的...优于 B 的...（简要说明具体原因，1-2句话）"}
{"winner": "B", "reason": "B 的...优于 A 的...（简要说明具体原因，1-2句话）"}
reason 字段必须简要说明选择该模型的具体原因（1-2句话，中文）。除了这个 JSON 之外，任何输出都视为无效。"""


def _register(task_name: str):
    """Decorator: register prompt function to PROMPT_REGISTRY."""

    def decorator(fn: Callable[[str, str, str], str]):
        PROMPT_REGISTRY[task_name] = fn
        return fn

    return decorator


@_register("SOAP")
@_register("病历标准化")
def soap_judge_prompt(raw_case: str, output_a: str, output_b: str) -> str:
    """SOAP medical record standardization pairwise prompt."""
    return f"""【医疗病历标准化评估】
你是一名资深临床医师，请评估以下两份 SOAP 格式病历的质量。

【原始病历】
{raw_case}

【模型 A 生成的 SOAP 病历】
{output_a}

【模型 B 生成的 SOAP 病历】
{output_b}

【评估维度】
1. 结构完整性：是否包含 S(主观)、O(客观)、A(评估)、P(计划) 四部分，且分段清晰
2. 术语规范性：医学术语使用是否准确、符合临床书写规范
3. 逻辑连贯性：主诉→现病史→评估→计划的推理链条是否合理
4. 关键信息覆盖：是否遗漏原始病历中的重要诊断、检查或治疗信息

{STRICT_OUTPUT_REQUIREMENT}
"""


@_register("题目生成")
@_register("考题生成")
def question_judge_prompt(raw_case: str, output_a: str, output_b: str) -> str:
    """Medical question generation pairwise prompt."""
    return f"""【医疗考题生成质量评估】
你是一名医学教育专家，请评估以下两份基于同一病历生成的考题质量。

【原始病历】
{raw_case}

【模型 A 生成的考题】
{output_a}

【模型 B 生成的考题】
{output_b}

【评判原则】
考题应基于病例事实，允许合理概括或适度延展，但不能扭曲核心诊断、检查、治疗或病程信息。
不要因为回答更长、格式更华丽、题量更多或覆盖更多无关知识而加分。
如果题干、选项或表述直接泄露答案，使考生无需医学推理即可作答，应降低评价。

【评估维度】
1. 题干清晰度：问题表述是否明确、无歧义，符合医学考试命题规范
2. 答案准确性：正确答案是否有充分病历依据，解析是否专业
3. 干扰项合理性：错误选项是否为临床常见误区，具有鉴别价值
4. 临床相关性：考题是否紧扣病历核心诊疗要点，避免偏题
5. 推理深度：是否考察临床思维层次，而非简单记忆复述

{STRICT_OUTPUT_REQUIREMENT}
"""


def question_judge_prompt_v3(raw_case: str, output_a: str, output_b: str) -> str:
    """Medical question generation pairwise prompt with minimal QDR-sensitive wording."""
    return f"""【医疗考题生成质量评估】
你是一名医学教育专家，请评估以下两份基于同一病历生成的考题质量。

【原始病历】
{raw_case}

【模型 A 生成的考题】
{output_a}

【模型 B 生成的考题】
{output_b}

【评判原则】
考题应基于病例事实，允许合理概括或适度延展，但不能扭曲核心诊断、检查、治疗或病程信息。
不要因为回答更长、格式更华丽、题量更多或覆盖更多无关知识而加分。
如果题干、选项或表述直接泄露答案，使考生无需医学推理即可作答，应降低评价。
题目正确性优先：若存在明确答案错误、多答案、无答案、题干不可解或解析与答案矛盾，应显著降分，且不能由格式、题量或篇幅抵消。

【评估维度】
1. 题干清晰度：问题表述是否明确、无歧义，符合医学考试命题规范
2. 答案准确性：标注答案是否正确且唯一，是否有充分病历依据，解析是否专业并与答案一致
3. 干扰项合理性：错误选项是否为临床常见误区，具有鉴别价值
4. 临床相关性：考题是否紧扣病历核心诊疗要点，避免偏题
5. 推理深度：是否考察临床思维层次，而非简单记忆复述

{STRICT_OUTPUT_REQUIREMENT}
"""


def question_judge_prompt_v4(raw_case: str, output_a: str, output_b: str) -> str:
    """Medical question generation pairwise prompt that penalizes generic short-template questions."""
    return f"""【医疗考题生成质量评估】
你是一名医学教育专家，请评估以下两份基于同一病历生成的考题质量。

【原始病历】
{raw_case}

【模型 A 生成的考题】
{output_a}

【模型 B 生成的考题】
{output_b}

【评判原则】
考题应基于病例事实，允许合理概括或适度延展，但不能扭曲核心诊断、检查、治疗或病程信息。
题目正确性优先：若存在明确答案错误、多答案、无答案、题干不可解或解析与答案矛盾，应显著降分，且不能由格式、题量或篇幅抵消。只有在病历事实、题干信息或明确临床共识直接支持时，才应判定为“明确答案错误”；对可能存在不同临床处理路径或表述不够完美的题目，不要轻易按严重错误处理。
病例特异性优先：优秀考题应充分利用原始病历中的关键诊断、主诉病程、阳性/阴性检查、治疗经过、并发症风险和随访管理信息；若只围绕疾病通用常识、金标准、最常见表现、常规治疗等泛化知识出题，即使格式简洁规范，也应降低评价。
临床推理优先：优先选择能考察诊断鉴别、检查选择、治疗决策、风险预判、术后/出院管理等连续临床思维链条的考题；不要偏好只需记忆单个知识点即可作答的题目。
整体质量优先：不要只抓住一两道题的局部问题就忽略整套考题的综合质量；只有当错误明确、严重且影响核心诊疗判断时，才应压倒病例特异性、临床推理深度和整体教学价值。
不要因为回答更短、更整齐、格式更统一、题干更像模板、题量更多或覆盖更多无关知识而加分。
如果题干、选项或表述直接泄露答案，使考生无需结合病历和医学推理即可作答，应降低评价。
如果一份考题虽然答案看似正确，但主要由短小、通用、模板化的单点知识题组成，缺少对原始病历独有细节和诊疗链条的利用，应明显降分；不要把“简洁正确”误判为“高质量”。这类考题的质量上限通常不应高于中等，除非另一份考题存在大量明确严重错误或整体脱离病历。

【高质量考题参考样例】
以下样例来自专家认可的医学考题风格，仅用于校准“高质量”的判断标准，不要求两份候选考题复刻样例内容。

样例 1（A2 型题）：患者男，30 岁。近 4 年经常间发四肢关节疼痛，近来乏力、纳差、心悸、气促；肝肋下 2.5 cm 触及，轻触痛；血红蛋白 97g/L，尿蛋白(+)，双下肢轻度浮肿。问最可能原因。正确答案为“二尖瓣狭窄致右心衰”。该题的质量来自：题干整合了长期关节痛、心悸气促、肝大、蛋白尿、水肿等病例线索；答案不是单点记忆，而是要求考生把症状体征串联到右心衰机制；解析能说明正确答案依据，并辨析肝硬化、急性肾炎、左心衰、营养不良等干扰项。

样例 2（A4 型题）：COPD 患者长期呼吸困难，突发气急显著加重并伴胸痛，后续问题依次考察病史采集重点、体检重点、确诊检查、确诊气胸后的治疗目的，以及若不能证实气胸时需考虑肺栓塞。该题的质量来自：围绕同一病例逐步展开病情；问题覆盖病史、查体、辅助检查、治疗目标和鉴别诊断；每一问都依赖病例中的“COPD、突发气急、胸痛、急诊”这些特异信息，而不是孤立考“气胸金标准”或“最常见表现”。

【评估维度】
1. 题干清晰度：问题表述是否明确、无歧义，符合医学考试命题规范
2. 答案准确性：标注答案是否正确且唯一，是否有充分病历依据，解析是否专业并与答案一致；区分核心错误和局部可争议表述，避免因个别小瑕疵否定整体更优的考题
3. 干扰项合理性：错误选项是否为临床常见误区，具有鉴别价值，而不是机械凑数
4. 临床相关性：考题是否紧扣病历核心诊疗要点，体现病例特异信息和诊疗过程，避免泛化、偏题或只考疾病常识
5. 推理深度：是否考察临床思维层次、诊疗链条和决策权衡，而非简单记忆复述；连续使用病历细节进行推理的考题应优于短模板知识题

在两份考题都存在瑕疵时，优先选择病例特异性更强、临床推理更深入、整体教学价值更高且答案基本可靠的一份；不要被短小、整洁、通用化的表面质量误导。reason 中应同时说明胜出方的核心优势和落败方的核心缺陷，避免只列举落败方个别题目的局部问题。

{STRICT_OUTPUT_REQUIREMENT}
"""


def legacy_question_judge_prompt(raw_case: str, output_a: str, output_b: str) -> str:
    """Original medical question generation pairwise prompt before refinement."""
    return f"""【医疗考题生成质量评估】
你是一名医学教育专家，请评估以下两份基于同一病历生成的考题质量。

【原始病历】
{raw_case}

【模型 A 生成的考题】
{output_a}

【模型 B 生成的考题】
{output_b}

【评估维度】
1. 题干清晰度：问题表述是否明确、无歧义，符合医学考试命题规范
2. 答案准确性：正确答案是否有充分病历依据，解析是否专业
3. 干扰项合理性：错误选项是否为临床常见误区，具有鉴别价值
4. 临床相关性：考题是否紧扣病历核心诊疗要点，避免偏题
5. 推理深度：是否考察临床思维层次，而非简单记忆复述

{STRICT_OUTPUT_REQUIREMENT}
"""


@_register("临床思维")
@_register("临床思维提取")
def clinical_thinking_judge_prompt(raw_case: str, output_a: str, output_b: str) -> str:
    """Clinical thinking extraction pairwise prompt."""
    return f"""【临床思维提取质量评估】
你是一名资深临床带教老师，请比较以下两份基于同一病历生成的临床思维要点，选择质量更高的一份。

【原始病历】
{raw_case}

【模型 A 生成的临床思维要点】
{output_a}

【模型 B 生成的临床思维要点】
{output_b}

【评估维度】
1. 病史整合：是否准确提炼主诉、现病史时间线和关键鉴别信息
2. 检查解读：是否针对性分析体格检查、辅助检查的阳性和阴性意义
3. 诊断推理：是否给出合理诊断依据、鉴别诊断和排除逻辑
4. 治疗决策：是否体现个体化治疗、风险预判、动态评估和随访计划
5. 教学价值：是否结构清晰、依据充分，能帮助学员形成临床思维

{STRICT_OUTPUT_REQUIREMENT}
"""


@_register("病历综合评分")
@_register("病历评分")
def scoring_judge_prompt(raw_case: str, output_a: str, output_b: str) -> str:
    """Medical record comprehensive scoring pairwise prompt."""
    return f"""【病历综合评分质量评估】
你是一名资深医疗质量评审专家，请比较以下两份基于同一病历生成的综合评价与评分，选择更专业、更可靠的一份。

【原始病历】
{raw_case}

【模型 A 生成的病历综合评分】
{output_a}

【模型 B 生成的病历综合评分】
{output_b}

【评估维度】
1. 维度覆盖：是否覆盖病例资料完整度、书写规范性、学习内容丰富度、典型性、稀缺性、诊疗思路完整性、MDT需求度
2. 依据充分：评价是否紧扣原始病历信息，避免空泛或臆测
3. 评分合理：1-10分评分是否与各维度评价一致，尺度使用是否稳定
4. 医学专业性：术语、诊疗逻辑、指南或规范引用是否准确可信
5. 可读性：结构是否清晰，便于后续人工复核和教学使用

{STRICT_OUTPUT_REQUIREMENT}
"""


# Default task priority order (for --tasks auto)
DEFAULT_TASK_ORDER = ["病历标准化", "考题生成", "临床思维", "病历综合评分"]

# Task name aliases (normalize to PROMPT_REGISTRY keys)
TASK_ALIASES: dict[str, str] = {
    "SOAP": "病历标准化",
    "题目生成": "考题生成",
    "临床思维提取": "临床思维",
    "病历评分": "病历综合评分",
}


def resolve_task_name(raw_name: str) -> str:
    """Resolve task name alias to standard name."""
    return TASK_ALIASES.get(raw_name, raw_name)


def get_judge_prompt(
    task: str,
    raw_case: str,
    output_a: str,
    output_b: str,
    *,
    prompt_variant: str = "current",
) -> str:
    """Get pairwise judge prompt by task name.

    Args:
        task:     Task name (supports aliases)
        raw_case: Raw medical record text
        output_a: Model A output
        output_b: Model B output
        prompt_variant: Prompt variant to use. Question-only variants change 考题生成.

    Returns:
        Complete judge prompt string

    Raises:
        NotImplementedError: Task not registered
    """
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt_variant {prompt_variant!r}. Expected one of {PROMPT_VARIANTS}.")
    resolved = resolve_task_name(task)
    if prompt_variant == "prompt_v3" and resolved == "考题生成":
        return question_judge_prompt_v3(raw_case, output_a, output_b)
    if prompt_variant == "prompt_v4" and resolved == "考题生成":
        return question_judge_prompt_v4(raw_case, output_a, output_b)
    if prompt_variant == "legacy_question" and resolved == "考题生成":
        return legacy_question_judge_prompt(raw_case, output_a, output_b)
    fn = PROMPT_REGISTRY.get(resolved)
    if fn is None:
        raise NotImplementedError(
            f"Task '{task}' (resolved to '{resolved}') judge prompt not implemented."
            f"Registered tasks: {list(PROMPT_REGISTRY.keys())}"
        )
    return fn(raw_case, output_a, output_b)
