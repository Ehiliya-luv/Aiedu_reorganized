from __future__ import annotations


def test_public_entry_modules_import() -> None:
    import dpo_main
    import paper_reproduction_main
    import question_llm_judge_main
    import question_quality_main
    import reasoning_quality_main

    assert callable(dpo_main.main)
    assert callable(paper_reproduction_main.main)
    assert callable(question_llm_judge_main.main)
    assert callable(question_quality_main.main)
    assert callable(reasoning_quality_main.main)
