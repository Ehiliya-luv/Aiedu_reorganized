from __future__ import annotations

from tools.reasoning_hallucination.run_reasoning_hallucination_audit import _mock_counts, _normalize


def test_reasoning_fixture_counts_are_consistent() -> None:
    result = _normalize(_mock_counts("[supported] one\n[inferable] two\n[unsupported] three"))
    assert result["total_claims"] == 3
    assert result["case_consistency"] == 2 / 3
