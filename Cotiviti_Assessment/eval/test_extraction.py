"""
Phase 7 – Rule extraction tests (require Ollama, marked with 'llm').
"""

import pytest
from pydantic import ValidationError

from src.schemas import PolicyRule, Condition
from src.retrieve import retrieve
from src.extract import extract_rule


pytestmark = pytest.mark.llm


def test_extract_returns_policy_rule():
    chunks = retrieve("chiropractic maintenance therapy", top_k=1)
    rule = extract_rule(chunks[0])
    assert isinstance(rule, PolicyRule)


def test_extracted_rule_has_source_citation():
    chunks = retrieve("chiropractic maintenance therapy", top_k=1)
    rule = extract_rule(chunks[0])
    assert rule.source_chapter in {"15", "16"}
    assert rule.source_section
    assert rule.source_clause


def test_extracted_rule_conditions_are_valid():
    chunks = retrieve("cosmetic surgery exclusion", top_k=1)
    rule = extract_rule(chunks[0])
    for cond in rule.covered_if + rule.excluded_if:
        assert cond.operator in {"==", "!=", ">=", "<=", "in", "not_in"}
        assert isinstance(cond.field, str) and cond.field


def test_extracted_rule_validates_with_pydantic():
    chunks = retrieve("services outside the United States", top_k=1)
    rule = extract_rule(chunks[0])
    # Re-validate round-trip: serialize → parse
    raw = rule.model_dump()
    rule2 = PolicyRule.model_validate(raw)
    assert rule2.rule_id == rule.rule_id
