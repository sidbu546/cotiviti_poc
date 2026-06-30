"""
Phase 7 – Adjudication tests (pure Python, no LLM, always-on).
"""

import pytest
from src.schemas import Claim, Condition, PolicyRule
from src.adjudicate import adjudicate

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _rule(
    rule_id: str = "test_rule",
    covered_if: list[dict] | None = None,
    excluded_if: list[dict] | None = None,
    chapter: str = "15",
    section: str = "1",
) -> PolicyRule:
    return PolicyRule(
        rule_id=rule_id,
        description="Test rule",
        covered_if=[Condition(**c) for c in (covered_if or [])],
        excluded_if=[Condition(**c) for c in (excluded_if or [])],
        source_chapter=chapter,
        source_section=section,
        source_clause="Test clause text.",
    )


def _claim(claim_id: str, attrs: dict) -> Claim:
    return Claim(claim_id=claim_id, service_type="test", attributes=attrs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_excluded_if_match_returns_deny():
    rule = _rule(excluded_if=[{"field": "therapy_goal", "operator": "==", "value": "maintenance"}])
    claim = _claim("C1", {"therapy_goal": "maintenance"})
    decision = adjudicate(rule, claim)
    assert decision.outcome == "DENY"
    assert decision.claim_id == "C1"


def test_all_covered_if_match_returns_pay():
    rule = _rule(covered_if=[
        {"field": "location", "operator": "==", "value": "domestic"},
        {"field": "initiator", "operator": "==", "value": "patient"},
    ])
    claim = _claim("C2", {"location": "domestic", "initiator": "patient"})
    decision = adjudicate(rule, claim)
    assert decision.outcome == "PAY"


def test_partial_covered_if_returns_review():
    rule = _rule(covered_if=[
        {"field": "location", "operator": "==", "value": "domestic"},
        {"field": "initiator", "operator": "==", "value": "patient"},
    ])
    claim = _claim("C3", {"location": "domestic"})  # initiator missing
    decision = adjudicate(rule, claim)
    assert decision.outcome == "REVIEW"


def test_exclusion_beats_coverage():
    rule = _rule(
        covered_if=[{"field": "location", "operator": "==", "value": "domestic"}],
        excluded_if=[{"field": "therapy_goal", "operator": "==", "value": "maintenance"}],
    )
    claim = _claim("C4", {"location": "domestic", "therapy_goal": "maintenance"})
    decision = adjudicate(rule, claim)
    assert decision.outcome == "DENY", "Exclusion must win over matching coverage condition"


def test_empty_conditions_returns_review():
    rule = _rule(covered_if=[], excluded_if=[])
    claim = _claim("C5", {"location": "domestic"})
    decision = adjudicate(rule, claim)
    assert decision.outcome == "REVIEW"


def test_decision_has_citation():
    rule = _rule(excluded_if=[{"field": "therapy_goal", "operator": "==", "value": "maintenance"}])
    claim = _claim("C6", {"therapy_goal": "maintenance"})
    decision = adjudicate(rule, claim)
    assert decision.matched_clause is not None
    assert decision.matched_rule_id == "test_rule"


def test_in_operator():
    rule = _rule(covered_if=[{"field": "service_type", "operator": "in", "value": ["PT", "OT", "ST"]}])
    claim_yes = _claim("C7a", {"service_type": "PT"})
    claim_no = _claim("C7b", {"service_type": "chiro"})
    assert adjudicate(rule, claim_yes).outcome == "PAY"
    assert adjudicate(rule, claim_no).outcome == "REVIEW"


def test_not_in_operator():
    rule = _rule(excluded_if=[{"field": "location", "operator": "not_in", "value": ["domestic", "us_territory"]}])
    claim_deny = _claim("C8a", {"location": "outside_united_states"})
    claim_ok = _claim("C8b", {"location": "domestic"})
    assert adjudicate(rule, claim_deny).outcome == "DENY"
    assert adjudicate(rule, claim_ok).outcome == "REVIEW"  # no covered_if
