"""
Phase 4 – Deterministic Adjudication
Pure Python, no LLM. adjudicate(rule, claim) -> Decision.

Trust boundary:
  excluded_if any match  → DENY
  covered_if all match   → PAY
  otherwise              → REVIEW
"""

from __future__ import annotations

from src.schemas import Claim, Condition, Decision, PolicyRule


def _eval_condition(condition: Condition, attributes: dict) -> bool:
    """Evaluate a single Condition against claim attributes. Missing field → False."""
    if condition.field not in attributes:
        return False

    actual = attributes[condition.field]
    expected = condition.value
    op = condition.operator

    if op == "==":
        return str(actual).lower() == str(expected).lower()
    if op == "!=":
        return str(actual).lower() != str(expected).lower()
    if op == ">=":
        try:
            return float(actual) >= float(expected)
        except (TypeError, ValueError):
            return False
    if op == "<=":
        try:
            return float(actual) <= float(expected)
        except (TypeError, ValueError):
            return False
    if op == "in":
        if not isinstance(expected, list):
            expected = [expected]
        return str(actual).lower() in [str(v).lower() for v in expected]
    if op == "not_in":
        if not isinstance(expected, list):
            expected = [expected]
        return str(actual).lower() not in [str(v).lower() for v in expected]

    return False


def adjudicate(rule: PolicyRule, claim: Claim) -> Decision:
    """
    Deterministic adjudication — no model in the loop.

    Priority:
      1. If any excluded_if condition matches → DENY
      2. If all covered_if conditions match   → PAY
      3. Otherwise                            → REVIEW
    """
    attrs = claim.attributes

    # Check exclusions first (Ch.16 exclusions win)
    for cond in rule.excluded_if:
        if _eval_condition(cond, attrs):
            return Decision(
                claim_id=claim.claim_id,
                outcome="DENY",
                rationale=(
                    f"Excluded per {rule.source_clause_span()}: "
                    f"claim attribute '{cond.field}' {cond.operator} {cond.value!r} matched."
                ),
                matched_rule_id=rule.rule_id,
                matched_clause=rule.source_clause,
            )

    # Check coverage conditions
    if rule.covered_if:
        unmet = [c for c in rule.covered_if if not _eval_condition(c, attrs)]
        if not unmet:
            return Decision(
                claim_id=claim.claim_id,
                outcome="PAY",
                rationale=(
                    f"All coverage conditions met per {rule.source_clause_span()}: "
                    f"{len(rule.covered_if)} condition(s) satisfied."
                ),
                matched_rule_id=rule.rule_id,
                matched_clause=rule.source_clause,
            )
        # Some covered_if conditions unmet → REVIEW
        missing_fields = [c.field for c in unmet]
        return Decision(
            claim_id=claim.claim_id,
            outcome="REVIEW",
            rationale=(
                f"Coverage conditions not fully met per {rule.source_clause_span()}. "
                f"Unmet conditions on field(s): {missing_fields}."
            ),
            matched_rule_id=rule.rule_id,
            matched_clause=rule.source_clause,
        )

    # No conditions at all → cannot determine → REVIEW
    return Decision(
        claim_id=claim.claim_id,
        outcome="REVIEW",
        rationale=f"Rule {rule.rule_id} has no conditions defined; manual review required.",
        matched_rule_id=rule.rule_id,
        matched_clause=rule.source_clause,
    )

