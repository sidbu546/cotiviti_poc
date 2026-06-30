"""
Policy-to-Code
Converts an English policy paragraph into a complete, executable Python rule module.
"""

from __future__ import annotations

import json
import re

from src.llm.provider import chat

_MIN_CHARS = 80

_SYSTEM = """\
You are a senior policy engineer. The user will paste a healthcare coverage policy paragraph.

Your job: generate a COMPLETE, EXECUTABLE Python 3 module that fully implements every rule in that policy.

Rules for the code you generate:
1. Define one or more CONSTANT sets/dicts at the top for any enumerated values \
   (e.g. RED_FLAG_CONDITIONS, CONSERVATIVE_TREATMENTS, COVERED_DIAGNOSES).
2. Write a main function  def is_covered(claim: dict) -> dict  that returns
   {"decision": "APPROVE" | "DENY", "reason": "<plain-english explanation>"}.
3. Implement EVERY rule in the paragraph:
   - Coverage criteria (what must be true to approve)
   - Exclusions (what causes denial)
   - Waivers / exceptions (conditions that skip a requirement)
   - Time/frequency limits (e.g. "max 1 per 6-month period")
   - Numeric thresholds (e.g. "at least six weeks")
4. Each major branch MUST have an inline comment that quotes the relevant policy clause.
5. Include a short  __main__  block with 2-3 example claims that demonstrate APPROVE, DENY, and waiver.
6. Use snake_case field names that naturally match the policy concepts.
7. The code must be syntactically valid Python 3 — no pseudocode, no placeholders.

Output ONLY the Python code. No markdown fences, no explanations, no extra text before or after the code.
"""

_JSON_SYSTEM = """\
You are a senior policy engineer. The user will paste a healthcare coverage policy paragraph.

Your job: model EVERY sentence and EVERY rule in that paragraph as a generalized JSON rules document.
Read the paragraph carefully sentence by sentence. Each sentence that states a rule, condition, exclusion,
waiver, or limit MUST produce at least one rule object. Do NOT skip, merge, or summarize sentences.

There is NO fixed schema — do not limit yourself to flat fields like rule_id/description/covered_if/excluded_if.
Build whatever nested structure is needed to fully capture the policy's logic.

Required per rule object:
- "rule_type": one of "coverage_criterion", "exclusion", "waiver", "frequency_limit", "threshold", \
  or another name that fits.
- "source_text": copy the EXACT sentence (or clause) from the paragraph that this rule was derived from.
- "conditions": a nested boolean tree using "all_of", "any_of", "not", or \
  {"field": ..., "comparator": ..., "value": ...} / {"field": ..., "in": [...]} — nest as deeply as needed.
- "effect": what happens when the conditions are met — e.g. \
  {"decision": "APPROVE"} / {"decision": "DENY"} / \
  {"waives_rule": "<rule_type of the rule being waived>"} / \
  {"max_count": 1, "period_months": 6} etc.

Additional guidance:
- Waivers must name which rule they override in "waives_rule".
- Frequency / time limits must include "max_count" and "period_months" (or period_days/period_weeks).
- Numeric thresholds must include "field", "comparator" (>=, <=, <, >, ==), and "value".
- Enumerated value lists (e.g. red-flag conditions, treatment types) must appear as "in": [...] arrays.
- Every field name must be snake_case and describe the concept from the policy.
- The structure must generalize: the same schema style must work for a completely different policy paragraph.

Output ONLY a single valid JSON object with a top-level "rules" array. \
No markdown fences, no comments, no explanations — pure JSON.
"""


def policy_to_code(text: str) -> str:
    text = text.strip()
    if len(text) < _MIN_CHARS:
        return (
            "Please **paste the full policy text** (at least a few sentences) "
            "and I will convert it into executable Python rule code.\n\n"
            "**Example:** Paste a paragraph from a CMS manual describing what is "
            "covered, excluded, and any waiver or frequency conditions."
        )

    try:
        raw = chat(
            prompt=f"Convert this policy paragraph into complete Python code:\n\n{text}",
            system=_SYSTEM,
        )
    except Exception as e:
        return f"⚠ Code generation failed: {e}"

    python_code = _extract_code(raw)
    json_rules = _generate_json_rules(text)

    preview = text if len(text) <= 220 else text[:220] + "…"

    return (
        f"### Policy → Code\n\n"
        f"> {preview}\n\n"
        f"---\n\n"
        f"**Python**\n\n"
        f"```python\n{python_code}\n```\n\n"
        f"**JSON Rules**\n\n"
        f"```json\n{json_rules}\n```"
    )


def _generate_json_rules(text: str) -> str:
    """Generate a generalized, nested JSON rules document for the policy paragraph."""
    try:
        raw = chat(
            prompt=(
                f"Read every sentence of this paragraph and produce the JSON rules document.\n\n"
                f"{text}"
            ),
            system=_JSON_SYSTEM,
        )
    except Exception as e:
        return json.dumps({"error": f"JSON rule generation failed: {e}"}, indent=2)

    candidate = _extract_json(raw)
    try:
        return json.dumps(json.loads(candidate), indent=2)
    except json.JSONDecodeError:
        return candidate


def _extract_code(response: str) -> str:
    """Strip markdown fences if the LLM wrapped the code anyway."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


def _extract_json(response: str) -> str:
    """Strip markdown fences from JSON output; fall back to first {...} block."""
    m = re.search(r"```(?:json)?\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try to find a raw JSON object in the response
    m2 = re.search(r"(\{.*\})", response, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return response.strip()
