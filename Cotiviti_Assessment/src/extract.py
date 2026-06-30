"""
Phase 3 – Rule Extraction
LLM translates a retrieved policy clause → PolicyRule JSON.
Validates with Pydantic; re-prompts once on failure.
The LLM never makes the pay/deny decision — only extracts structure.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from src.retrieve import RetrievedChunk, retrieve
from src.schemas import PolicyRule
from src.llm.provider import chat_json

SYSTEM_PROMPT = """\
You are a structured data extractor for CMS Medicare Benefit Policy documents.
Given a policy clause, extract the coverage rule as JSON matching this schema exactly:

{
  "rule_id": "<string, e.g. 'ch15_s240_1'>",
  "description": "<one-sentence plain-English description of the rule>",
  "covered_if": [
    {"field": "<claim attribute name>", "operator": "<one of: ==, !=, >=, <=, in, not_in>", "value": "<value>"}
  ],
  "excluded_if": [
    {"field": "<claim attribute name>", "operator": "<one of: ==, !=, >=, <=, in, not_in>", "value": "<value>"}
  ],
  "source_chapter": "<chapter number as string>",
  "source_section": "<section id as string>",
  "source_clause": "<exact short quote or heading from the clause that justifies the rule>"
}

Rules:
- Use ONLY information from the provided clause text. Do not infer beyond what is stated.
- covered_if lists conditions that ALL must be true for coverage to apply.
- excluded_if lists conditions where ANY match causes denial.
- If a condition list is empty, use [].
- Field names must be snake_case strings (e.g. service_type, therapy_goal, location).
- Values must be strings, numbers, or arrays of strings.
- Respond with ONLY valid JSON, no commentary.
"""


def _build_prompt(chunk: RetrievedChunk) -> str:
    return (
        f"Citation: {chunk.citation()}\n\n"
        f"Policy clause text:\n{chunk.text[:3000]}\n\n"
        "Extract the PolicyRule JSON:"
    )


def extract_rule(chunk: RetrievedChunk, retry: bool = True) -> PolicyRule:
    """
    Extract a PolicyRule from a single policy chunk.
    Re-prompts once on Pydantic validation failure.
    Raises ValueError if both attempts fail.
    """
    prompt = _build_prompt(chunk)

    raw = chat_json(prompt=prompt, system=SYSTEM_PROMPT)
    try:
        return PolicyRule.model_validate(raw)
    except ValidationError as first_err:
        if not retry:
            raise ValueError(
                f"Rule extraction failed validation:\n{first_err}\nRaw: {raw}"
            ) from first_err

        # Re-prompt with error context
        correction_prompt = (
            f"{prompt}\n\n"
            f"Your previous response failed validation with this error:\n{first_err}\n"
            "Fix the JSON and return a valid PolicyRule:"
        )
        raw2 = chat_json(prompt=correction_prompt, system=SYSTEM_PROMPT)
        try:
            return PolicyRule.model_validate(raw2)
        except ValidationError as second_err:
            raise ValueError(
                f"Rule extraction failed twice.\nFirst error: {first_err}\n"
                f"Second error: {second_err}\nLast raw output: {raw2}"
            ) from second_err


def extract_rule_from_query(query: str, chapter_filter: str | None = None) -> PolicyRule:
    """
    Convenience: retrieve the top chunk for a query, then extract a rule from it.
    """
    chunks = retrieve(query, top_k=1, chapter_filter=chapter_filter)
    if not chunks:
        raise ValueError(f"No policy chunks found for query: {query!r}")
    return extract_rule(chunks[0])


if __name__ == "__main__":
    query = "chiropractic maintenance therapy"
    print(f"Extracting rule for: {query!r}\n")
    rule = extract_rule_from_query(query)
    print(rule.model_dump_json(indent=2))
