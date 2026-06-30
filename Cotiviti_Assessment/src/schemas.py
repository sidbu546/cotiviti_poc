from pydantic import BaseModel
from typing import Literal


class Condition(BaseModel):
    field: str
    operator: Literal["==", "!=", ">=", "<=", "in", "not_in"]
    value: str | int | float | list


class PolicyRule(BaseModel):
    rule_id: str
    description: str
    covered_if: list[Condition]
    excluded_if: list[Condition]
    source_chapter: str
    source_section: str
    source_clause: str

    def source_clause_span(self) -> str:
        return f"Ch.{self.source_chapter} §{self.source_section}"


class Claim(BaseModel):
    claim_id: str
    service_type: str
    attributes: dict


class Decision(BaseModel):
    claim_id: str
    outcome: Literal["PAY", "DENY", "REVIEW"]
    rationale: str
    matched_rule_id: str | None
    matched_clause: str | None
