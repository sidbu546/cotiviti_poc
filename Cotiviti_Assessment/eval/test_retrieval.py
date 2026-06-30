"""
Phase 7 – Retrieval tests (no LLM required, always-on).
"""

import pytest
from src.retrieve import retrieve, RetrievedChunk


def test_chiropractic_returns_ch15():
    results = retrieve("chiropractic maintenance therapy", top_k=3)
    assert len(results) > 0
    chapters = {r.chapter for r in results}
    assert "15" in chapters, "Chiropractic query must return Ch.15 results"


def test_chiropractic_top_result_section():
    results = retrieve("chiropractic maintenance therapy", top_k=1)
    assert results[0].section in {"240.1", "30.5", "240.1.5", "240"}, (
        f"Top result section was {results[0].section!r}"
    )


def test_outside_us_returns_ch16():
    results = retrieve("services furnished outside the United States exclusion", top_k=3)
    chapters = {r.chapter for r in results}
    assert "16" in chapters, "Outside-US query must return Ch.16 results"


def test_results_have_citations():
    results = retrieve("physical therapy outpatient", top_k=5)
    for r in results:
        assert r.source_clause_span, "Every result must have a source_clause_span"
        assert r.chapter in {"15", "16"}
        assert r.section


def test_chapter_filter_ch15():
    results = retrieve("surgery", top_k=5, chapter_filter="15")
    for r in results:
        assert r.chapter == "15", f"Chapter filter '15' returned Ch.{r.chapter}"


def test_chapter_filter_ch16():
    results = retrieve("exclusion", top_k=5, chapter_filter="16")
    for r in results:
        assert r.chapter == "16", f"Chapter filter '16' returned Ch.{r.chapter}"


def test_scores_are_floats():
    results = retrieve("coverage", top_k=3)
    for r in results:
        assert isinstance(r.score, float)
