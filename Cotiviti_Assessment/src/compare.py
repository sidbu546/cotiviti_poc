"""
Compare
Retrieve chunks from specified chapters and synthesize a table-format comparison.

Routing logic:
  - Single chapter, one source file   → bullet summary (nothing to compare)
  - Single chapter, multiple files    → file-vs-file table (version comparison)
  - Multiple chapters                 → chapter-vs-chapter table
  - No chapters specified             → default Ch.15 vs Ch.16 table
"""

from __future__ import annotations

import re
from collections import defaultdict

from src.retrieve import retrieve, RetrievedChunk
from src.llm.provider import chat

# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_table_system(col_labels: list[str]) -> str:
    """
    Build a system prompt that instructs the LLM to output a Markdown comparison table.
    col_labels: e.g. ['Chapter 15 — bp102c15.pdf', 'Chapter 16 — bp102c16.pdf']
    """
    header = "| Aspect | " + " | ".join(col_labels) + " |"
    sep    = "|--------|" + "|".join(["---"] * len(col_labels)) + "|"
    return f"""\
You are a Medicare policy analyst. Compare the provided policy excerpts and output a Markdown comparison table.

The table MUST use these exact column headers:
{header}
{sep}

Rules:
- Identify 5-8 distinct policy aspects found across the excerpts \
(e.g. "Covered services", "Exclusion criteria", "Eligibility conditions", \
"Therapy goal requirement", "Documentation needed", "Limitations / caps").
- For each row, write 1-2 sentences per cell describing what that chapter or document says. \
End every cell with the citation in square brackets, e.g. [§220.2.D] or [Ch.15 §220].
- If a chapter or document has no relevant content for a given aspect, write "Not addressed".
- After the table, add one line:
  **Key difference:** <one sentence on the most important divergence between the columns>
- Output ONLY the markdown table followed by the Key difference line — no other text before or after.
- Do not invent information not present in the provided excerpts.
"""


_SINGLE_SYSTEM = """\
You are a Medicare policy analyst. Summarize the provided Chapter {chapter} excerpts on the given topic.
Use clear bullet points — 4-8 bullets. Cite every point [Ch.{chapter} §Y.Z].
Do not add information not present in the excerpts.
"""


# ── Public API ────────────────────────────────────────────────────────────────

def parse_chapter_numbers(text: str) -> list[str]:
    """
    Extract only EXPLICIT chapter references from the query.
    Handles: 'chapter 15', 'ch.15', 'Ch 15', 'chapter15', 'chap 16'.
    Returns deduplicated list, e.g. ['15', '16'].

    No bare-number fallback — that causes false positives like "compare 2 files" → ["2"].
    When no explicit reference is found, returns [] and the caller decides the fallback.
    """
    named = re.findall(
        r'\b(?:chapter|chap(?:ter)?|ch)\.?\s*(\w+)\b',
        text,
        re.IGNORECASE,
    )
    seen: set[str] = set()
    out: list[str] = []
    for n in named:
        # Skip stop-words that aren't chapter IDs
        if n.lower() in {"on", "of", "the", "a", "an", "and", "or", "in", "for", "to"}:
            continue
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def compare_topic(
    query: str,
    top_k_each: int = 3,
    chapters: list[str] | None = None,
) -> tuple[str, str, str]:
    """
    Returns (left_raw, right_raw, llm_synthesis).

    chapters=None        → default Ch.15 vs Ch.16 table
    chapters=['15']      → auto-detects multiple files → file-vs-file table
    chapters=['15','16'] → explicit chapter-vs-chapter table
    """
    if chapters is None:
        chapters = ["15", "16"]

    if len(chapters) == 1:
        return _single_chapter(query, chapters[0], top_k_each)

    return _multi_chapter(query, chapters, top_k_each)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _group_by_source(chunks: list[RetrievedChunk]) -> dict[str, list[RetrievedChunk]]:
    """
    Group chunks by source PDF filename.
    Chunks without source_file (legacy) fall under 'Chapter X (base)'.
    """
    groups: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for c in chunks:
        key = c.source_file if c.source_file else f"Chapter {c.chapter} (base)"
        groups[key].append(c)
    return dict(groups)


def _single_chapter(query: str, chapter: str, top_k: int) -> tuple[str, str, str]:
    """
    Retrieve from one chapter.
    Multiple source files → table comparing them file-vs-file.
    Single source file   → bullet summary.
    """
    chunks = retrieve(query, top_k=top_k * 3, chapter_filter=chapter)

    if not chunks:
        msg = f"No content found tagged as Chapter {chapter} for this topic."
        return msg, "", msg

    groups = _group_by_source(chunks)

    if len(groups) > 1:
        # Build column labels: "filename (Ch.N)"
        labels = [
            f"{fname} (Ch.{groups[fname][0].chapter})"
            for fname in groups
        ]
        return _table_compare(query, groups, labels)

    # Single source: plain bullet summary
    raw = _fmt_plain(chunks)
    system = _SINGLE_SYSTEM.format(chapter=chapter)
    prompt = f"Topic: {query}\n\nChapter {chapter} excerpts:\n{raw}\n\nSummarize:"
    synthesis = chat(prompt=prompt, system=system)
    return raw, "", synthesis


def _multi_chapter(query: str, chapters: list[str], top_k: int) -> tuple[str, str, str]:
    """Retrieve from each chapter and produce a chapter-vs-chapter comparison table."""
    chunk_map: dict[str, list[RetrievedChunk]] = {}
    for ch in chapters:
        chunk_map[ch] = retrieve(query, top_k=top_k, chapter_filter=ch)

    # Build column labels: "Chapter N — filename"  (use most common source_file in that chapter)
    def _label(ch: str) -> str:
        chunks = chunk_map[ch]
        if not chunks:
            return f"Chapter {ch}"
        fname = _most_common_file(chunks)
        return f"Chapter {ch} — {fname}" if fname else f"Chapter {ch}"

    labels = [_label(ch) for ch in chapters]

    # Re-key groups by label (so we can pass to _table_compare)
    groups = {label: chunk_map[ch] for label, ch in zip(labels, chapters)}

    left_raw  = _fmt_with_file(chunk_map[chapters[0]], chapters[0])
    right_raw = _fmt_with_file(chunk_map[chapters[1]], chapters[1]) if len(chapters) > 1 else ""

    context_parts = [
        f"--- {label} ---\n{_fmt_plain(chunk_map[ch])}"
        for label, ch in zip(labels, chapters)
    ]
    prompt = (
        f"Topic: {query}\n\n"
        + "\n\n".join(context_parts)
        + "\n\nWrite the comparison table:"
    )
    synthesis = chat(prompt=prompt, system=_build_table_system(labels))
    return left_raw, right_raw, synthesis


def _table_compare(
    query: str,
    groups: dict[str, list[RetrievedChunk]],
    labels: list[str],
) -> tuple[str, str, str]:
    """Generic table comparison across any set of groups with given column labels."""
    doc_names = list(groups.keys())

    context_parts: list[str] = []
    raw_parts: list[str] = []
    for label, doc_name in zip(labels, doc_names):
        block = _fmt_plain(groups[doc_name])
        context_parts.append(f"--- {label} ---\n{block}")
        raw_parts.append(f"**{label}**\n{block}")

    prompt = (
        f"Topic: {query}\n\n"
        + "\n\n".join(context_parts)
        + "\n\nWrite the comparison table:"
    )
    synthesis = chat(prompt=prompt, system=_build_table_system(labels))

    left_raw  = raw_parts[0] if raw_parts else ""
    right_raw = "\n\n".join(raw_parts[1:]) if len(raw_parts) > 1 else ""
    return left_raw, right_raw, synthesis


# ── Formatting helpers ────────────────────────────────────────────────────────

def _most_common_file(chunks: list[RetrievedChunk]) -> str:
    """Return the most frequent source_file among a list of chunks."""
    counts: dict[str, int] = defaultdict(int)
    for c in chunks:
        if c.source_file:
            counts[c.source_file] += 1
    return max(counts, key=counts.__getitem__) if counts else ""


def _fmt_plain(chunks: list[RetrievedChunk]) -> str:
    """Citation + excerpt block for LLM context."""
    if not chunks:
        return "(no content)"
    return "\n\n".join(
        f"[{c.citation()}]\n{c.text[:800].strip()}" for c in chunks
    )


def _fmt_with_file(chunks: list[RetrievedChunk], chapter: str) -> str:
    """Citation + source filename + excerpt (used for raw display)."""
    if not chunks:
        return f"No content found in Chapter {chapter} for this topic."
    return "\n\n".join(
        f"[{c.citation()} | {c.source_file or 'base'}]\n{c.text[:800].strip()}"
        for c in chunks
    )
