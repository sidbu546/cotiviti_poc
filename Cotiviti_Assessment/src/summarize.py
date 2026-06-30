"""
Phase 5 – Summarize
Thin layer over retrieval: given a topic, produce a plain-language
bullet summary with citations.

Single-chapter: retrieve from that chapter only.
Both chapters:  retrieve from Ch.15 AND Ch.16 separately, then ask
                the LLM for a combined summary with two clearly labelled sections.
"""

from __future__ import annotations

from src.retrieve import retrieve, RetrievedChunk
from src.llm.provider import chat

# Minimum chars to be treated as pasted content rather than a search query
_PASTE_THRESHOLD = 200

# Single LLM call can safely handle ~4 000 chars (~1 200 tokens) of pasted text.
# Larger pastes are split into overlapping chunks, each summarized independently,
# then merged in a final "combine" call.
_CHUNK_SIZE = 4_000
_CHUNK_OVERLAP = 300

_PASTE_SYSTEM = """\
You are a Medicare policy analyst. The user has pasted a policy section for you to summarize.
Summarize it in plain language using clear bullet points.
Focus on: what is covered, what is excluded, any conditions or limits, and key definitions.
Do not add information not present in the pasted text.
If the text contains section numbers or headings, reference them in your bullets (e.g. §30.5).
"""

_COMBINE_SYSTEM = """\
You are a Medicare policy analyst. You have been given several partial summaries of a long policy document.
Merge them into one clean, deduplicated bullet-point summary.
Remove duplicate points, preserve all unique facts, and keep section references where present.
"""

_SINGLE_SYSTEM = """\
You are a Medicare policy analyst. Summarize the provided CMS policy section(s)
in plain language for a claims adjudicator. Use bullet points — 3-6 bullets max.
After each bullet, cite the source in the format [Ch.X §Y.Z].
Do not add information not present in the source text.
"""

_BOTH_SYSTEM = """\
You are a Medicare policy analyst. You have been given excerpts from two chapters
of the CMS Benefit Policy Manual on the same topic:
  • Chapter 15 — what Medicare COVERS (covered services and conditions)
  • Chapter 16 — what Medicare EXCLUDES (general exclusions)

Write a combined summary with two clearly labelled sections:

## Chapter 15 — Coverage
[3-5 bullet points from Ch.15 excerpts, each ending with its citation e.g. [Ch.15 §30.5]]

## Chapter 16 — Exclusions
[3-5 bullet points from Ch.16 excerpts, each ending with its citation e.g. [Ch.16 §120]]

If one chapter has no relevant content, write "No directly relevant content found in this chapter."
Do not invent information beyond what is in the provided excerpts.
"""


def summarize_pasted(text: str) -> str:
    """
    Summarize text pasted directly by the user — no retrieval.
    For large pastes (> _CHUNK_SIZE chars) the text is split into overlapping
    chunks, each summarized independently, then merged in a second LLM call.
    """
    text = text.strip()
    if len(text) <= _CHUNK_SIZE:
        return chat(
            prompt=f"Summarize this policy section:\n\n{text}",
            system=_PASTE_SYSTEM,
        )

    # Split into overlapping chunks
    chunks = _split_text(text, _CHUNK_SIZE, _CHUNK_OVERLAP)
    partial_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        part = chat(
            prompt=f"Summarize this part ({i}/{len(chunks)}) of a policy document:\n\n{chunk}",
            system=_PASTE_SYSTEM,
        )
        partial_summaries.append(f"[Part {i}]\n{part}")

    combined_prompt = (
        "Here are summaries of each part of a long policy document. "
        "Merge them into one final summary:\n\n"
        + "\n\n".join(partial_summaries)
    )
    return chat(prompt=combined_prompt, system=_COMBINE_SYSTEM)


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character chunks, breaking at whitespace."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at the last whitespace within the chunk
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        chunks.append(text[start:end].strip())
        start = end - overlap if end - overlap > start else end
    return chunks


def is_pasted_text(text: str) -> bool:
    """True when the input looks like pasted content rather than a search query."""
    return len(text.strip()) >= _PASTE_THRESHOLD


def summarize_topic(topic: str, top_k: int = 3, chapter_filter: str | None = None) -> str:
    """
    Summarize a topic or pasted section.
    - If topic is long (≥200 chars) → treat as pasted text, summarize directly.
    - chapter_filter='15' or '16'   → retrieve from that chapter only.
    - chapter_filter=None           → combined summary from both chapters.
    """
    if is_pasted_text(topic):
        return summarize_pasted(topic)
    if chapter_filter is None:
        return _summarize_both(topic, top_k=top_k)
    return _summarize_single(topic, top_k=top_k, chapter=chapter_filter)


def _summarize_single(topic: str, top_k: int, chapter: str) -> str:
    chunks = retrieve(topic, top_k=top_k, chapter_filter=chapter)
    if not chunks:
        return f"No relevant policy sections found in Chapter {chapter} for this topic."

    context = _fmt(chunks)
    prompt = f"Topic: {topic}\n\nSource sections:\n{context}\n\nSummarize:"
    summary = chat(prompt=prompt, system=_SINGLE_SYSTEM)
    citations = _cites(chunks)
    return f"{summary}\n\n---\n**Sources:** {citations}"


def _summarize_both(topic: str, top_k: int) -> str:
    ch15 = retrieve(topic, top_k=top_k, chapter_filter="15")
    ch16 = retrieve(topic, top_k=top_k, chapter_filter="16")

    if not ch15 and not ch16:
        return "No relevant policy sections found in either chapter for this topic."

    ch15_block = _fmt(ch15) if ch15 else "No content retrieved from Chapter 15."
    ch16_block = _fmt(ch16) if ch16 else "No content retrieved from Chapter 16."

    prompt = (
        f"Topic: {topic}\n\n"
        f"--- Chapter 15 excerpts (Coverage) ---\n{ch15_block}\n\n"
        f"--- Chapter 16 excerpts (Exclusions) ---\n{ch16_block}\n\n"
        "Write the combined summary:"
    )
    summary = chat(prompt=prompt, system=_BOTH_SYSTEM)

    all_cites = _cites(ch15) + (" | " if ch15 and ch16 else "") + _cites(ch16)
    return f"{summary}\n\n---\n**Sources:** {all_cites}"


def _fmt(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.citation()}]\n{c.text[:1200].strip()}" for c in chunks)


def _cites(chunks: list[RetrievedChunk]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for c in chunks:
        if c.citation() not in seen:
            seen.add(c.citation())
            out.append(c.citation())
    return " | ".join(out)


if __name__ == "__main__":
    print("=== Both chapters ===")
    print(summarize_topic("chiropractic therapy"))
    print("\n=== Ch.15 only ===")
    print(summarize_topic("chiropractic therapy", chapter_filter="15"))
    print("\n=== Ch.16 only ===")
    print(summarize_topic("chiropractic therapy", chapter_filter="16"))
