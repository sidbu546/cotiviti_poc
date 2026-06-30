"""
Phase 2 – Retrieval with mandatory citations
Given a query string, return top-k chunks with chapter + section citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "cms_policy"


@dataclass
class RetrievedChunk:
    text: str
    chapter: str
    section: str
    source_clause_span: str
    start_page: int
    score: float  # cosine distance (lower = more similar)
    source_file: str = ""  # filename of the originating PDF

    def citation(self) -> str:
        return self.source_clause_span


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def retrieve(
    query: str,
    top_k: int = 5,
    chapter_filter: str | None = None,
    chapter_filters: list[str] | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve top-k policy chunks relevant to the query.

    chapter_filter   – single chapter string, e.g. '15'
    chapter_filters  – list of chapters, e.g. ['15', '16']; uses $or so ALL
                       documents tagged with any of those chapters are included
                       (covers base PDFs + any uploads tagged with the same chapter)
    """
    collection = get_collection()

    # Build ChromaDB where clause
    if chapter_filter:
        where: dict | None = {"chapter": chapter_filter}
    elif chapter_filters and len(chapter_filters) == 1:
        where = {"chapter": chapter_filters[0]}
    elif chapter_filters and len(chapter_filters) > 1:
        where = {"$or": [{"chapter": c} for c in chapter_filters]}
    else:
        where = None

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        chunks.append(
            RetrievedChunk(
                text=doc,
                chapter=meta.get("chapter", ""),
                section=meta.get("section", ""),
                source_clause_span=meta.get("source_clause_span", ""),
                start_page=int(meta.get("start_page", 0)),
                score=float(dist),
                source_file=meta.get("source_file", ""),
            )
        )

    return chunks


RAG_SYSTEM_PROMPT = """\
You are a Medicare policy expert. Answer the user's question using ONLY the provided policy excerpts.
Be direct and specific. After each factual claim, cite the source in the format [Ch.X §Y.Z].
If the excerpts do not contain enough information to answer, say so clearly.
Do not invent information beyond what is in the excerpts.
"""


def rag_answer(query: str, top_k: int = 4) -> str:
    """
    Full RAG: retrieve top-k chunks, then generate a grounded LLM answer with citations.
    Returns a markdown string with the answer followed by the source excerpts.
    """
    from src.llm.provider import chat  # local import to avoid circular at module load

    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return "No relevant policy sections found."

    context = "\n\n".join(
        f"[{c.citation()} | page {c.start_page}]\n{c.text[:1000].strip()}"
        for c in chunks
    )

    prompt = f"Question: {query}\n\nPolicy excerpts:\n{context}\n\nAnswer:"
    answer = chat(prompt=prompt, system=RAG_SYSTEM_PROMPT)

    citations = " | ".join(dict.fromkeys(c.citation() for c in chunks))
    return f"{answer}\n\n---\n**Sources retrieved:** {citations}"


def retrieve_for_display(query: str, top_k: int = 5) -> str:
    """Return raw retrieved chunks formatted for display (no LLM)."""
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return "No relevant policy sections found."

    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"### Result {i} — {chunk.citation()} (page {chunk.start_page})")
        lines.append(chunk.text[:800].strip())
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    query = "chiropractic maintenance therapy coverage"
    print(f"Query: {query}\n")
    results = retrieve(query, top_k=3)
    for r in results:
        print(f"[{r.citation()}] score={r.score:.4f}")
        print(r.text[:300])
        print("---")
