"""
Phase 1 – Ingestion
Parse CMS Benefit Policy Manual PDFs, chunk by section headings,
embed with sentence-transformers, persist to ChromaDB.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import chromadb
import pdfplumber
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "cms_policy"

# Matches numbered section headings like: 10, 20.1, 30.5.B, 100, 200.1
SECTION_RE = re.compile(r"^(\d{1,3}(?:\.\d+)*(?:\.[A-Z])?)\s*[-–—]?\s+\S", re.MULTILINE)


def _chapter_from_filename(path: Path) -> str:
    """
    Derive a chapter label from the filename.
    Handles patterns like: bp102c15, chapter15, ch15, chap_15, c15, c16, etc.
    Falls back to the bare stem if no chapter number is found.
    """
    name = path.stem.lower()
    m = re.search(r'(?:chapter|chap|ch|c)\.?\s*(\d+)', name)
    if m:
        return m.group(1)
    # bare number anywhere in the stem, e.g. "policy_15_v2"
    m2 = re.search(r'\b(\d{1,3})\b', name)
    if m2:
        return m2.group(1)
    return name


def _extract_text_by_page(pdf_path: Path) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages


def _chunk_pages(
    pages: list[tuple[int, str]], chapter: str
) -> list[dict]:
    """
    Concatenate all page text, then split on section heading boundaries.
    Each chunk carries metadata: chapter, section, start_page, source_clause_span.
    """
    # Build full text keeping track of page boundaries
    full_text = ""
    page_offsets: list[tuple[int, int]] = []  # (char_start, page_num)
    for page_num, text in pages:
        start = len(full_text)
        full_text += text + "\n"
        page_offsets.append((start, page_num))

    # Find all section heading positions
    boundaries: list[tuple[int, str]] = []  # (char_pos, section_id)
    for m in SECTION_RE.finditer(full_text):
        boundaries.append((m.start(), m.group(1)))

    # If no headings found, treat the whole document as one chunk
    if not boundaries:
        return [
            {
                "text": full_text.strip(),
                "chapter": chapter,
                "section": "0",
                "source_clause_span": f"Ch.{chapter} full",
                "start_page": 1,
            }
        ]

    chunks: list[dict] = []
    for idx, (char_pos, section_id) in enumerate(boundaries):
        end_pos = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(full_text)
        chunk_text = full_text[char_pos:end_pos].strip()
        if not chunk_text:
            continue

        # Resolve start page
        start_page = 1
        for offset, pnum in page_offsets:
            if offset <= char_pos:
                start_page = pnum

        chunks.append(
            {
                "text": chunk_text,
                "chapter": chapter,
                "section": section_id,
                "source_clause_span": f"Ch.{chapter} §{section_id}",
                "start_page": start_page,
            }
        )

    return chunks


def ingest(pdf_paths: list[Path] | None = None, reset: bool = False) -> int:
    """
    Ingest PDFs into ChromaDB. Returns total chunks stored.
    """
    if pdf_paths is None:
        pdf_paths = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {RAW_DIR}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    if reset and COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks: list[dict] = []
    for pdf_path in pdf_paths:
        chapter = _chapter_from_filename(pdf_path)
        print(f"  Parsing Ch.{chapter}: {pdf_path.name}")
        pages = _extract_text_by_page(pdf_path)
        chunks = _chunk_pages(pages, chapter)
        for chunk in chunks:
            chunk["source_file"] = pdf_path.name  # track which PDF each chunk came from
        print(f"    → {len(chunks)} sections")
        all_chunks.extend(chunks)

    # Upsert in batches
    batch_size = 64
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding & storing"):
        batch = all_chunks[i : i + batch_size]
        collection.upsert(
            ids=[f"ch{c['chapter']}_s{c['section']}_{j}" for j, c in enumerate(batch, start=i)],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "chapter": c["chapter"],
                    "section": c["section"],
                    "source_clause_span": c["source_clause_span"],
                    "start_page": c["start_page"],
                    "source_file": c.get("source_file", ""),
                }
                for c in batch
            ],
        )

    total = collection.count()
    print(f"\nIngestion complete. Collection '{COLLECTION_NAME}' has {total} chunks.")
    return total


def ingest_single(pdf_path: Path, chapter_override: str | None = None) -> tuple[int, int]:
    """
    Ingest one PDF file into the existing collection (no reset).
    Returns (new_chunks_added, total_collection_count).
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    chapter = chapter_override or _chapter_from_filename(pdf_path)
    pages = _extract_text_by_page(pdf_path)
    chunks = _chunk_pages(pages, chapter)

    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[f"upload_{pdf_path.stem}_s{c['section']}_{j}" for j, c in enumerate(batch, start=i)],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "chapter": c["chapter"],
                    "section": c["section"],
                    "source_clause_span": c["source_clause_span"],
                    "start_page": c["start_page"],
                    "source_file": pdf_path.name,
                }
                for c in batch
            ],
        )

    total = collection.count()
    return len(chunks), total


def list_ingested_files() -> list[str]:
    """Return PDF filenames present in data/raw/."""
    return sorted(p.name for p in RAW_DIR.glob("*.pdf"))


if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv
    ingest(reset=reset_flag)
