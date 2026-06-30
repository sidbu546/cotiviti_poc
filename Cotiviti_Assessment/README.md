# CMS Policy Intelligence — System Design

A local-first, privacy-safe POC that ingests CMS Benefit Policy Manual PDFs, answers natural-language questions with cited policy excerpts, and converts English policy paragraphs into executable Python + JSON rules — all powered by a locally-running LLM (no cloud API calls, no PHI leaves the machine).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Gradio UI  (app.py)                          │
│   Sidebar              │         Main Chat Panel                    │
│   - Mode selector      │  Welcome screen  /  Chatbot messages       │
│   - Dataset list       │                                            │
│   - Add Dataset ↑      │  Input bar ──────────────────── Send       │
└──────────┬─────────────┴────────────────────────────────────────────┘
           │  mode dispatch
     ┌─────┴─────────────────────────────────────────────┐
     │                                                   │
  Ask / Summarize / Policy→Code               Compare Chapters
  (retrieve.py / summarize.py /               (compare.py)
   policy_to_code.py)
           │
  ┌────────┴──────────────────────────────────────────────────────┐
  │                        src/  pipeline                         │
  │                                                               │
  │  ingest.py         ──►  ChromaDB  (data/chroma/)             │
  │  retrieve.py       ◄──  ChromaDB                              │
  │  summarize.py      ──►  LLM                                   │
  │  compare.py        ──►  LLM                                   │
  │  policy_to_code.py ──►  LLM  (Python + JSON outputs)         │
  │  llm/provider.py   ──►  Ollama  qwen2.5:7b  (localhost)      │
  └───────────────────────────────────────────────────────────────┘
```

---

## Pipeline Phases

### Phase 1 — Ingest (`src/ingest.py`)

Reads every PDF in `data/raw/`, splits it into section-level chunks, embeds each chunk, and writes to ChromaDB.

| Step | Detail |
|---|---|
| PDF parsing | `pdfplumber` extracts text page-by-page |
| Chunking | Regex splits on CMS section headings (`10`, `20.1`, `30.5.B`, …) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no API) |
| Store | ChromaDB persisted to `data/chroma/` |
| Metadata per chunk | `chapter`, `section`, `source_clause_span`, `start_page`, `source_file` |

Chapter label is auto-detected from the filename (e.g. `bp102c15.pdf` → chapter `15`).
An upload-time override is supported via the UI's **Add Dataset** accordion.

---

### Phase 2 — Retrieve (`src/retrieve.py`)

Cosine-similarity search over the embedded chunks.

```python
retrieve(query, top_k=4, chapter_filter="15")
# → list[RetrievedChunk]  each with .text, .citation(), .score
```

Supports:
- **No filter** — searches all ingested chapters
- **`chapter_filter`** — restricts to one chapter (e.g. `"15"` or `"16"`)
- **`chapter_filters`** — list of chapters, converted to a ChromaDB `$or` clause

---

### Phase 3 — Summarize (`src/summarize.py`)

Two input modes:

| Input type | Behaviour |
|---|---|
| Short query (< 200 chars) | Retrieves top-k chunks from ChromaDB, then LLM summarizes with citations |
| Pasted text (≥ 200 chars) | Summarizes directly; oversized pastes are split into overlapping 4 000-char chunks, each summarized, then merged in a second LLM call |

When no chapter filter is provided, Chapter 15 (coverage) and Chapter 16 (exclusions) are retrieved separately and combined into a two-section response.

---

### Phase 4 — Compare (`src/compare.py`)

Detects explicit chapter references in the query (`chapter 15`, `ch.16`, etc.) and routes to:

| Scenario | Output |
|---|---|
| Two or more chapters | Markdown comparison table: rows = policy aspects, columns = chapters |
| Single chapter, multiple source files | File-vs-file table |
| Single chapter, single file | Bullet-point summary |
| No chapters detected + no uploads | Default: Ch.15 vs Ch.16 table |

The LLM generates the table; the column header format (`Chapter N — filename`) is passed in the system prompt so formatting is consistent across any chapter pair.

---

### Phase 5 — Policy → Code (`src/policy_to_code.py`)

Converts a pasted English policy paragraph into two artifacts via two sequential LLM calls.

**Python output**

System prompt instructs the LLM to produce a complete, executable Python 3 module:
- Top-level `CONSTANT` sets/dicts for all enumerated values (e.g. `RED_FLAG_CONDITIONS`)
- A `def is_covered(claim: dict) -> dict` function returning `{"decision": "APPROVE"|"DENY", "reason": "..."}`
- Every rule in the paragraph: coverage criteria, exclusions, waivers, numeric thresholds, frequency limits
- Inline comments quoting the exact policy clause for each branch
- A `__main__` block with 2–3 example claims (APPROVE, DENY, waiver)

**JSON Rules output**

System prompt instructs the LLM to model every sentence of the paragraph as a rule object. No fixed shallow schema — the structure adapts to whatever the paragraph requires:

```json
{
  "rules": [
    {
      "rule_type": "coverage_criterion",
      "source_text": "...exact sentence from the policy...",
      "conditions": {
        "all_of": [
          { "field": "weeks_of_conservative_treatment", "comparator": ">=", "value": 6 },
          { "field": "conservative_treatment_type", "in": ["physical_therapy", "..."] }
        ]
      },
      "effect": { "decision": "APPROVE" }
    },
    {
      "rule_type": "waiver",
      "source_text": "...exact sentence...",
      "conditions": { "any_of": [{ "field": "red_flag_findings", "in": ["..."] }] },
      "effect": { "waives_rule": "coverage_criterion" }
    },
    {
      "rule_type": "frequency_limit",
      "source_text": "...exact sentence...",
      "effect": { "max_count": 1, "period_months": 6 }
    }
  ]
}
```

Every rule object includes `source_text` (verbatim quote from the paragraph), `conditions` (nested boolean tree), and `effect`.
A regex-based fence stripper (`_extract_json`) handles any markdown the LLM wraps around the output.

---

## LLM Layer (`src/llm/provider.py`)

| Function | Usage |
|---|---|
| `chat(prompt, system)` | Free-form text generation — Python code, JSON rules, RAG answers, summaries, comparisons |
| `chat_json(prompt, system)` | Forced JSON mode via Ollama `format="json"` — used internally when strict JSON output is needed |

**Model:** `qwen2.5:7b` via Ollama (default; overridable with `OLLAMA_MODEL` env var)
**Host:** `http://localhost:11434` (overridable with `OLLAMA_HOST`)
**Temperature:** `0.0` on all calls for deterministic output

The LLM is used exclusively as a translator (natural language → structure / code). It never makes live adjudication decisions.

---

## UI Modes (`app.py`)

| Mode | What it does |
|---|---|
| **Ask** | RAG: retrieve top-4 chunks → LLM answers with section citations |
| **Summarize** | Paste a policy section or type a topic → bullet summary with citations |
| **Compare Chapters** | Auto-detects chapters in the query → Markdown comparison table |
| **Policy → Code** | Paste any English policy paragraph → full Python module + nested JSON rules |

The **Add Dataset** accordion (bottom-left, beside the chat bar) lets you upload additional PDFs and ingest them at runtime. Ingested files appear in the sidebar dataset list.

---

## Directory Structure

```
Cotiviti_Assessment/
├── app.py                     # Gradio UI — layout, routing, event wiring
├── src/
│   ├── ingest.py              # Phase 1 — PDF → ChromaDB
│   ├── retrieve.py            # Phase 2 — semantic search + RAG answer
│   ├── summarize.py           # Phase 3 — topic summary + pasted-text summary
│   ├── compare.py             # Phase 4 — chapter comparison table
│   ├── policy_to_code.py      # Phase 5 — English policy → Python + JSON rules
│   └── llm/
│       └── provider.py        # Ollama wrapper: chat() and chat_json()
├── data/
│   ├── raw/                   # Source PDFs (CMS Benefit Policy Manual chapters)
│   └── chroma/                # ChromaDB vector store (auto-created on first ingest)
├── claims/
│   └── synthetic_claims.json  # Sample claims for ad-hoc reference
├── eval/
│   ├── test_retrieval.py
│   └── test_extraction.py
├── requirements.txt
└── .env.example
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama with the required model
ollama pull qwen2.5:7b
ollama serve          # runs on localhost:11434 by default

# 3. Place CMS PDFs in data/raw/ then ingest
python -m src.ingest

# 4. Launch the UI
python app.py         # opens at http://127.0.0.1:7860
```

Environment variables (copy `.env.example` to `.env` to override):

```
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_HOST=http://localhost:11434
```

Run tests:

```bash
pytest eval/ -v -m "not llm"   # fast tests, no Ollama required
pytest eval/ -v                  # full suite (requires Ollama running)
```

---

## Key Design Decisions

**No cloud API.** All inference runs locally via Ollama. No PHI or policy text leaves the machine.

**LLM as translator only.** The LLM answers questions with retrieved excerpts, summarizes text, compares chapters, and translates policy paragraphs into code/rules. It never makes coverage decisions.

**Copyright guardrail.** Only CMS Benefit Policy Manual PDFs are ingested. NCCI manuals and CPT descriptor text are excluded. Procedure codes are referenced by number only.

**Generalized JSON schema in Policy → Code.** `policy_to_code.py` deliberately avoids a fixed flat schema. The LLM builds a nested boolean condition tree (`all_of`/`any_of`/`not`/`in`/comparators) that adapts to whatever logic each paragraph requires — waivers, frequency limits, numeric thresholds, enumerated value lists — rather than forcing every paragraph into `covered_if`/`excluded_if` pairs.

**Sentence-level coverage.** The JSON rules prompt explicitly requires one rule object per sentence so no clause is skipped or merged. A `source_text` field on every rule ties each object back to the exact policy language it was derived from.
