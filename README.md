# Cotiviti Assessment

## Overview

This project is a local-first CMS policy intelligence proof of concept built for the Cotiviti assessment. It ingests Medicare policy PDFs, indexes them into a vector store, and provides a simple chat-based interface for policy retrieval, summarization, chapter comparison, and policy-to-code generation.

The goal of the assessment is to show how unstructured policy content can be turned into a practical reviewer tool that is:

- grounded in source documents
- privacy-conscious and local-first
- useful for both human review and downstream automation

## Demo Video

[Watch the demo video](https://drive.google.com/file/d/15qaWm1H3VKCkVASXl4zT09Dm7gNNpcXI/view?usp=sharing)

## What The Assessment Includes

- PDF ingestion of CMS Benefit Policy Manual chapters
- semantic search over indexed policy sections
- cited question answering using retrieved excerpts
- plain-language topic summarization
- chapter-to-chapter comparison for coverage vs. exclusion analysis
- policy-to-code conversion that generates executable Python and structured JSON rules
- a Gradio UI for interacting with the workflow end to end

## Solution Summary

The application follows a straightforward pipeline:

1. CMS PDFs are loaded from `data/raw/`.
2. Text is extracted and split into section-aware chunks.
3. Chunks are embedded and stored in ChromaDB.
4. User queries are routed into one of four modes:
   - Ask
   - Summarize
   - Compare Chapters
   - Policy to Code
5. A local Ollama-hosted LLM synthesizes final outputs from retrieved policy text.

This keeps the system explainable and grounded because answers always start from retrieved policy excerpts rather than free-form generation.

## Core Features

### 1. Ask

Natural-language Q&A over the indexed CMS policy corpus with source citations.

### 2. Summarize

Summarizes either:

- a short topic query using retrieval
- a pasted policy section directly when the input is long-form text

### 3. Compare Chapters

Compares policy content across chapters, with the default behavior focused on:

- Chapter 15: coverage
- Chapter 16: exclusions

### 4. Policy to Code

Transforms an English policy paragraph into:

- executable Python decision logic
- structured JSON rules for downstream rule engines or auditing workflows

## Architecture At A Glance

```text
Gradio UI (app.py)
    -> mode routing
    -> retrieval / summarization / comparison / policy-to-code
    -> local LLM via Ollama

PDFs (data/raw/)
    -> ingestion (src/ingest.py)
    -> embeddings
    -> ChromaDB vector store (data/chroma/)
```

## Tech Stack

- Python
- Gradio
- ChromaDB
- sentence-transformers
- pdfplumber / pypdf
- Ollama
- pytest

Default local model:

- `qwen2.5:7b`

## Project Structure

```text
Cotiviti_Assessment/
├── app.py
├── README.md
├── requirements.txt
├── src/
│   ├── ingest.py
│   ├── retrieve.py
│   ├── summarize.py
│   ├── compare.py
│   ├── policy_to_code.py
│   └── llm/
│       └── provider.py
├── data/
│   ├── raw/
│   └── chroma/
├── claims/
│   └── synthetic_claims.json
└── eval/
    ├── test_extraction.py
    └── test_retrieval.py
```

## How To Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama

```bash
ollama pull qwen2.5:7b
ollama serve
```

### 3. Ingest the source PDFs

```bash
python -m src.ingest
```

### 4. Launch the app

```bash
python app.py
```

The UI runs locally at `http://127.0.0.1:7860`.

## Environment Variables

You can copy `.env.example` to `.env` and override:

```bash
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_HOST=http://localhost:11434
```

## Testing

Run the fast test suite:

```bash
pytest eval/ -v -m "not llm"
```

Run the full suite:

```bash
pytest eval/ -v
```

## Assessment Notes

- The system is designed to run locally, which helps keep policy text and workflow data on the machine.
- Retrieval outputs include citations so responses can be traced back to source sections.
- The policy-to-code path is intended as a structured translation aid, not an autonomous adjudication engine.
- The repository also includes supporting deliverables such as the report and slide deck used for the assessment submission.

## Deliverables

- Demo video: [Google Drive link](https://drive.google.com/file/d/15qaWm1H3VKCkVASXl4zT09Dm7gNNpcXI/view?usp=sharing)
- Report: `Cotiviti_Report_sub (1).docx`
- Slide decks:
  - `Cotiviti_POC_Deck (1).pptx`
  - `system_design_slide.pptx`
