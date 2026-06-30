# CMS Policy-to-Rule Adjudication POC

A demonstration of how written CMS coverage policy (Medicare Benefit Policy Manual, Chapters 15 & 16) can be converted into machine-checkable rules and used to adjudicate claims — with every decision traceable to the exact policy clause that justified it.

## Architecture

```
PDF ingestion → section-aware chunking → ChromaDB (local embeddings)
                                              ↓
                                    Retrieval + citation
                                              ↓
                              LLM: clause → PolicyRule JSON (Ollama)
                                              ↓
                              Human verification checkpoint
                                              ↓
                        Deterministic Python engine → PAY / DENY / REVIEW
                                              ↓
                                    Gradio UI (4 tabs)
```

**Trust boundary:** The LLM is used only for retrieval and rule extraction. It never makes the pay/deny decision. A deterministic Python engine makes all adjudication decisions.

## Setup

```bash
# 1. Copy env template
cp .env.example .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place PDFs in data/raw/
#    bp102c15.pdf  — Medicare Benefit Policy Manual Ch.15
#    bp102c16.pdf  — Medicare Benefit Policy Manual Ch.16

# 4. Ingest
python -m src.ingest

# 5. Launch UI
python app.py
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) running locally with `llama3.1:8b` or `qwen2.5:7b`

## Running tests

```bash
pytest eval/ -v -m "not llm"   # fast, no LLM required
pytest eval/ -v                  # full suite (requires Ollama)
```
