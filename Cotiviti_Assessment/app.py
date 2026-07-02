"""
CMS Policy Adjudicator — Chat Interface
ChatGPT / Claude-style layout with dynamic dataset upload.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import gradio as gr

from src.retrieve import rag_answer, retrieve_for_display
from src.summarize import summarize_topic
from src.compare import compare_topic, parse_chapter_numbers
from src.policy_to_code import policy_to_code
from src.ingest import ingest_single, list_ingested_files, RAW_DIR

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — dark sidebar + clean chat canvas (Claude-inspired)
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
/* ── Reset ── */
.gradio-container { max-width:100% !important; padding:0 !important; margin:0 !important; }
footer { display:none !important; }
.contain { padding:0 !important; }
* { box-sizing: border-box; }

/* ── Layout shell ── */
/* Top row (sidebar + chat): everything except the input bar */
#app-row {
    height:calc(100vh - 112px) !important; gap:0 !important;
    background:#0d1117; overflow:hidden !important;
}

/* Bottom input outer row — matches sidebar+main layout */
#input-outer {
    height:112px !important; gap:0 !important; padding:0 0 22px 0 !important;
    border-top:1px solid #21262d !important;
}
#input-outer > div, #input-outer > div > div { padding:0 !important; gap:0 !important; }

/* Spacer column (left of input) — mirrors sidebar */
#input-spacer {
    flex:0 0 220px !important; width:220px !important; max-width:220px !important;
    min-width:unset !important; background:#010409 !important;
    border-right:1px solid #21262d !important;
    padding:0 !important;
}

/* Column that holds the actual input row */
#input-col {
    flex:1 !important; background:#161b22 !important; padding:0 !important;
}
#input-col > div, #input-col > div > div { padding:0 !important; gap:0 !important; }

/* ═══════════════════════════════════════
   SIDEBAR  — fixed narrow width
═══════════════════════════════════════ */
#sidebar {
    background:#010409 !important;
    height:100% !important; overflow-y:auto !important;
    padding:0 !important;
    border-right:1px solid #21262d !important;
    flex:0 0 220px !important;
    width:220px !important;
    min-width:unset !important;
    max-width:220px !important;
}

/* Dataset list capped — so upload section always shows below */
#datasets-list {
    max-height:190px; overflow-y:auto;
}
#datasets-list::-webkit-scrollbar { width:3px; }
#datasets-list::-webkit-scrollbar-thumb { background:#30363d; border-radius:3px; }
#sidebar > div { padding:0 !important; }

.sidebar-logo {
    padding:18px 16px 14px;
    border-bottom:1px solid #21262d;
    margin-bottom:4px;
}
.sidebar-logo h2 {
    color:#e6edf3; font-size:1rem; font-weight:700;
    margin:0 0 2px; letter-spacing:-0.2px;
}
.sidebar-logo .logo-sub {
    color:#484f58; font-size:0.72rem;
    display:flex; align-items:center; gap:5px; margin-top:4px;
}
.logo-dot {
    width:6px; height:6px; border-radius:50%;
    background:#3fb950; display:inline-block;
}

.section-label {
    color:#484f58; font-size:0.67rem; font-weight:700;
    text-transform:uppercase; letter-spacing:1px;
    padding:14px 16px 5px; margin:0;
}

.dataset-item {
    display:flex; align-items:center; gap:8px;
    padding:6px 14px; font-size:0.8rem; color:#8b949e;
    cursor:default; transition:background 0.1s;
}
.dataset-item:hover { background:#161b22; }
.dataset-item .ds-name {
    flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    color:#c9d1d9; font-size:0.78rem;
}
.ds-badge {
    font-size:0.6rem; padding:1px 7px;
    border-radius:20px; font-weight:700;
}

.sidebar-divider { border:none; border-top:1px solid #21262d; margin:6px 0; }

#sidebar label { color:#8b949e !important; font-size:0.78rem !important; margin-bottom:4px !important; }
#sidebar input, #sidebar select, #sidebar textarea {
    background:#0d1117 !important; color:#e6edf3 !important;
    border-color:#30363d !important; border-radius:6px !important;
    font-size:0.82rem !important;
}
#sidebar input::placeholder, #sidebar textarea::placeholder { color:#484f58 !important; }
#sidebar button {
    border-radius:6px !important; font-size:0.8rem !important;
    background:#21262d !important; color:#c9d1d9 !important;
    border:1px solid #30363d !important;
}
#sidebar button:hover { background:#30363d !important; color:#e6edf3 !important; }

/* ═══════════════════════════════════════
   MAIN CHAT AREA
═══════════════════════════════════════ */
#main-col {
    background:#0d1117 !important; padding:0 !important;
    height:calc(100vh - 112px) !important;
    overflow:hidden !important; flex:1 !important;
}

/* ── Welcome screen ── */
#welcome-screen {
    width:100% !important;
    height:calc(100vh - 112px) !important;
    overflow-y:auto !important;
    background:#0d1117 !important;
}

/* ── Chatbot — fills full available area ── */
#chatbot {
    background:#0d1117 !important;
    border:none !important; box-shadow:none !important;
    width:100% !important;
    height:calc(100vh - 112px) !important;
    max-height:calc(100vh - 112px) !important;
    overflow:hidden !important;
}
/* Gradio renders a .wrap div inside — make it scrollable and full height */
#chatbot .wrap {
    height:calc(100vh - 112px) !important;
    max-height:calc(100vh - 112px) !important;
    overflow-y:auto !important;
    background:#0d1117 !important;
    border:none !important;
}
#chatbot .message-wrap { max-width:820px; margin:0 auto; padding:8px 28px 0; }

/* ── User bubble ── */
#chatbot [data-testid="user"] > div,
#chatbot .user > div {
    background:#1f6feb !important;
    border-radius:18px 18px 4px 18px !important;
    color:#ffffff !important; border:none !important;
    padding:12px 16px !important;
    box-shadow:0 2px 8px rgba(31,111,235,0.4) !important;
    max-width:75% !important; margin-left:auto !important;
}
#chatbot [data-testid="user"] > div *,
#chatbot .user > div * { color:#ffffff !important; }

/* ── Assistant bubble ── */
#chatbot [data-testid="bot"] > div,
#chatbot .bot > div {
    background:#161b22 !important;
    border-radius:4px 18px 18px 18px !important;
    color:#e6edf3 !important;
    border:1px solid #30363d !important;
    padding:14px 18px !important;
    box-shadow:0 1px 4px rgba(0,0,0,0.3) !important;
    max-width:88% !important;
}

/* All text in assistant bubble */
#chatbot [data-testid="bot"] p,  #chatbot .bot p,
#chatbot [data-testid="bot"] li, #chatbot .bot li,
#chatbot [data-testid="bot"] h1, #chatbot .bot h1,
#chatbot [data-testid="bot"] h2, #chatbot .bot h2,
#chatbot [data-testid="bot"] h3, #chatbot .bot h3,
#chatbot [data-testid="bot"] h4, #chatbot .bot h4,
#chatbot [data-testid="bot"] strong, #chatbot .bot strong,
#chatbot [data-testid="bot"] em,  #chatbot .bot em,
#chatbot [data-testid="bot"] span, #chatbot .bot span,
#chatbot [data-testid="bot"] blockquote, #chatbot .bot blockquote
    { color:#e6edf3 !important; }

#chatbot [data-testid="bot"] a, #chatbot .bot a { color:#58a6ff !important; }

/* Inline code */
#chatbot [data-testid="bot"] code, #chatbot .bot code {
    background:#0d1117 !important; color:#79c0ff !important;
    padding:2px 6px !important; border-radius:4px !important;
    border:1px solid #30363d !important; font-size:0.82em !important;
}

/* Fenced code blocks */
#chatbot pre {
    background:#010409 !important;
    border:1px solid #30363d !important;
    border-radius:8px !important;
    padding:14px 16px !important;
    overflow-x:auto !important; margin:10px 0 !important;
}
#chatbot pre code {
    background:transparent !important; color:#e6edf3 !important;
    border:none !important; padding:0 !important;
    font-size:0.83rem !important; line-height:1.65 !important;
}

/* Blockquote */
#chatbot [data-testid="bot"] blockquote, #chatbot .bot blockquote {
    border-left:3px solid #1f6feb !important;
    margin:8px 0 !important; padding:4px 12px !important;
    background:#0d1117 !important; border-radius:0 6px 6px 0 !important;
}

/* ── Comparison & policy table ── */
#chatbot table {
    width:100%; border-collapse:collapse !important;
    font-size:0.82rem !important; margin:12px 0 !important;
    border-radius:8px !important; overflow:hidden;
    border:1px solid #30363d !important;
}
#chatbot th {
    background:#161b22 !important; color:#58a6ff !important;
    padding:10px 14px !important; text-align:left !important;
    border-bottom:2px solid #1f6feb !important;
    border-right:1px solid #30363d !important;
    font-weight:700 !important; font-size:0.8rem !important;
    text-transform:uppercase; letter-spacing:0.5px;
}
#chatbot td {
    padding:9px 14px !important; border:1px solid #21262d !important;
    vertical-align:top !important; color:#c9d1d9 !important;
    line-height:1.5 !important;
}
#chatbot tr:nth-child(even) td { background:#010409 !important; }
#chatbot tr:nth-child(odd)  td { background:#0d1117 !important; }
#chatbot td:first-child {
    font-weight:700 !important; color:#79c0ff !important;
    background:#0d1117 !important; border-right:2px solid #21262d !important;
    white-space:nowrap;
}

/* ═══════════════════════════════════════
   INPUT BAR  — in #input-col, naturally at bottom
═══════════════════════════════════════ */
#input-row {
    background:#161b22 !important;
    padding:12px 24px !important; gap:10px !important;
    height:90px !important; align-items:center !important;
    width:100% !important;
}
#msg-box textarea {
    border-radius:10px !important;
    border:1px solid #30363d !important;
    background:#0d1117 !important;
    font-size:0.93rem !important; color:#e6edf3 !important;
    padding:11px 16px !important; resize:none !important;
    transition:border-color 0.15s !important; line-height:1.5 !important;
}
#msg-box textarea::placeholder { color:#484f58 !important; }
#msg-box textarea:focus {
    border-color:#1f6feb !important;
    box-shadow:0 0 0 3px rgba(31,111,235,0.15) !important; outline:none !important;
}

/* Mode dropdown in input bar */
#mode-drop { min-width:170px !important; }
#mode-drop select {
    border-radius:8px !important; border:1px solid #30363d !important;
    background:#0d1117 !important; color:#c9d1d9 !important;
    font-size:0.82rem !important; font-weight:600 !important; padding:8px !important;
}

/* Send button */
#send-btn {
    min-width:44px !important; width:44px !important; height:44px !important;
    border-radius:10px !important; font-size:1.1rem !important; padding:0 !important;
    background:#1f6feb !important; color:#ffffff !important;
    border:none !important;
    box-shadow:0 2px 6px rgba(31,111,235,0.4) !important;
    transition:all 0.15s !important;
}
#send-btn:hover { background:#388bfd !important; transform:translateY(-1px) !important; }

/* New chat button */
#new-chat-btn {
    background:#0d1117 !important; border:1px solid #30363d !important;
    color:#8b949e !important; border-radius:6px !important;
    font-size:0.8rem !important; width:calc(100% - 32px) !important;
    margin:8px 16px !important; transition:all 0.12s !important;
}
#new-chat-btn:hover { background:#161b22 !important; color:#e6edf3 !important; border-color:#484f58 !important; }

/* Upload accordion — lives in #input-spacer (bottom-left) */
#upload-accordion {
    background:transparent !important;
    border:none !important; border-radius:0 !important;
    padding:8px 8px 0 !important; margin:0 !important;
    width:100% !important;
}
#upload-accordion > .label-wrap {
    background:#161b22 !important;
    border:1px solid #30363d !important;
    border-radius:6px !important;
    padding:7px 12px !important;
    color:#8b949e !important;
    font-size:0.74rem !important; font-weight:700 !important;
    cursor:pointer !important;
}
#upload-accordion > .label-wrap:hover {
    background:#21262d !important; color:#c9d1d9 !important;
    border-color:#484f58 !important;
}
#upload-accordion > .label-wrap span { color:#8b949e !important; }
/* Panel expands UPWARD using bottom-anchored absolute */
#upload-accordion > div:last-child {
    position:absolute !important;
    bottom:90px !important;          /* above the input bar height */
    left:0 !important;
    width:220px !important;
    background:#161b22 !important;
    border:1px solid #30363d !important;
    border-radius:8px !important;
    padding:12px !important;
    z-index:500 !important;
    box-shadow:0 -4px 24px rgba(0,0,0,0.5) !important;
}

/* Input spacer — needed so the accordion panel has a positioning context */
#input-spacer { position:relative !important; overflow:visible !important; }
#input-spacer > div { overflow:visible !important; }

/* Ingest status */
#ingest-status {
    background:#0d1117 !important; border:1px solid #21262d !important;
    border-radius:8px !important; padding:8px 12px !important; margin-top:8px !important;
}
#ingest-status p, #ingest-status span, #ingest-status li,
#ingest-status strong, #ingest-status code {
    color:#c9d1d9 !important; font-size:0.8rem !important;
}

/* Footer */
#cms-footer {
    background:#010409 !important; border-top:1px solid #21262d !important;
    padding:10px 24px !important;
    color:#484f58 !important; font-size:0.7rem !important; text-align:center;
}
"""

WELCOME_HTML = """
<div style="max-width:700px;margin:52px auto 0;text-align:center;padding:0 28px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">

  <!-- Icon -->
  <div style="width:60px;height:60px;margin:0 auto 20px;border-radius:16px;
              background:linear-gradient(135deg,#1f6feb 0%,#388bfd 100%);
              display:flex;align-items:center;justify-content:center;
              font-size:1.7rem;box-shadow:0 4px 20px rgba(31,111,235,0.4)">⚕</div>

  <!-- Title -->
  <h2 style="color:#e6edf3;font-size:2rem;font-weight:700;margin:0 0 10px;letter-spacing:-0.5px;line-height:1.2">
    CMS Policy Assistant
  </h2>

  <!-- Subtitle -->
  <p style="color:#8b949e;font-size:0.95rem;margin:0 0 10px;line-height:1.7">
    AI-powered Medicare policy analysis grounded in the CMS Benefit Policy Manual.<br>
    Every answer is cited. Runs fully local.
  </p>

  <!-- Model badge -->
  <div style="display:inline-flex;align-items:center;gap:6px;
              background:#161b22;border:1px solid #30363d;border-radius:20px;
              padding:4px 12px;margin-bottom:36px">
    <span style="width:7px;height:7px;border-radius:50%;background:#3fb950;display:inline-block"></span>
    <span style="color:#8b949e;font-size:0.75rem;font-weight:600">qwen2.5:7b · Ollama · ChromaDB</span>
  </div>

  <!-- Feature cards -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;text-align:left">

    <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
                padding:16px 18px;cursor:pointer;transition:border-color 0.15s"
         onmouseover="this.style.borderColor='#1f6feb'" onmouseout="this.style.borderColor='#30363d'">
      <div style="color:#58a6ff;font-size:0.68rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:7px">🔍 Ask</div>
      <div style="color:#c9d1d9;font-size:0.86rem;line-height:1.4">
        Is chiropractic maintenance therapy covered under Medicare?
      </div>
    </div>

    <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
                padding:16px 18px;cursor:pointer;transition:border-color 0.15s"
         onmouseover="this.style.borderColor='#1f6feb'" onmouseout="this.style.borderColor='#30363d'">
      <div style="color:#3fb950;font-size:0.68rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:7px">📋 Summarize</div>
      <div style="color:#c9d1d9;font-size:0.86rem;line-height:1.4">
        Paste any policy paragraph — get a plain-language bullet summary.
      </div>
    </div>

    <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
                padding:16px 18px;cursor:pointer;transition:border-color 0.15s"
         onmouseover="this.style.borderColor='#1f6feb'" onmouseout="this.style.borderColor='#30363d'">
      <div style="color:#d29922;font-size:0.68rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:7px">📊 Compare Chapters</div>
      <div style="color:#c9d1d9;font-size:0.86rem;line-height:1.4">
        Compare Ch.15 vs Ch.16 — or upload docs to compare versions side-by-side.
      </div>
    </div>

    <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
                padding:16px 18px;cursor:pointer;transition:border-color 0.15s"
         onmouseover="this.style.borderColor='#1f6feb'" onmouseout="this.style.borderColor='#30363d'">
      <div style="color:#bc8cff;font-size:0.68rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:7px">🔧 Policy → Code</div>
      <div style="color:#c9d1d9;font-size:0.86rem;line-height:1.4">
        Paste English policy text — get Python rule code + JSON ready for automation.
      </div>
    </div>

  </div>
</div>
"""

SIDEBAR_LOGO = """
<div class="sidebar-logo">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
    <div style="width:32px;height:32px;border-radius:8px;
                background:linear-gradient(135deg,#1f6feb,#388bfd);
                display:flex;align-items:center;justify-content:center;
                font-size:1rem;flex-shrink:0">⚕</div>
    <span style="color:#e6edf3;font-size:0.95rem;font-weight:700;letter-spacing:-0.2px">CMS Assistant</span>
  </div>
  <div class="logo-sub">
    <span class="logo-dot"></span>
    <span>Medicare Policy · Ch.15 &amp; Ch.16</span>
  </div>
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_datasets(uploads: list | None = None) -> str:
    """
    uploads: list of {"chapter": str, "filename": str} from this session.
    Shows chapter tag badge for every file so users know what to reference in Compare.
    """
    files = list_ingested_files()
    upload_ch_map = {u["filename"]: u["chapter"] for u in (uploads or [])}

    if not files:
        return '<div class="dataset-item" style="color:#475569;font-style:italic">No datasets loaded</div>'

    rows = []
    for f in files:
        is_cms = "bp102" in f.lower()
        label  = "CMS" if is_cms else "Upload"
        color  = "#166534" if is_cms else "#1e3a5f"
        bg     = "#bbf7d0" if is_cms else "#bfdbfe"

        # Determine chapter tag for display
        if is_cms:
            ch = "15" if "c15" in f.lower() else "16" if "c16" in f.lower() else ""
        else:
            ch = upload_ch_map.get(f, "")

        ch_badge = (
            f'<span style="font-size:0.68rem;background:#1e3a5f;color:#93c5fd;'
            f'padding:1px 7px;border-radius:8px;margin-left:4px;font-weight:700">Ch.{ch}</span>'
            if ch else ""
        )

        rows.append(
            f'<div class="dataset-item">'
            f'<span class="ds-icon">📄</span>'
            f'<span class="ds-name" title="{f}">{f}</span>'
            f'<span class="ds-badge" style="background:{color};color:{bg}">{label}</span>'
            f'{ch_badge}'
            f'</div>'
        )
    return "".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Chat backend
# ─────────────────────────────────────────────────────────────────────────────

MODES = ["Ask", "Summarize", "Compare Chapters", "Policy → Code"]

def _mode_key(mode: str) -> str:
    return mode.split()[-1]  # "Ask", "Summarize", "Chapters"


def chat_respond(
    message: str,
    history: list,
    mode: str,
    uploads: list,  # session upload registry: [{"chapter": "15", "filename": "doc.pdf"}, ...]
):
    """Returns (history, cleared_msg, welcome_col_update, chatbot_update)."""
    if not message.strip():
        return gr.update(), "", gr.update()

    history = list(history) + [{"role": "user", "content": message}]

    try:
        key = _mode_key(mode)
        if key == "Ask":
            reply = rag_answer(message, top_k=4)
        elif key == "Summarize":
            reply = summarize_topic(message, top_k=4)
        elif key == "Chapters":
            # 1. Explicit chapter references in the query take priority
            detected = parse_chapter_numbers(message)

            if detected:
                chapters = detected
            elif uploads:
                # 2. No explicit chapters → use whatever the user uploaded this session
                seen: dict[str, None] = {}
                for u in uploads:
                    seen[u["chapter"]] = None
                chapters = list(seen.keys())
                # If only one uploaded chapter, compare it against Ch.15 (base)
                if len(chapters) == 1 and chapters[0] not in ("15", "16"):
                    chapters = [chapters[0], "15"]
            else:
                # 3. Nothing specified, nothing uploaded → default Ch.15 vs Ch.16
                chapters = None

            _, _, synthesis = compare_topic(message, top_k_each=3, chapters=chapters)
            reply = synthesis
        elif key == "Code":
            reply = policy_to_code(message)
        else:
            reply = rag_answer(message, top_k=4)
    except Exception as e:
        reply = f"⚠ Error: {e}"

    history = history + [{"role": "assistant", "content": reply}]
    # Hide welcome screen, reveal chatbot on first message
    return gr.update(value=history, visible=True), "", gr.update(visible=False)


def _resolve_upload(file) -> tuple[Path, str]:
    """Return (temp_path, original_filename) from any Gradio file object format."""
    if hasattr(file, "path") and hasattr(file, "orig_name"):   # Gradio 6 FileData
        return Path(file.path), file.orig_name or Path(file.path).name
    if isinstance(file, str):
        p = Path(file)
        return p, p.name
    if isinstance(file, dict):
        p = Path(file.get("path") or file.get("name") or "")
        orig = file.get("orig_name") or file.get("original_name") or p.name
        return p, orig
    if hasattr(file, "name"):
        p = Path(file.name)
        return p, p.name
    raise ValueError(f"Unknown Gradio file type: {type(file).__name__}")


def upload_and_ingest(file, chapter_name: str, current_uploads: list) -> tuple[str, str, list]:
    if file is None:
        return "No file selected.", _render_datasets(current_uploads), current_uploads

    new_uploads = list(current_uploads)
    try:
        src, orig_name = _resolve_upload(file)

        if not src.exists():
            return f"✗ Temp file missing ({src}) — try re-uploading.", _render_datasets(current_uploads), current_uploads

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        dst = RAW_DIR / orig_name
        shutil.copy(str(src), str(dst))

        override = chapter_name.strip() if chapter_name.strip() else None
        added, total = ingest_single(dst, chapter_override=override)

        from src.ingest import _chapter_from_filename
        detected_ch = override if override else _chapter_from_filename(dst)

        # Register in session state
        new_uploads = [u for u in current_uploads if u["filename"] != orig_name]  # dedup
        new_uploads.append({"chapter": detected_ch, "filename": orig_name})

        ch_is_numeric = detected_ch.isdigit()
        if ch_is_numeric:
            tip = (
                f"💡 In **Compare** mode you can now:\n"
                f"- Say *compare chapter {detected_ch}* → compares this file vs the existing Ch.{detected_ch} document\n"
                f"- Say *compare chapter {detected_ch} and chapter 16* → compares this file vs Ch.16"
            )
        else:
            tip = (
                f"⚠️ No chapter number found in filename — tagged as **\"{detected_ch}\"**.\n\n"
                f"Re-upload and type a number (e.g. **15**) in the chapter label field "
                f"so Compare mode can find it."
            )
        msg = (
            f"✅ **{orig_name}** → **Chapter {detected_ch}**\n\n"
            f"*{added} chunks · {total} total in store*\n\n"
            f"{tip}"
        )
    except Exception as e:
        import traceback
        msg = f"✗ Upload failed: {e}\n\n```\n{traceback.format_exc()[-600:]}\n```"

    return msg, _render_datasets(new_uploads), new_uploads


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="CMS Policy Adjudicator") as demo:

    # Session state: tracks uploaded files so Compare can use them without explicit chapter mentions
    upload_state = gr.State([])  # list of {"chapter": str, "filename": str}

    with gr.Row(equal_height=True, elem_id="app-row"):

        # ══════════════════════════════════════════════════════════════════
        # SIDEBAR
        # ══════════════════════════════════════════════════════════════════
        with gr.Column(scale=1, min_width=180, elem_id="sidebar"):

            gr.HTML(SIDEBAR_LOGO)

            new_chat_btn = gr.Button("✏  New Chat", elem_id="new-chat-btn", size="sm")

            gr.HTML('<div class="sidebar-divider"></div>')

            # Mode selector
            gr.HTML('<div class="section-label">Mode</div>')
            mode_select = gr.Dropdown(
                choices=MODES,
                value=MODES[0],
                show_label=False,
                container=False,
                elem_id="mode-drop",
            )

            gr.HTML('<div class="sidebar-divider"></div>')

            # Dataset list
            gr.HTML('<div class="section-label">Loaded Datasets</div>')
            datasets_html = gr.HTML(value=_render_datasets(), elem_id="datasets-list")

        # ══════════════════════════════════════════════════════════════════
        # MAIN CHAT COLUMN
        # ══════════════════════════════════════════════════════════════════
        with gr.Column(scale=6, elem_id="main-col"):

            # ── Welcome screen (shown when chat is empty) ─────────────────
            with gr.Column(visible=True, elem_id="welcome-screen") as welcome_col:
                gr.HTML(WELCOME_HTML)

            # ── Chat history (hidden until first message) ─────────────────
            chatbot = gr.Chatbot(
                value=[],
                height=None,
                show_label=False,
                elem_id="chatbot",
                layout="bubble",
                visible=False,
            )

    # ── Input bar — OUTSIDE app-row so it spans only the chat column ────
    with gr.Row(elem_id="input-outer"):
        with gr.Column(scale=1, min_width=180, elem_id="input-spacer"):
            with gr.Accordion("＋ Add Dataset", open=False, elem_id="upload-accordion"):
                pdf_upload = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    file_count="single",
                    type="filepath",
                )
                chapter_input = gr.Textbox(
                    label="Chapter label (optional)",
                    placeholder="e.g. 15  or  16",
                    container=True,
                )
                ingest_btn = gr.Button("⬆  Ingest", variant="secondary", size="sm")
                ingest_status = gr.Markdown(elem_id="ingest-status")
        with gr.Column(scale=6, elem_id="input-col"):
            with gr.Row(elem_id="input-row"):
                msg_box = gr.Textbox(
                    placeholder="Ask a question · Summarize (paste policy text) · Compare chapters · Policy → Code (paste English policy to generate rules)",
                    show_label=False,
                    container=False,
                    scale=6,
                    lines=1,
                    max_lines=30,
                    elem_id="msg-box",
                )
                send_btn = gr.Button("↑", variant="primary", scale=0, elem_id="send-btn")

    # ── Footer ──────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="cms-footer">
      CMS Benefit Policy Manual &nbsp;·&nbsp; Ch.15 &amp; Ch.16 &nbsp;·&nbsp;
      RAG: ChromaDB + all-MiniLM-L6-v2 &nbsp;·&nbsp;
      LLM: local Ollama (qwen2.5:7b) &nbsp;·&nbsp; All decisions are deterministic — LLM never adjudicates
    </div>
    """)

    # ─────────────────────────────────────────────────────────────────────
    # Wiring
    # ─────────────────────────────────────────────────────────────────────

    # Send on button click
    send_btn.click(
        chat_respond,
        inputs=[msg_box, chatbot, mode_select, upload_state],
        outputs=[chatbot, msg_box, welcome_col],
    )

    # Send on Enter (submit)
    msg_box.submit(
        chat_respond,
        inputs=[msg_box, chatbot, mode_select, upload_state],
        outputs=[chatbot, msg_box, welcome_col],
    )

    # New chat — clear history, show welcome again
    new_chat_btn.click(
        fn=lambda: (gr.update(value=[], visible=False), "", gr.update(visible=True)),
        outputs=[chatbot, msg_box, welcome_col],
    )

    # Ingest PDF — updates status, dataset list, and session state
    ingest_btn.click(
        upload_and_ingest,
        inputs=[pdf_upload, chapter_input, upload_state],
        outputs=[ingest_status, datasets_html, upload_state],
    )


if __name__ == "__main__":
    demo.launch(share=False, show_error=True, css=CSS)
