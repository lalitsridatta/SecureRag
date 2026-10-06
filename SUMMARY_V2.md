# SecureRAG — Session Summary V2
## RAG System + Frontend Implementation

> **Session date:** September 2026  
> **Scope:** RAG pipeline, Streamlit frontend, API debugging, full validation

---

## What Was Done in This Session

This session covered the complete implementation of the RAG (Retrieval-Augmented Generation) system and Streamlit frontend for the SecureRAG university chatbot. The ML/prompt-injection component was already built in a previous session — this session added the information retrieval layer on top.

---

## 1. Project Inspection

Before writing any code, the existing project was inspected:

- **Knowledge base:** 10 PDF documents confirmed in `knowledge_base/`
  - `academic_regulations.pdf`, `admission_guidelines.pdf`, `attendance_policy.pdf`, `course_regulations.pdf`, `department_information.pdf`, `examination_rules.pdf`, `hostel_rules.pdf`, `library_rules.pdf`, `scholarship_information.pdf`, `student_handbook.pdf`
- **`.env` file:** Existing API keys found — Gemini Flash, Gemini Flash Lite, Groq
- **Existing venv:** ML packages installed (PyTorch, transformers, etc.) — no RAG packages
- **Existing models:** `saved_models/`, `saved_models_v2/`, `saved_models_v3/` untouched throughout

---

## 2. Dependency Installation

### First attempt — failed
Tried `langchain==0.3.1` — not available for Python 3.8 (max version is `0.0.27`).  
Tried `chromadb==0.3.29` — failed because it requires `cython==3.0.11` which has no Python 3.8 wheel.

### Solution
Switched to a lighter, Python-3.8-compatible stack:

| Package | Version | Purpose |
|:---|:---|:---|
| `faiss-cpu` | 1.8.0 | Vector store — installs cleanly, no build deps |
| `sentence-transformers` | 2.7.0 | Local embeddings — no API needed |
| `pypdf` | 3.17.4 | PDF text extraction |
| `groq` | 0.11.0 | Groq LLM API client |
| `streamlit` | 1.38.0 | Frontend |
| `python-dotenv` | 1.0.1 | `.env` loading |
| `tiktoken` | 0.7.0 | Tokenisation utility |

All installed successfully into the existing `.venv`.

---

## 3. RAG System Built

### Architecture implemented

```
USER QUERY
    │
    ▼
Query Embedding (all-MiniLM-L6-v2, local, no API)
    │
    ▼
FAISS Vector Search (cosine similarity on L2-normalised vectors)
    │
    ▼
Top-K Relevant Chunks (configurable, default K=5)
    │
    ▼
Context + Query → LLM (Groq primary / Gemini fallback)
    │
    ▼
Grounded Answer + Source Documents
```

### Files created

| File | What it does |
|:---|:---|
| `rag/__init__.py` | Package marker |
| `rag/document_loader.py` | Loads all PDFs from `knowledge_base/`, extracts text page-by-page via `pypdf` |
| `rag/chunker.py` | Splits page text into overlapping chunks: 600 chars, 100 char overlap, sliding window |
| `rag/embeddings.py` | Generates L2-normalised embeddings using `all-MiniLM-L6-v2` locally — no API call |
| `rag/vector_store.py` | Builds FAISS `IndexFlatIP` index, saves/loads from `vector_store/`, searches by cosine similarity |
| `rag/generator.py` | Sends retrieved context + query to LLM with university-specific system prompt |
| `rag/pipeline.py` | Main interface — `RAGPipeline.answer(query)` returns `{answer, sources, retrieved_chunks, model, error}` |

### Index built
- **10 documents** loaded and chunked
- **75 chunks** generated
- FAISS index saved to `vector_store/faiss.index`

### Clean interface for future integration
The `rag/pipeline.py` `answer()` method is designed as the insertion point for the transformer security gate (V3 DistilBERT) and NLI validation — neither component touches the RAG logic:

```python
# Future Stage 2 — transformer gate wraps around this:
result = rag_pipeline.answer(query)

# Future Stage 3 — NLI validates result["answer"] against result["retrieved_chunks"]
```

---

## 4. Frontend Built

**File:** `frontend/app.py`  
**Framework:** Streamlit 1.38.0

### Features
- Chat-style interface using `st.chat_message()` — theme-aware
- Sidebar with knowledge base stats (doc count, chunk count, embedding model)
- Configurable Top-K slider (1–10) and similarity threshold slider
- "Rebuild Knowledge Base" button — re-indexes all documents
- "Clear Chat" button
- Auto-builds index on first launch
- Every answer shows: **answer text**, **source documents**, **expandable retrieved context with similarity scores**
- Model badge below each answer
- Clean error messages — no Python tracebacks shown to user

### First version issue (dark mode)
The first version used raw HTML with hardcoded light colours (`color: #1e293b`). In Streamlit's dark theme (which is the default), this made answer text invisible — white text on white background, as visible in the screenshot.

**Fix:** Replaced all chat message rendering with `st.chat_message()` + `st.write()` which respect Streamlit's active theme. Removed hardcoded text colour CSS. Sources rendered with `st.columns()` instead of HTML divs.

---

## 5. API Debugging

This was the most involved part of the session.

### Issue 1 — `langchain` and `chromadb` not available for Python 3.8
**Root cause:** Python 3.8 end-of-life; newer versions of these packages dropped 3.8 support.  
**Fix:** Switched to `faiss-cpu` + `sentence-transformers` directly — no LangChain needed.

### Issue 2 — Groq `TypeError: __init__() got an unexpected keyword argument 'proxies'`
**Root cause:** `httpx 0.28.1` removed the `proxies` parameter that the `groq` SDK internally passes.  
**Fix:** Downgraded to `httpx==0.27.2` — fully compatible with `groq==0.11.0`.

### Issue 3 — Gemini 404: `gemini-2.5-flash` not available
**Root cause:** The initial `.env` had `FLASH_MODEL=gemini-2.5-flash`. When called via `generateContent` REST endpoint, it returned:
```
"This model models/gemini-2.5-flash is no longer available to new users."
```
**Root cause (deeper):** The API key format `AQ.Ab8RN6L...` is an **Interactions API key**, not a standard Gemini REST key (`AIza...`). The `generateContent` REST endpoint does not accept Interactions API keys.

**Fix applied:** Switched Groq to be the **primary LLM** (confirmed working). Gemini remains as fallback with the `google-generativeai` SDK which handles both key formats. Updated `.env`:
- `FLASH_MODEL=gemini-2.5-flash` (restored — correct model name)
- `GROQ_MODEL=qwen/qwen3.8-27b` (updated — `qwen3-32b` not available on this account)

### Issue 4 — Groq model `qwen/qwen3-32b` not found
**Root cause:** This specific model was not available on the Groq account's tier.  
**Fix:** Ran model discovery, found `qwen/qwen3.8-27b` is available. Updated `.env`.

### API status after all fixes

| Provider | Status | Model used |
|:---|:---|:---|
| Groq | ✅ Working — primary | `qwen/qwen3.8-27b` |
| Gemini REST | ❌ Key type mismatch | — |
| Gemini SDK | ⚠️ Fallback (SDK not installed yet) | `gemini-2.5-flash` |

---

## 6. Test Results

**Script:** `test_rag.py`  
**Result:** ✅ **11/11 tests passed**

| # | Query | Sources retrieved | Result |
|---|:---|:---|:---:|
| 1 | What are the attendance requirements? | attendance_policy.pdf, student_handbook.pdf | ✅ |
| 2 | How do I apply for a transcript? | scholarship, admission, attendance, course PDFs | ✅ |
| 3 | What are the examination rules? | examination_rules.pdf, student_handbook.pdf | ✅ |
| 4 | What facilities are available? | student_handbook.pdf, academic_regulations.pdf, hostel_rules.pdf | ✅ |
| 5 | What is the procedure for course registration? | examination_rules.pdf, course_regulations.pdf, admission_guidelines.pdf | ✅ |
| 6 | What scholarships are available? | scholarship_information.pdf, student_handbook.pdf | ✅ |
| 7 | What are the hostel rules? | hostel_rules.pdf | ✅ |
| 8 | How are grades calculated? | academic_regulations.pdf, course_regulations.pdf | ✅ |
| 9 | What happens if a student fails an exam? | attendance_policy.pdf, examination_rules.pdf | ✅ |
| 10 | What are the library borrowing rules? | library_rules.pdf | ✅ |
| 11 | What is the price of a Mars bar? *(edge case)* | — | ✅ "I could not find this information…" |

---

## 7. Final File Structure Created

```
SecureRag/
│
├── rag/                              ← NEW — RAG pipeline
│   ├── __init__.py
│   ├── document_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── generator.py
│   └── pipeline.py
│
├── frontend/                         ← NEW — Streamlit UI
│   └── app.py
│
├── vector_store/                     ← AUTO-GENERATED — FAISS index
│   ├── faiss.index
│   ├── metadata.pkl
│   └── stats.json
│
├── test_rag.py                       ← NEW — RAG test suite
├── requirements.txt                  ← NEW — all dependencies pinned
├── README.md                         ← NEW — full documentation
│
├── .env                              ← MODIFIED — Groq model updated
│
├── knowledge_base/                   ← UNTOUCHED — 10 PDFs
├── saved_models/                     ← UNTOUCHED — V1 models
├── saved_models_v2/                  ← UNTOUCHED — V2 models
├── saved_models_v3/                  ← UNTOUCHED — V3 DistilBERT
├── ml/                               ← UNTOUCHED — datasets
└── train_guardrails*.py              ← UNTOUCHED — ML training scripts
```

---

## 8. How to Run

```bash
# Activate environment
.venv\Scripts\activate

# Start the application
streamlit run frontend/app.py
```

Opens at **http://localhost:8501**

```bash
# Run RAG tests only
python test_rag.py

# Rebuild index from command line
python -c "from rag.pipeline import RAGPipeline; RAGPipeline().build()"
```

---

## 9. Current Status

| Component | Status |
|:---|:---:|
| PDF document loading | ✅ Working |
| Text chunking | ✅ Working |
| Local embeddings (all-MiniLM-L6-v2) | ✅ Working |
| FAISS vector index | ✅ Working |
| Groq LLM generation | ✅ Working |
| Gemini LLM generation | ⚠️ Requires `google-generativeai` SDK install |
| Streamlit frontend | ✅ Running at localhost:8501 |
| Answer visibility (dark mode fix) | ✅ Fixed |
| Source document display | ✅ Working |
| Retrieved context expander | ✅ Working |
| Auto-index on first launch | ✅ Working |
| 11/11 RAG tests passing | ✅ |

---

## 10. Next Steps (Planned)

1. **Stage 2 — Transformer Security Gate:** Integrate `saved_models_v3/distilbert-base-cased-naturalistic` as a pre-RAG guardrail. Insert before `pipeline.answer()`.
2. **Stage 3 — NLI Response Validation:** Validate that answers are entailed by retrieved context.
3. **Stage 4 — RBAC:** Role-based access control for student/faculty/admin queries.
4. **Optional:** Install `google-generativeai` SDK for Gemini as active fallback.

---

*This summary was generated at the end of the RAG + frontend implementation session.*
