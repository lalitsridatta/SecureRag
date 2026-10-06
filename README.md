# SecureRAG — Secure RAG-Based University Information Assistant

A domain-specific university information assistant built on a Retrieval-Augmented Generation (RAG) pipeline, with a separately implemented prompt-injection detection component.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Current Implementation                │
│                                                         │
│  User Query                                             │
│      │                                                  │
│      ▼                                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │              RAG Pipeline                       │   │
│  │  Query Embedding (all-MiniLM-L6-v2)             │   │
│  │      │                                          │   │
│  │      ▼                                          │   │
│  │  FAISS Vector Search                            │   │
│  │      │                                          │   │
│  │      ▼                                          │   │
│  │  Top-K Relevant Chunks                          │   │
│  │      │                                          │   │
│  │      ▼                                          │   │
│  │  Gemini Flash / Groq (LLM Generation)           │   │
│  │      │                                          │   │
│  │      ▼                                          │   │
│  │  Answer + Sources                               │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  Planned Full Architecture               │
│                                                         │
│  User Query                                             │
│      │                                                  │
│      ▼                                                  │
│  Transformer Prompt-Injection Detector (V3)  ←── TODO  │
│      │                                                  │
│      ▼                                                  │
│  RAG Pipeline  (implemented ✅)                         │
│      │                                                  │
│      ▼                                                  │
│  NLI Response Validator                      ←── TODO  │
│      │                                                  │
│      ▼                                                  │
│  Final Response                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
SecureRag/
│
├── frontend/
│   └── app.py                    ← Streamlit UI
│
├── rag/
│   ├── __init__.py
│   ├── document_loader.py        ← PDF loading
│   ├── chunker.py                ← Text chunking
│   ├── embeddings.py             ← Sentence-transformer embeddings
│   ├── vector_store.py           ← FAISS index build/load/search
│   ├── generator.py              ← Gemini/Groq LLM generation
│   └── pipeline.py               ← Main RAG interface
│
├── knowledge_base/               ← Put PDF documents here
│   ├── academic_regulations.pdf
│   ├── attendance_policy.pdf
│   └── ...
│
├── vector_store/                 ← Auto-generated index (do not edit)
│   ├── faiss.index
│   ├── metadata.pkl
│   └── stats.json
│
├── ml/                           ← Prompt-injection ML (separate component)
├── saved_models_v3/              ← Trained DistilBERT V3 guardrail model
│
├── test_rag.py                   ← RAG test suite
├── .env                          ← API keys (never commit this)
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd SecureRag
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root (never commit it):

```env
# Gemini Flash — Main answer generation
GEMINI_FLASH_API_KEY=your_gemini_api_key_here
FLASH_MODEL=gemini-2.5-flash

# Gemini Flash Lite — Optional (query cleaning, future use)
GEMINI_FLASH_LITE_API_KEY=your_gemini_flash_lite_key_here
FLASH_LITE_MODEL=gemini-2.5-flash-lite

# Groq — Fallback LLM
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3-32b
```

The system tries **Gemini Flash** first and automatically falls back to **Groq** if Gemini fails.

Get API keys:
- Gemini: https://aistudio.google.com/app/apikey
- Groq: https://console.groq.com/keys

---

## Adding Documents

1. Place PDF files in the `knowledge_base/` directory.
2. Rebuild the index (see below).

Supported format: **PDF**

---

## Building / Rebuilding the Vector Index

### Via the UI (recommended)

Click **"🔄 Rebuild Knowledge Base"** in the sidebar. The index builds automatically on first launch.

### Via command line

```bash
python -c "from rag.pipeline import RAGPipeline; p = RAGPipeline(); p.build()"
```

The index is saved to `vector_store/` and reused on subsequent runs. Rebuild whenever you add or change documents.

---

## Running the Application

```bash
streamlit run frontend/app.py
```

Then open http://localhost:8501 in your browser.

---

## Running the RAG Test Suite

```bash
python test_rag.py
```

Tests 11 standard university queries including an edge case (unknown topic). Verifies:
- Relevant chunks are retrieved
- Answers are grounded in retrieved context
- Sources are displayed
- System handles unknown topics gracefully

---

## RAG Architecture Details

| Component | Implementation |
|:---|:---|
| Document loading | `pypdf` — extracts text page-by-page |
| Chunking | Sliding window: 600 chars, 100 char overlap |
| Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers` — local, no API |
| Vector store | FAISS flat inner-product index (cosine similarity on L2-normalised vectors) |
| LLM | Gemini 2.5 Flash (primary) → Groq Qwen3-32B (fallback) |
| Retrieval | Top-K similarity search, configurable K and threshold |

---

## Current Limitations

1. **PDF text extraction only** — scanned/image-based PDFs may not extract cleanly.
2. **Session memory only** — chat history resets on page refresh.
3. **No authentication** — single-user interface; RBAC is a planned future component.
4. **No prompt-injection detection** — the transformer guardrail (V3 DistilBERT) is implemented separately and will be integrated in the next stage.
5. **No NLI response validation** — planned for a future integration stage.
6. **Embedding model downloads on first run** — `all-MiniLM-L6-v2` (~90 MB) is cached after the first download.

---

## Planned Integration Stages

### Stage 2 — Transformer Security Gate
Integrate the trained DistilBERT V3 model (`saved_models_v3/distilbert-base-cased-naturalistic`) as a pre-RAG guardrail. The pipeline interface is already designed to accept this:

```python
# Future integration point in pipeline.py
def answer(self, query: str) -> Dict:
    # Stage 2: transformer gate goes here
    # if classifier.predict(query) == MALICIOUS: return blocked_response
    
    result = self._rag_answer(query)   # current implementation
    
    # Stage 3: NLI validation goes here
    return result
```

### Stage 3 — NLI Response Validation
Validate that generated answers are entailed by the retrieved context, preventing hallucination.

---

## Security Notes

- API keys are read from `.env` — never hardcoded or exposed in the frontend.
- `.env` should be added to `.gitignore`.
- The knowledge base contains only university policy documents — no personal data.
