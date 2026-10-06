"""
frontend/app.py
SecureRAG — University Information Assistant
Streamlit frontend — works locally AND on Streamlit Community Cloud.

Run locally:
    streamlit run frontend/app.py

Streamlit Cloud entry point:
    frontend/app.py
"""

import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
# Works whether launched from project root or from frontend/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Knowledge base is always relative to project root
KB_DIR = os.path.join(ROOT, "knowledge_base")

import streamlit as st
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SecureRAG — University Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Light-mode CSS (enforced via config.toml) ─────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .rag-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2563a8 100%);
        color: #ffffff !important;
        padding: 1.2rem 1.8rem;
        border-radius: 10px;
        margin-bottom: 1.4rem;
    }
    .rag-header h1 { color:#ffffff !important; margin:0; font-size:1.6rem; font-weight:700; }
    .rag-header p  { color:#dbeafe !important; margin:0.3rem 0 0; font-size:0.88rem; }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cache the pipeline across sessions ───────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_pipeline():
    from rag.pipeline import RAGPipeline
    return RAGPipeline(knowledge_base_dir=KB_DIR)


# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False


def fmt_source(filename: str) -> str:
    return os.path.splitext(filename)[0].replace("_", " ").title()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 SecureRAG")
    st.divider()

    pipeline = get_pipeline()
    from rag.vector_store import index_exists

    st.markdown("### 📚 Knowledge Base")

    if index_exists():
        stats = pipeline.get_stats()
        col1, col2 = st.columns(2)
        col1.metric("Documents", stats.get("doc_count", "—"))
        col2.metric("Chunks",    stats.get("chunk_count", "—"))
        st.caption("Embedding: `all-MiniLM-L6-v2`")
        st.success("Index ready ✅")
        st.session_state.index_ready = True
    else:
        st.warning("No index yet — click **Build** below.")

    st.divider()
    st.markdown("### ⚙️ Settings")

    top_k = st.slider(
        "Top-K retrieved chunks", 1, 10, 5,
        help="How many document chunks to retrieve per query.",
    )
    score_threshold = st.slider(
        "Similarity threshold", 0.0, 1.0, 0.0, 0.05,
        help="Minimum similarity score to include a chunk.",
    )

    st.divider()
    st.markdown("### 🔧 Actions")

    if st.button("🔄  Build / Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Indexing documents — please wait …"):
            try:
                get_pipeline.clear()
                p       = get_pipeline()
                summary = p.build()
                st.session_state.index_ready = True
                st.success(
                    f"✅ {summary['doc_count']} documents · "
                    f"{summary['chunk_count']} chunks indexed."
                )
                st.rerun()
            except FileNotFoundError as e:
                st.error(f"Knowledge base not found: {e}")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    if st.button("🗑️  Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption(
        "**SecureRAG v1.0** — RAG module  \n"
        "Transformer security layer: *planned*"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="rag-header">'
    '<h1>🎓 Secure RAG-Based University Information Assistant</h1>'
    '<p>Ask questions about university policies, academics, facilities and other '
    'information available in the knowledge base.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Auto-build index on first cloud run ───────────────────────────────────────
if not index_exists() and not st.session_state.index_ready:
    with st.spinner("🔄 Building knowledge base index for the first time …"):
        try:
            summary = pipeline.build()
            st.session_state.index_ready = True
            st.success(
                f"✅ Knowledge base ready — "
                f"{summary['doc_count']} documents, {summary['chunk_count']} chunks."
            )
            st.rerun()
        except FileNotFoundError:
            st.error(
                "❌ `knowledge_base/` folder not found. "
                "Make sure the PDF documents are committed to the repository."
            )
            st.stop()
        except Exception as e:
            st.error(f"❌ Could not build index: {e}")
            st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown(
        "<br><br>"
        "<div style='text-align:center; color:#94a3b8;'>"
        "<p style='font-size:2.5rem;margin-bottom:0.5rem;'>💬</p>"
        "<p style='font-size:1rem;'>Ask a question about university policies, "
        "examinations, attendance, scholarships, hostel, library and more.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.chat_history:

        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])

        else:
            with st.chat_message("assistant"):
                # Answer text — st.markdown respects active theme
                st.markdown(msg["content"])

                # ── Sources ───────────────────────────────────────────────────
                if msg.get("sources"):
                    st.markdown("---")
                    st.markdown("**📂 Sources used:**")
                    for src in msg["sources"]:
                        st.markdown(f"&nbsp;&nbsp;📄 {fmt_source(src)}")

                # ── Expandable retrieved context ──────────────────────────────
                if msg.get("retrieved_chunks"):
                    with st.expander("🔍 View retrieved context", expanded=False):
                        for i, chunk in enumerate(msg["retrieved_chunks"], 1):
                            score = chunk.get("score", 0)
                            st.markdown(
                                f"**{i}. {fmt_source(chunk['source'])} — "
                                f"Page {chunk['page_num']}** "
                                f"&nbsp;*(similarity: {score:.3f})*"
                            )
                            st.info(
                                chunk["text"][:600] +
                                ("…" if len(chunk["text"]) > 600 else "")
                            )
                            if i < len(msg["retrieved_chunks"]):
                                st.divider()

                # ── Model badge ───────────────────────────────────────────────
                if msg.get("model") and msg["model"] != "none":
                    st.caption(f"Generated by `{msg['model']}`")

# ── Query input ───────────────────────────────────────────────────────────────
st.divider()

with st.form(key="query_form", clear_on_submit=True):
    col_q, col_b = st.columns([5, 1])
    with col_q:
        user_input = st.text_input(
            "Your question:",
            placeholder="e.g. What are the attendance requirements?",
            label_visibility="collapsed",
        )
    with col_b:
        submitted = st.form_submit_button("Ask →", use_container_width=True)

# ── Handle submission ─────────────────────────────────────────────────────────
if submitted:
    if not user_input or not user_input.strip():
        st.warning("⚠️ Please type a question before clicking Ask.")
    else:
        query = user_input.strip()
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.spinner("Searching knowledge base …"):
            try:
                result = pipeline.answer(query=query, top_k=top_k)

                # Apply score threshold (post-filter for display)
                filtered = [
                    c for c in result["retrieved_chunks"]
                    if c.get("score", 0) >= score_threshold
                ]
                if not filtered:
                    filtered = result["retrieved_chunks"][:1]

                st.session_state.chat_history.append({
                    "role":             "assistant",
                    "content":          result["answer"],
                    "sources":          result["sources"],
                    "retrieved_chunks": filtered,
                    "model":            result["model"],
                })

            except Exception as e:
                err = str(e)
                if "api" in err.lower() or "key" in err.lower():
                    display = (
                        "⚠️ Unable to connect to the language model. "
                        "Please check the API key configuration in Streamlit Secrets."
                    )
                elif "index" in err.lower():
                    display = (
                        "⚠️ Knowledge base is not ready. "
                        "Please click **Build / Rebuild Knowledge Base** in the sidebar."
                    )
                else:
                    display = f"⚠️ An error occurred: {err}"

                st.session_state.chat_history.append({
                    "role": "assistant", "content": display,
                    "sources": [], "retrieved_chunks": [], "model": "none",
                })

        st.rerun()
