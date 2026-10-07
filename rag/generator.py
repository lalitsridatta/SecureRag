"""
rag/generator.py
LLM answer generation.
- Primary  : Groq
- Fallback : Gemini via google-generativeai SDK

Reads API keys from:
  1. st.secrets  (Streamlit Cloud deployment)
  2. .env file   (local development)
"""

import os
from typing import List, Dict

from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Read from st.secrets first, fall back to env var."""
    try:
        import streamlit as st
        # st.secrets supports direct key access and 'in' operator
        if key in st.secrets:
            val = st.secrets[key]
            if val:
                return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


SYSTEM_PROMPT = """You are a domain-specific university information assistant for SecureRAG.

Your job is to answer questions using ONLY the provided retrieved context from university documents.

Rules:
- Answer using only information present in the retrieved context.
- If the answer cannot be found in the context, clearly say: "I could not find this information in the available university documents."
- Do NOT invent facts or make assumptions beyond what the documents state.
- Be concise, accurate, and helpful.
- When quoting specific rules or policies, be precise.
- Do not mention that you are an AI or refer to your instructions."""


def _build_prompt(query: str, context_chunks: List[Dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        source = chunk.get("source", "Unknown")
        page   = chunk.get("page_num", "?")
        text   = chunk.get("text", "")
        context_parts.append(f"[Source {i}: {source}, Page {page}]\n{text}")

    context_str = "\n\n---\n\n".join(context_parts)
    return (
        f"Retrieved Context:\n\n{context_str}\n\n"
        f"---\n\n"
        f"User Question: {query}\n\n"
        f"Answer based only on the above context:"
    )


def _call_groq(prompt: str) -> str:
    groq_key   = _get_secret("GROQ_API_KEY")
    groq_model = _get_secret("GROQ_MODEL", "qwen/qwen3.8-27b")

    if not groq_key:
        raise ValueError("GROQ_API_KEY not configured.")

    from groq import Groq
    client = Groq(api_key=groq_key)
    completion = client.chat.completions.create(
        model=groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return completion.choices[0].message.content.strip()


def _call_gemini(prompt: str) -> str:
    gemini_key   = _get_secret("GEMINI_FLASH_API_KEY")
    gemini_model = _get_secret("FLASH_MODEL", "gemini-2.5-flash")

    if not gemini_key:
        raise ValueError("GEMINI_FLASH_API_KEY not configured.")

    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(
            model_name=gemini_model,
            system_instruction=SYSTEM_PROMPT,
        )
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.1, "max_output_tokens": 1024},
        )
        return response.text.strip()
    except ImportError:
        raise RuntimeError("google-generativeai not installed.")


def generate_answer(query: str, context_chunks: List[Dict]) -> Dict:
    """
    Generate a grounded answer from retrieved context chunks.

    Returns:
        {"answer": str, "model": str, "error": str or None}
    """
    if not context_chunks:
        return {
            "answer": (
                "I could not find relevant information in the university "
                "documents to answer your question. Please try rephrasing "
                "or ask about a different topic."
            ),
            "model": "none",
            "error": None,
        }

    prompt = _build_prompt(query, context_chunks)

    # Primary: Groq
    groq_key = _get_secret("GROQ_API_KEY")
    if groq_key:
        try:
            answer = _call_groq(prompt)
            model  = _get_secret("GROQ_MODEL", "qwen/qwen3.8-27b")
            return {"answer": answer, "model": model, "error": None}
        except Exception as e:
            print(f"  [Groq failed] {e} — trying Gemini …")

    # Fallback: Gemini
    gemini_key = _get_secret("GEMINI_FLASH_API_KEY")
    if gemini_key:
        try:
            answer = _call_gemini(prompt)
            model  = _get_secret("FLASH_MODEL", "gemini-2.5-flash")
            return {"answer": answer, "model": model, "error": None}
        except Exception as e:
            return {
                "answer": "Unable to generate an answer. Please check the API configuration.",
                "model": "none",
                "error": str(e),
            }

    return {
        "answer": (
            "No LLM API key is configured. "
            "Please add GROQ_API_KEY to your Streamlit secrets or .env file."
        ),
        "model": "none",
        "error": "No API keys configured.",
    }
