"""
rag/document_loader.py
Load and extract text from PDF documents in the knowledge base.
"""

import os
from typing import List, Dict
from pypdf import PdfReader


def load_documents(knowledge_base_dir: str = "knowledge_base") -> List[Dict]:
    """
    Load all PDF documents from the knowledge base directory.

    Returns a list of dicts:
        {
            "filename": "academic_regulations.pdf",
            "filepath": "knowledge_base/academic_regulations.pdf",
            "pages": [{"page_num": 1, "text": "..."}],
            "full_text": "...",
            "total_pages": 12,
        }
    """
    if not os.path.exists(knowledge_base_dir):
        raise FileNotFoundError(
            f"Knowledge base directory not found: '{knowledge_base_dir}'. "
            "Please create it and add PDF documents."
        )

    pdf_files = [
        f for f in os.listdir(knowledge_base_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise ValueError(
            f"No PDF documents found in '{knowledge_base_dir}'. "
            "Please add PDF files to the knowledge base."
        )

    documents = []
    for filename in sorted(pdf_files):
        filepath = os.path.join(knowledge_base_dir, filename)
        try:
            reader = PdfReader(filepath)
            pages = []
            full_text_parts = []

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    pages.append({"page_num": page_num, "text": text})
                    full_text_parts.append(text)

            documents.append({
                "filename": filename,
                "filepath": filepath,
                "pages": pages,
                "full_text": "\n\n".join(full_text_parts),
                "total_pages": len(reader.pages),
            })
        except Exception as e:
            print(f"  [WARNING] Could not load '{filename}': {e}")

    return documents
