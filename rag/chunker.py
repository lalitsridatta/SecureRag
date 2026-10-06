"""
rag/chunker.py
Split document text into overlapping chunks for retrieval.
"""

from typing import List, Dict


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> List[Dict]:
    """
    Split documents into text chunks.

    Each chunk dict:
        {
            "chunk_id":   "academic_regulations.pdf_p3_c0",
            "text":       "...",
            "source":     "academic_regulations.pdf",
            "page_num":   3,
            "chunk_index": 0,
        }
    """
    chunks = []

    for doc in documents:
        filename = doc["filename"]

        for page in doc["pages"]:
            page_num = page["page_num"]
            text     = page["text"]

            # Split page text into overlapping chunks
            start = 0
            chunk_index = 0
            while start < len(text):
                end  = start + chunk_size
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append({
                        "chunk_id":    f"{filename}_p{page_num}_c{chunk_index}",
                        "text":        chunk_text,
                        "source":      filename,
                        "page_num":    page_num,
                        "chunk_index": chunk_index,
                    })
                    chunk_index += 1

                # Move forward by (chunk_size - overlap)
                step = max(chunk_size - chunk_overlap, 1)
                start += step

    return chunks
