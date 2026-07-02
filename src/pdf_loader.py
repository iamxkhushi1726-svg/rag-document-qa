import PyPDF2
import io
from typing import List


def load_pdf_text(uploaded_file) -> str:
    """Extract all text from an uploaded PDF file."""
    try:
        pdf_bytes = io.BytesIO(uploaded_file.read())
        reader = PyPDF2.PdfReader(pdf_bytes)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())
        text = "\n\n".join(pages)
        if not text:
            raise ValueError("The PDF contains no extractable text.")
        return text
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {str(e)}")
    

def chunk_text(
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping chunks for embedding.

    chunk_size: number of characters per chunk
    chunk_overlap: how many characters overlap between chunks
                    (overlap preserves context across chunk boundaries)

    Why overlap? Without it, a sentence split across two chunks loses meaning.
    Overlap ensures context is preserved.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks


def get_document_stats(text: str, chunks: List[str]) -> dict:
    """Return stats about the loaded document."""
    return {
        "total_chars": len(text),
        "total_words": len(text.split()),
        "total_chunks": len(chunks),
        "avg_chunk_size": sum(len(c) for c in chunks) // max(len(chunks), 1),
    }