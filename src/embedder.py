from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Load the sentences transformer model once and cache it globally.
    all-MiniLM-L6-v2 is fast, small (~90MB), and very effective for semantic similarity tasks.
    """
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_chunks(chunks: List[str]) -> np.ndarray:
    """
    Convert text chunks into dense vector embeddings.
    Each chunk is converted into a 384-dimensional float vector.
    Return numpy array of shape (num_chunks, 384).
    """
    model = get_embedding_model()
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Convert a single query string into an embedding vector."""
    model = get_embedding_model()
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding

