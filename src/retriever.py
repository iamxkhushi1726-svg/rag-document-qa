import faiss
import numpy as np
from typing import List, Tuple

from embedder import embed_chunks, embed_query

class FAISSRetriever:
    """
    Stores document chunk embeddings in a FAISS index.
    Retrieves the most semantically similar chunks for a query.

    This is the core of RAG: instead of searching by keyword,
    we search by meaning using vector similarity.
    """
    def __init__(self):
        self.index: faiss.Index | None = None
        self.chunks: List[str] = []
        self.dimension: int = 384           # all-MiniLM-L6-v2 output size

    def build_index(self, chunks: List[str]) -> None:
        """
        Embed all chunks and store in FAISS index.
        Uses IndexFlatIP (inner product) for cosine similarity
        since embeddings are L2-normalized.
        """
        if not chunks:
            raise ValueError("No chunks provided to build the index.")
        
        self.chunks = chunks
        print(f"Building FAISS index for {len(chunks)} chunks...")

        embeddings = embed_chunks(chunks).astype(np.float32)

        self.dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        print(f"Index built. Total vectors: {self.index.ntotal}")


    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[str, float]]:
        """
        Find the top_k most relevant chunks for a query.
        Returns list of (chunk_text, similarity_score) tuples.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        query_embedding = embed_query(query).astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)

        results: List[Tuple[str, float]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((self.chunks[idx], float(score)))

        return results
    
    def get_context(self, query: str, top_k: int = 4) -> str:
        """Retrieve relevant chunks and join them into a single context string to pass to the LLM."""

        results = self.retrieve(query, top_k=top_k)
        context_parts = [chunk for chunk, _ in results]
        return "\n\n---\n\n".join(context_parts)
