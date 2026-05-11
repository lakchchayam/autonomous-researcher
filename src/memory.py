"""
In-memory vector store for session-based research context.

Uses FAISS to maintain a short-term semantic memory of scraped content
during a research session. Enables the agent to search over documents
it has already downloaded without re-fetching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MemoryChunk:
    """A single chunk of text stored in vector memory."""
    chunk_id: str
    text: str
    source_url: str
    embedding: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)


class VectorMemory:
    """
    Session-scoped vector memory using FAISS.
    
    Stores document chunks with their embeddings during a research
    session. Provides semantic search over accumulated research
    material.
    """
    
    def __init__(self, embedding_dim: int = 1536, chunk_size: int = 800, chunk_overlap: int = 200) -> None:
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")
        
        self._dim = embedding_dim
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine on normalized vectors)
        self._chunks: list[MemoryChunk] = []
        
        logger.info("Vector memory initialized", dim=embedding_dim)
    
    def _split_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks for embedding."""
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + self._chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += self._chunk_size - self._chunk_overlap
        
        return chunks if chunks else [text]
    
    def add_document(self, text: str, source_url: str, embeddings: list[np.ndarray]) -> int:
        """
        Add a document to memory, splitting into chunks.
        
        Args:
            text: Full document text.
            source_url: The URL this text was scraped from.
            embeddings: Pre-computed embeddings for each chunk.
            
        Returns:
            Number of chunks added.
        """
        text_chunks = self._split_text(text)
        
        if len(embeddings) != len(text_chunks):
            logger.warning(
                "Embedding count mismatch, truncating",
                chunks=len(text_chunks),
                embeddings=len(embeddings),
            )
            min_len = min(len(embeddings), len(text_chunks))
            text_chunks = text_chunks[:min_len]
            embeddings = embeddings[:min_len]
        
        for i, (chunk_text, embedding) in enumerate(zip(text_chunks, embeddings)):
            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            chunk = MemoryChunk(
                chunk_id=f"{source_url}::{i}",
                text=chunk_text,
                source_url=source_url,
                embedding=embedding,
                metadata={"chunk_index": i, "total_chunks": len(text_chunks)},
            )
            self._chunks.append(chunk)
            self._index.add(embedding.reshape(1, -1).astype(np.float32))
        
        logger.info("Document added to memory", url=source_url, chunks=len(text_chunks))
        return len(text_chunks)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[MemoryChunk]:
        """
        Semantic search over stored chunks.
        
        Args:
            query_embedding: Embedding vector of the search query.
            top_k: Number of results to return.
            
        Returns:
            List of most similar MemoryChunks, ordered by relevance.
        """
        if self._index.ntotal == 0:
            logger.info("Memory is empty, no results")
            return []
        
        # Normalize query
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        query_vector = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        
        scores, indices = self._index.search(query_vector, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._chunks):
                chunk = self._chunks[idx]
                chunk.metadata["similarity_score"] = float(score)
                results.append(chunk)
        
        logger.info("Memory search completed", results=len(results), top_score=float(scores[0][0]) if len(scores[0]) > 0 else 0)
        return results
    
    @property
    def total_chunks(self) -> int:
        """Total number of chunks in memory."""
        return len(self._chunks)
    
    @property
    def unique_sources(self) -> set[str]:
        """Set of unique source URLs in memory."""
        return {c.source_url for c in self._chunks}
    
    def clear(self) -> None:
        """Clear all stored chunks and reset the index."""
        self._chunks.clear()
        self._index.reset()
        logger.info("Vector memory cleared")
