"""Exact similarity search over operational chunks.

FAISS stores L2-normalized vectors in an IndexFlatIP. Inner product on
those vectors is cosine similarity. The Embedder still returns raw
vectors; normalization belongs to this index, not to the model.

FAISS positions are not chunk identity. Position i in the index is
looked up in a parallel list, and the chunk keeps its own chunk_id.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import faiss
import numpy as np

from .embedder import Embedder
from .knowledge import Chunk


@dataclass(frozen=True)
class SearchHit:
    """One ranked chunk. score is cosine similarity after normalization."""

    chunk: Chunk
    score: float


class VectorStore:
    """In-memory exact index. It is not saved to disk."""

    def __init__(self, dimension: int) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError("dimension must be an int")
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[Chunk] = []

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def build(self, chunks: Sequence[Chunk], vectors: Sequence[np.ndarray]) -> None:
        """Replace the index with these chunks and their raw embeddings."""
        self._index = faiss.IndexFlatIP(self.dimension)
        self._chunks = []
        self.add(chunks, vectors)

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[np.ndarray]) -> None:
        """Append raw embeddings. They are normalized before FAISS sees them."""
        chunk_list = _chunks(chunks)
        matrix = _matrix(vectors, self.dimension)
        if len(chunk_list) != matrix.shape[0]:
            raise ValueError("chunks and vectors must have the same length")
        _reject_duplicate_ids(self._chunks, chunk_list)
        if not chunk_list:
            return
        self._index.add(_normalize(matrix))
        self._chunks.extend(chunk_list)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        """Return up to top_k hits, highest cosine similarity first."""
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an int")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if self.size == 0:
            raise ValueError("vector store is empty")
        query = _matrix([query_vector], self.dimension)
        neighbor_count = min(top_k, self.size)
        scores, positions = self._index.search(_normalize(query), neighbor_count)
        hits: list[SearchHit] = []
        for score, position in zip(scores[0], positions[0]):
            if int(position) < 0:
                continue
            hits.append(SearchHit(chunk=self._chunks[int(position)], score=float(score)))
        return hits


def index_chunks(store: VectorStore, embedder: Embedder, chunks: Sequence[Chunk]) -> None:
    """Embed chunks with the existing model and replace the store contents."""
    if not isinstance(store, VectorStore):
        raise TypeError("store must be a VectorStore")
    if not isinstance(embedder, Embedder):
        raise TypeError("embedder must be an Embedder")
    if store.dimension != embedder.dimension:
        raise ValueError(
            f"store dimension {store.dimension} does not match embedder dimension {embedder.dimension}"
        )
    chunk_list = _chunks(chunks)
    store.build(chunk_list, embedder.embed_chunks(chunk_list))


def _chunks(chunks: Sequence[Chunk]) -> list[Chunk]:
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
        raise TypeError("chunks must be a sequence of Chunk")
    if any(not isinstance(chunk, Chunk) for chunk in chunks):
        raise TypeError("chunks must be a sequence of Chunk")
    return list(chunks)


def _matrix(vectors: Sequence[np.ndarray], dimension: int) -> np.ndarray:
    if isinstance(vectors, np.ndarray) or isinstance(vectors, (str, bytes)) or not isinstance(vectors, Sequence):
        raise TypeError("vectors must be a sequence of vectors")
    rows: list[np.ndarray] = []
    for vector in vectors:
        row = np.asarray(vector, dtype=np.float32)
        if row.ndim != 1:
            raise ValueError(f"expected a 1-d vector of dimension {dimension}, got shape {row.shape}")
        if row.shape[0] != dimension:
            raise ValueError(f"expected dimension {dimension}, got {row.shape[0]}")
        rows.append(row)
    if not rows:
        return np.zeros((0, dimension), dtype=np.float32)
    return np.ascontiguousarray(np.vstack(rows), dtype=np.float32)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row. A zero vector cannot be placed on the unit sphere."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalize a zero vector")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _reject_duplicate_ids(existing: Sequence[Chunk], incoming: Sequence[Chunk]) -> None:
    seen = {chunk.chunk_id for chunk in existing}
    for chunk in incoming:
        if chunk.chunk_id in seen:
            raise ValueError(f"duplicate chunk id {chunk.chunk_id!r}")
        seen.add(chunk.chunk_id)
