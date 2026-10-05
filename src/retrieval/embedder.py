"""Turn operational text into dense vectors.

The model is loaded only when an Embedder is constructed. This module
does not build an index and does not call a remote embedding API.
"""

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .knowledge import Chunk

# Short English operational text. Same id for chunk text and queries.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Local sentence embedding model shared by documents and queries."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")
        dimension = self._model.get_embedding_dimension()
        if not isinstance(dimension, int) or dimension <= 0:
            raise RuntimeError(f"model {model_name!r} did not report a vector dimension")
        self.dimension = dimension

    def embed_text(self, text: str) -> np.ndarray:
        """Return one 1-d float32 vector for a non-empty string."""
        _require_text(text)
        return self._encode([text])[0]

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query with the same model as document text."""
        return self.embed_text(query)

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[np.ndarray]:
        """Embed chunk text in the given order. Chunks are not modified."""
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise TypeError("chunks must be a sequence of Chunk")
        if any(not isinstance(chunk, Chunk) for chunk in chunks):
            raise TypeError("chunks must be a sequence of Chunk")
        if not chunks:
            return []
        return self._encode([chunk.text for chunk in chunks])

    def _encode(self, texts: list[str]) -> list[np.ndarray]:
        # One sentence at a time. A padded batch can change later rows by
        # a few ulps, which is enough to fail an exact order check.
        vectors: list[np.ndarray] = []
        for text in texts:
            matrix = self._model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            row = np.asarray(matrix, dtype=np.float32)
            if row.ndim == 2:
                row = row[0]
            if row.shape != (self.dimension,):
                raise RuntimeError(f"model returned shape {row.shape}, expected ({self.dimension},)")
            vectors.append(row.copy())
        return vectors


def _require_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")
