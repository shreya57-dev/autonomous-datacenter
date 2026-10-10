"""Answer a question from retrieved operational knowledge.

The embedder, FAISS store, and local model already exist. This module
only retrieves chunks, places them in a prompt, and returns the model's
answer. It does not apply a relevance cutoff.
"""

from collections.abc import Sequence
from pathlib import Path

from retrieval.embedder import Embedder
from retrieval.knowledge import Chunk, load_operational_knowledge
from retrieval.vector_store import SearchHit, VectorStore, index_chunks

from .llm import LocalLLM

DEFAULT_TOP_K = 3


class KnowledgeRetriever:
    """Turn a question into a query vector and search the existing index."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        if not isinstance(embedder, Embedder):
            raise TypeError("embedder must be an Embedder")
        if not isinstance(store, VectorStore):
            raise TypeError("store must be a VectorStore")
        self._embedder = embedder
        self._store = store

    def retrieve(self, question: str, top_k: int) -> list[SearchHit]:
        """Return the highest-scoring chunks for this question."""
        return self._store.search(self._embedder.embed_query(question), top_k)


class RagPipeline:
    """Retrieve operational chunks, then ask the local model to answer."""

    def __init__(self, retriever: KnowledgeRetriever, llm: LocalLLM, *, top_k: int = DEFAULT_TOP_K) -> None:
        _require_top_k(top_k)
        self._retriever = retriever
        self._llm = llm
        self.top_k = top_k

    def answer_question(self, question: str) -> str:
        """Return the model's answer for a non-empty question."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        try:
            hits = self._retriever.retrieve(question, self.top_k)
        except Exception as exc:
            raise RuntimeError(f"retrieval failed: {exc}") from exc
        if not hits:
            raise RuntimeError("retrieval returned no operational knowledge")
        prompt = build_grounded_prompt(question, hits)
        try:
            return self._llm.generate_response(prompt)
        except Exception as exc:
            raise RuntimeError(f"answer generation failed: {exc}") from exc


def answer_question(question: str, pipeline: RagPipeline) -> str:
    """Run one question through an already built pipeline."""
    if not isinstance(pipeline, RagPipeline):
        raise TypeError("pipeline must be a RagPipeline")
    return pipeline.answer_question(question)


def build_grounded_prompt(question: str, hits: Sequence[SearchHit]) -> str:
    """Place ranked chunks and the question in separate prompt sections."""
    if not hits:
        raise ValueError("prompt requires at least one retrieved chunk")
    blocks: list[str] = []
    for number, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        blocks.append(
            "\n".join(
                (
                    f"[{number}] document_id: {chunk.document_id}",
                    f"chunk_id: {chunk.chunk_id}",
                    f"topic: {chunk.topic}",
                    chunk.text,
                )
            )
        )
    context = "\n\n".join(blocks)
    return (
        "Answer the user's question using the operational knowledge in CONTEXT.\n"
        "Use only that supplied knowledge.\n"
        "\n"
        f"CONTEXT\n{context}\n"
        "\n"
        f"USER QUESTION\n{question.strip()}\n"
    )


def answer_with_local_knowledge(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    knowledge_dir: Path | None = None,
) -> str:
    """Index the policy notes and answer with the local Ollama model.

    This builds the real embedder, FAISS index, and LocalLLM. Unit tests
    should call RagPipeline with substitutes instead.
    """
    directory = knowledge_dir if knowledge_dir is not None else _default_knowledge_dir()
    embedder = Embedder()
    store = VectorStore(embedder.dimension)
    index_chunks(store, embedder, load_operational_knowledge(directory))
    pipeline = RagPipeline(KnowledgeRetriever(embedder, store), LocalLLM(), top_k=top_k)
    return pipeline.answer_question(question)


def _default_knowledge_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "knowledge"


def _require_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an int")
    if top_k < 1:
        raise ValueError("top_k must be positive")
