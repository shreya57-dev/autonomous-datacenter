"""Answer a question from retrieved operational knowledge.

The embedder, FAISS store, and local model already exist. This module
retrieves chunks, decides whether those hits are sufficient, and only
then asks the local model. FAISS still returns nearest neighbors with
no cutoff. Sufficiency is a separate prototype heuristic.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from retrieval.embedder import Embedder
from retrieval.knowledge import load_operational_knowledge
from retrieval.vector_store import SearchHit, VectorStore, index_chunks

from .llm import LocalLLM

DEFAULT_TOP_K = 3

# Cosine similarity of the best retrieved chunk. On this index the weakest
# known operational query still scored 0.456 at rank 1, and the strongest
# unrelated neighbor in the fixed evaluation scored 0.295. 0.38 sits in
# that gap. It is not a guarantee that every unknown question is rejected.
DEFAULT_MIN_SCORE = 0.38

INSUFFICIENT_KNOWLEDGE_RESPONSE = (
    "I don't have sufficient operational knowledge to answer that question."
)


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


@dataclass(frozen=True)
class GroundingPolicy:
    """Decide whether retrieved chunks may support an answer.

    The decision uses the best cosine similarity already stored on
    SearchHit. It does not change FAISS search.
    """

    min_score: float = DEFAULT_MIN_SCORE

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_score", _require_min_score(self.min_score))

    def is_sufficient(self, hits: Sequence[SearchHit]) -> bool:
        """True when the best retrieved chunk meets min_score."""
        if not hits:
            return False
        return max(hit.score for hit in hits) >= self.min_score


class RagPipeline:
    """Retrieve operational chunks, then ask the local model to answer."""

    def __init__(
        self,
        retriever: KnowledgeRetriever,
        llm: LocalLLM,
        *,
        top_k: int = DEFAULT_TOP_K,
        policy: GroundingPolicy | None = None,
    ) -> None:
        _require_top_k(top_k)
        if policy is None:
            policy = GroundingPolicy()
        elif not isinstance(policy, GroundingPolicy):
            raise TypeError("policy must be a GroundingPolicy")
        self._retriever = retriever
        self._llm = llm
        self.top_k = top_k
        self.policy = policy

    def answer_question(self, question: str) -> str:
        """Return a grounded answer, or a refusal when context is too weak."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        try:
            hits = self._retriever.retrieve(question, self.top_k)
        except Exception as exc:
            raise RuntimeError(f"retrieval failed: {exc}") from exc
        if not hits:
            raise RuntimeError("retrieval returned no operational knowledge")
        if not self.policy.is_sufficient(hits):
            return INSUFFICIENT_KNOWLEDGE_RESPONSE
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
    policy: GroundingPolicy | None = None,
) -> str:
    """Index the policy notes and answer with the local Ollama model.

    This builds the real embedder, FAISS index, and LocalLLM. Unit tests
    should call RagPipeline with substitutes instead.
    """
    directory = knowledge_dir if knowledge_dir is not None else _default_knowledge_dir()
    embedder = Embedder()
    store = VectorStore(embedder.dimension)
    index_chunks(store, embedder, load_operational_knowledge(directory))
    pipeline = RagPipeline(
        KnowledgeRetriever(embedder, store),
        LocalLLM(),
        top_k=top_k,
        policy=policy,
    )
    return pipeline.answer_question(question)


def _default_knowledge_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "knowledge"


def _require_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an int")
    if top_k < 1:
        raise ValueError("top_k must be positive")


def _require_min_score(min_score: float) -> float:
    if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
        raise TypeError("min_score must be a real number")
    value = float(min_score)
    if value != value or value < -1.0 or value > 1.0:
        raise ValueError("min_score must be between -1 and 1")
    return value
