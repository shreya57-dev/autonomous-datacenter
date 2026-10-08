"""Score semantic retrieval against a fixed query list.

This is a small synthetic benchmark over the project's operational notes.
It does not estimate retrieval quality on a real data center.

Known queries contribute to Recall@1, Recall@3, and MRR. A hit is the
first retrieved chunk whose document_id matches the expected document.
Unknown queries are searched and printed, and they are left out of the
metrics. No similarity cutoff is applied.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval.embedder import Embedder
from retrieval.knowledge import OPERATIONAL_DOCUMENTS, load_operational_knowledge
from retrieval.vector_store import SearchHit, VectorStore, index_chunks

QUERY_PATH = Path(__file__).resolve().parent / "retrieval_queries.json"
KNOWLEDGE_PATH = ROOT / "knowledge"


@dataclass(frozen=True)
class EvalQuery:
    """One benchmark question. Unknown queries have no expected document."""

    query_id: str
    text: str
    expected_document_id: str | None
    expected_chunk_id: str | None
    known: bool


@dataclass(frozen=True)
class QueryScore:
    """Rank of the expected document. rank is None when it was not retrieved."""

    query: EvalQuery
    document_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    scores: tuple[float, ...]
    rank: int | None


@dataclass(frozen=True)
class EvaluationSummary:
    """Metrics over known queries only."""

    known_count: int
    unknown_count: int
    recall_at_1: float
    recall_at_3: float
    mrr: float


def load_queries(path: Path) -> list[EvalQuery]:
    """Load the benchmark file. Known labels must name a real policy document."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not path.is_file():
        raise FileNotFoundError(f"query file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("queries must be a non-empty list")
    queries: list[EvalQuery] = []
    seen: set[str] = set()
    for row in rows:
        query = _query(row)
        if query.query_id in seen:
            raise ValueError(f"duplicate query id {query.query_id!r}")
        seen.add(query.query_id)
        queries.append(query)
    return queries


def first_relevant_rank(document_ids: list[str] | tuple[str, ...], expected_document_id: str) -> int | None:
    """Return the 1-based rank of the expected document, or None if it is absent."""
    if not isinstance(expected_document_id, str) or not expected_document_id.strip():
        raise ValueError("expected_document_id must be a non-empty string")
    for index, document_id in enumerate(document_ids, start=1):
        if document_id == expected_document_id:
            return index
    return None


def recall_at(ranks: list[int | None] | tuple[int | None, ...], k: int) -> float:
    """Fraction of ranks that land inside the first k positions."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an int")
    if k < 1:
        raise ValueError("k must be positive")
    if not ranks:
        raise ValueError("ranks must not be empty")
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None] | tuple[int | None, ...]) -> float:
    """Mean of 1/rank. A missing document contributes 0."""
    if not ranks:
        raise ValueError("ranks must not be empty")
    total = 0.0
    for rank in ranks:
        if rank is not None:
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise ValueError("rank must be a positive int or None")
            total += 1.0 / rank
    return total / len(ranks)


def summarize(scores: list[QueryScore] | tuple[QueryScore, ...]) -> EvaluationSummary:
    """Compute metrics from known queries. Unknown queries are counted only."""
    known = [score for score in scores if score.query.known]
    unknown = [score for score in scores if not score.query.known]
    if not known:
        raise ValueError("evaluation has no known queries")
    ranks = [score.rank for score in known]
    return EvaluationSummary(
        known_count=len(known),
        unknown_count=len(unknown),
        recall_at_1=recall_at(ranks, 1),
        recall_at_3=recall_at(ranks, 3),
        mrr=mean_reciprocal_rank(ranks),
    )


def score_hits(query: EvalQuery, hits: list[SearchHit] | tuple[SearchHit, ...]) -> QueryScore:
    """Record one query's ranking. Unknown queries do not receive a relevance rank."""
    document_ids = tuple(hit.chunk.document_id for hit in hits)
    chunk_ids = tuple(hit.chunk.chunk_id for hit in hits)
    scores = tuple(hit.score for hit in hits)
    rank = None
    if query.known:
        if query.expected_document_id is None:
            raise ValueError(f"known query {query.query_id!r} has no expected document")
        rank = first_relevant_rank(document_ids, query.expected_document_id)
    return QueryScore(query, document_ids, chunk_ids, scores, rank)


def run_evaluation(knowledge_dir: Path, query_path: Path, embedder: Embedder) -> list[QueryScore]:
    """Index the policy chunks and rank every benchmark query."""
    chunks = load_operational_knowledge(knowledge_dir)
    store = VectorStore(embedder.dimension)
    index_chunks(store, embedder, chunks)
    queries = load_queries(query_path)
    scored: list[QueryScore] = []
    for query in queries:
        hits = store.search(embedder.embed_query(query.text), top_k=store.size)
        scored.append(score_hits(query, hits))
    return scored


def format_report(scores: list[QueryScore]) -> str:
    """Render metrics, then each known query and each unknown query."""
    summary = summarize(scores)
    lines = [
        f"known queries: {summary.known_count}",
        f"unknown queries: {summary.unknown_count}",
        f"Recall@1: {summary.recall_at_1:.4f}",
        f"Recall@3: {summary.recall_at_3:.4f}",
        f"MRR: {summary.mrr:.4f}",
        "",
        "known:",
    ]
    for score in scores:
        if not score.query.known:
            continue
        top = score.document_ids[0] if score.document_ids else "none"
        rank = "absent" if score.rank is None else str(score.rank)
        lines.append(
            f"  {score.query.query_id}: rank={rank} top={top} expected={score.query.expected_document_id}"
        )
    lines.append("")
    lines.append("unknown:")
    for score in scores:
        if score.query.known:
            continue
        preview = ", ".join(
            f"{document_id} ({value:.4f})"
            for document_id, value in list(zip(score.document_ids, score.scores))[:3]
        )
        lines.append(f"  {score.query.query_id}: {preview}")
    return "\n".join(lines)


def _query(row: object) -> EvalQuery:
    if not isinstance(row, dict):
        raise ValueError("each query must be an object")
    query_id = _text(row, "query_id")
    text = _text(row, "text")
    known = row.get("known")
    if not isinstance(known, bool):
        raise ValueError(f"query {query_id!r} known must be a bool")
    expected_document_id = row.get("expected_document_id")
    expected_chunk_id = row.get("expected_chunk_id")
    if known:
        if not isinstance(expected_document_id, str) or expected_document_id not in OPERATIONAL_DOCUMENTS:
            raise ValueError(f"query {query_id!r} has an unknown expected document")
        if expected_chunk_id is not None:
            if not isinstance(expected_chunk_id, str) or not expected_chunk_id.startswith(
                expected_document_id + ":"
            ):
                raise ValueError(f"query {query_id!r} chunk id does not belong to its document")
    else:
        if expected_document_id is not None or expected_chunk_id is not None:
            raise ValueError(f"unknown query {query_id!r} must not name an expected document")
    return EvalQuery(query_id, text, expected_document_id, expected_chunk_id, known)


def _text(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def main() -> None:
    scores = run_evaluation(KNOWLEDGE_PATH, QUERY_PATH, Embedder())
    print(format_report(scores))


if __name__ == "__main__":
    main()
