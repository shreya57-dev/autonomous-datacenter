import unittest
from pathlib import Path

import numpy as np

from retrieval.embedder import Embedder
from retrieval.knowledge import Chunk, load_operational_knowledge
from retrieval.vector_store import VectorStore, index_chunks


KNOWLEDGE = Path(__file__).resolve().parents[1] / "knowledge"
DIMENSION = 4


def chunk(chunk_id: str, text: str, document_id: str = "policy") -> Chunk:
    return Chunk(document_id=document_id, chunk_id=chunk_id, topic=chunk_id, text=text)


def vector(*values: float) -> np.ndarray:
    row = np.zeros(DIMENSION, dtype=np.float32)
    for index, value in enumerate(values):
        row[index] = value
    return row


class VectorStoreTests(unittest.TestCase):
    def test_empty_store_has_no_hits(self) -> None:
        store = VectorStore(DIMENSION)

        self.assertEqual(store.size, 0)
        with self.assertRaises(ValueError) as caught:
            store.search(vector(1, 0), top_k=1)
        self.assertIn("empty", str(caught.exception))

    def test_indexing_sets_size_and_dimension(self) -> None:
        store = VectorStore(DIMENSION)
        chunks = [chunk("policy:1", "one"), chunk("policy:2", "two")]
        store.build(chunks, [vector(1, 0), vector(0, 1)])

        self.assertEqual(store.size, 2)
        self.assertEqual(store.dimension, DIMENSION)
        self.assertEqual(store._index.d, DIMENSION)

    def test_search_orders_by_descending_similarity_and_keeps_identity(self) -> None:
        recovery = chunk("recovery:1", "Move workloads off the failed server.")
        other = chunk("other:1", "Temperature is a comparative reading.")
        store = VectorStore(DIMENSION)
        store.add([recovery, other], [vector(1, 0), vector(0, 1)])

        hits = store.search(vector(1, 0), top_k=2)

        self.assertEqual([hit.chunk.chunk_id for hit in hits], ["recovery:1", "other:1"])
        self.assertEqual(hits[0].chunk.text, recovery.text)
        self.assertGreater(hits[0].score, hits[1].score)

    def test_top_k_limits_results(self) -> None:
        store = VectorStore(DIMENSION)
        store.add(
            [chunk("policy:1", "a"), chunk("policy:2", "b"), chunk("policy:3", "c")],
            [vector(1, 0), vector(0, 1), vector(0, 0, 1)],
        )

        hits = store.search(vector(1, 0), top_k=1)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk.chunk_id, "policy:1")

    def test_top_k_larger_than_the_index_returns_every_chunk(self) -> None:
        store = VectorStore(DIMENSION)
        store.add([chunk("policy:1", "a"), chunk("policy:2", "b")], [vector(1, 0), vector(0, 1)])

        hits = store.search(vector(1, 0), top_k=10)

        self.assertEqual(len(hits), 2)

    def test_wrong_dimension_is_rejected(self) -> None:
        store = VectorStore(DIMENSION)
        with self.assertRaises(ValueError):
            store.add([chunk("policy:1", "a")], [np.zeros(DIMENSION + 1, dtype=np.float32)])
        self.assertEqual(store.size, 0)
        store.add([chunk("policy:1", "a")], [vector(1, 0)])
        with self.assertRaises(ValueError):
            store.search(np.ones(2, dtype=np.float32), top_k=1)

    def test_duplicate_chunk_id_is_rejected(self) -> None:
        store = VectorStore(DIMENSION)
        first = chunk("policy:1", "a")
        store.add([first], [vector(1, 0)])

        with self.assertRaises(ValueError):
            store.add([chunk("policy:1", "again")], [vector(0, 1)])
        self.assertEqual(store.size, 1)
        self.assertEqual(store._chunks[0].text, "a")

    def test_rebuilding_the_same_chunks_searches_the_same_way(self) -> None:
        chunks = [chunk("policy:1", "a"), chunk("policy:2", "b")]
        vectors = [vector(1, 0.2), vector(0.2, 1)]
        query = vector(1, 0)
        first = VectorStore(DIMENSION)
        second = VectorStore(DIMENSION)
        first.build(chunks, vectors)
        second.build(chunks, vectors)
        first.build(chunks, vectors)

        left = first.search(query, top_k=2)
        right = second.search(query, top_k=2)

        self.assertEqual([hit.chunk.chunk_id for hit in left], [hit.chunk.chunk_id for hit in right])
        self.assertTrue(np.allclose([hit.score for hit in left], [hit.score for hit in right], rtol=1e-5, atol=1e-5))


class OperationalSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.embedder = Embedder()
        cls.knowledge = load_operational_knowledge(KNOWLEDGE)
        cls.unrelated = Chunk(
            document_id="unrelated",
            chunk_id="unrelated:1",
            topic="cafeteria",
            text="The office cafeteria serves soup at noon on weekdays.",
        )
        cls.store = VectorStore(cls.embedder.dimension)
        index_chunks(cls.store, cls.embedder, [*cls.knowledge, cls.unrelated])

    def test_queries_rank_operational_knowledge_above_unrelated_text(self) -> None:
        cases = (
            ("What should happen when a server fails?", "server_failure_recovery"),
            ("Can an inactive server receive a workload?", "workload_placement_policy"),
            ("What should happen when resource utilization becomes too high?", "resource_threshold_policy"),
        )
        for query, document_id in cases:
            with self.subTest(query=query):
                hits = self.store.search(self.embedder.embed_query(query), top_k=self.store.size)
                relevant = [hit for hit in hits if hit.chunk.document_id == document_id]
                unrelated = next(hit for hit in hits if hit.chunk.chunk_id == "unrelated:1")
                self.assertTrue(relevant)
                self.assertGreater(relevant[0].score, unrelated.score)
                self.assertLess(
                    hits.index(relevant[0]),
                    hits.index(unrelated),
                )


if __name__ == "__main__":
    unittest.main()
