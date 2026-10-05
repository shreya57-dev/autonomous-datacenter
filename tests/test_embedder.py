import unittest

import numpy as np

from retrieval.embedder import EMBEDDING_MODEL_NAME, Embedder
from retrieval.knowledge import Chunk


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        document_id="capacity_management",
        chunk_id=chunk_id,
        topic="capacity",
        text=text,
    )


class EmbedderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.embedder = Embedder(EMBEDDING_MODEL_NAME)

    def test_text_produces_a_non_empty_vector(self) -> None:
        vector = self.embedder.embed_text("Inactive servers cannot receive workloads.")

        self.assertEqual(vector.ndim, 1)
        self.assertGreater(vector.shape[0], 0)
        self.assertEqual(vector.dtype, np.float32)

    def test_dimensionality_matches_for_text_and_queries(self) -> None:
        document_vector = self.embedder.embed_text("Check CPU and memory before placement.")
        other_vector = self.embedder.embed_text("A threshold breach is a fact, not a move.")
        query_vector = self.embedder.embed_query("Where can a failed server's workloads go?")

        self.assertEqual(document_vector.shape, (self.embedder.dimension,))
        self.assertEqual(other_vector.shape, document_vector.shape)
        self.assertEqual(query_vector.shape, document_vector.shape)

    def test_chunk_embedding_preserves_order(self) -> None:
        first = make_chunk("capacity_management:1", "Available CPU is capacity minus assigned demand.")
        second = make_chunk("capacity_management:2", "Planning CPU uses the larger of current and predicted demand.")
        chunks = [first, second]

        vectors = self.embedder.embed_chunks(chunks)

        self.assertEqual(len(vectors), 2)
        self.assertTrue(np.allclose(vectors[0], self.embedder.embed_text(first.text), rtol=1e-5, atol=1e-5))
        self.assertTrue(np.allclose(vectors[1], self.embedder.embed_text(second.text), rtol=1e-5, atol=1e-5))

    def test_empty_text_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.embedder.embed_text("   \n\t")
        with self.assertRaises(ValueError):
            self.embedder.embed_query("")

    def test_repeated_embedding_matches_within_tolerance(self) -> None:
        text = "Validate the recovery plan on a copy before allocating."
        first = self.embedder.embed_text(text)
        second = self.embedder.embed_text(text)

        self.assertTrue(np.allclose(first, second, rtol=1e-5, atol=1e-5))

    def test_embedding_does_not_mutate_chunks(self) -> None:
        original = make_chunk("server_failure_recovery:1", "Affected workloads stay listed until a valid move.")
        snapshot = Chunk(
            document_id=original.document_id,
            chunk_id=original.chunk_id,
            topic=original.topic,
            text=original.text,
        )

        self.embedder.embed_chunks([original])

        self.assertEqual(original, snapshot)


if __name__ == "__main__":
    unittest.main()
