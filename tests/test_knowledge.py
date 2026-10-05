import unittest
from pathlib import Path

from retrieval.knowledge import (
    OPERATIONAL_DOCUMENTS,
    Chunk,
    chunk_document,
    load_document,
    load_operational_knowledge,
)


KNOWLEDGE = Path(__file__).resolve().parents[1] / "knowledge"


class KnowledgeLoaderTests(unittest.TestCase):
    def test_all_five_documents_load(self) -> None:
        chunks = load_operational_knowledge(KNOWLEDGE)

        loaded = {chunk.document_id for chunk in chunks}
        self.assertEqual(loaded, set(OPERATIONAL_DOCUMENTS))
        self.assertEqual(
            [chunk.document_id for chunk in chunks if chunk.chunk_id.endswith(":1")],
            list(OPERATIONAL_DOCUMENTS),
        )

    def test_document_ids_are_deterministic(self) -> None:
        first = load_document(KNOWLEDGE / "server_failure_recovery.md")
        second = load_document(KNOWLEDGE / "server_failure_recovery.md")

        self.assertEqual(first.document_id, "server_failure_recovery")
        self.assertEqual(first, second)

    def test_chunks_are_non_empty_and_carry_metadata(self) -> None:
        chunks = load_operational_knowledge(KNOWLEDGE)

        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, Chunk)
            self.assertTrue(chunk.document_id.strip())
            self.assertTrue(chunk.chunk_id.strip())
            self.assertTrue(chunk.topic.strip())
            self.assertTrue(chunk.text.strip())

    def test_chunk_ids_are_unique(self) -> None:
        chunks = load_operational_knowledge(KNOWLEDGE)
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))

    def test_repeated_loading_produces_the_same_chunks(self) -> None:
        self.assertEqual(
            load_operational_knowledge(KNOWLEDGE),
            load_operational_knowledge(KNOWLEDGE),
        )

    def test_missing_document_fails(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_document(KNOWLEDGE / "missing_policy.md")

    def test_missing_required_document_fails(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_operational_knowledge(KNOWLEDGE / "does-not-exist")

    def test_empty_document_fails(self) -> None:
        self._assert_malformed("", "empty")

    def test_document_without_sections_fails(self) -> None:
        self._assert_malformed("# Title only\n\nNo procedure.\n", "no sections")

    def test_empty_section_fails(self) -> None:
        self._assert_malformed("# Title\n\n## Empty procedure\n\n", "empty")

    def _assert_malformed(self, text: str, message: str) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            for document_id in OPERATIONAL_DOCUMENTS:
                if document_id == "capacity_management":
                    (directory / f"{document_id}.md").write_text(text, encoding="utf-8")
                else:
                    source = (KNOWLEDGE / f"{document_id}.md").read_text(encoding="utf-8")
                    (directory / f"{document_id}.md").write_text(source, encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_operational_knowledge(directory)
        self.assertIn(message, str(caught.exception).lower())


class ChunkDocumentTests(unittest.TestCase):
    def test_chunk_ids_follow_section_order(self) -> None:
        document = load_document(KNOWLEDGE / "server_maintenance.md")
        chunks = chunk_document(document)

        self.assertEqual(
            [chunk.chunk_id for chunk in chunks],
            [f"server_maintenance:{index}" for index in range(1, len(chunks) + 1)],
        )
        self.assertEqual(chunks[0].topic, "Mark the server inactive before maintenance")


if __name__ == "__main__":
    unittest.main()
