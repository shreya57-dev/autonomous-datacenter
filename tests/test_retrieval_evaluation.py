import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_retrieval import (
    QUERY_PATH,
    EvalQuery,
    QueryScore,
    first_relevant_rank,
    load_queries,
    mean_reciprocal_rank,
    recall_at,
    summarize,
)
from retrieval.knowledge import OPERATIONAL_DOCUMENTS


def known(query_id: str, document_id: str) -> EvalQuery:
    return EvalQuery(query_id, query_id, document_id, None, True)


def unknown(query_id: str) -> EvalQuery:
    return EvalQuery(query_id, query_id, None, None, False)


def scored(query: EvalQuery, document_ids: list[str], rank: int | None) -> QueryScore:
    return QueryScore(query, tuple(document_ids), tuple(document_ids), tuple(range(len(document_ids))), rank)


class RetrievalEvaluationTests(unittest.TestCase):
    def test_dataset_loads_and_labels_real_documents(self) -> None:
        queries = load_queries(QUERY_PATH)
        known_queries = [query for query in queries if query.known]

        self.assertGreaterEqual(len(known_queries), 15)
        self.assertTrue(any(not query.known for query in queries))
        self.assertEqual(
            {query.expected_document_id for query in known_queries},
            set(OPERATIONAL_DOCUMENTS),
        )
        self.assertEqual(load_queries(QUERY_PATH), queries)

    def test_recall_at_1_and_3(self) -> None:
        ranks = [1, 2, 4, None]

        self.assertEqual(recall_at(ranks, 1), 0.25)
        self.assertEqual(recall_at(ranks, 3), 0.5)

    def test_mrr(self) -> None:
        self.assertAlmostEqual(mean_reciprocal_rank([1, 2, None]), (1.0 + 0.5 + 0.0) / 3)

    def test_missing_document_has_no_rank(self) -> None:
        self.assertIsNone(first_relevant_rank(["capacity_management", "server_maintenance"], "server_failure_recovery"))
        self.assertEqual(recall_at([None], 3), 0.0)
        self.assertEqual(mean_reciprocal_rank([None]), 0.0)

    def test_unknown_queries_are_excluded_from_metrics(self) -> None:
        scores = [
            scored(known("a", "server_failure_recovery"), ["server_failure_recovery"], 1),
            scored(known("b", "capacity_management"), ["server_maintenance", "capacity_management"], 2),
            scored(unknown("zone"), ["server_failure_recovery"], None),
        ]

        summary = summarize(scores)

        self.assertEqual(summary.known_count, 2)
        self.assertEqual(summary.unknown_count, 1)
        self.assertEqual(summary.recall_at_1, 0.5)
        self.assertEqual(summary.recall_at_3, 1.0)
        self.assertAlmostEqual(summary.mrr, (1.0 + 0.5) / 2)

    def test_repeated_summary_matches(self) -> None:
        scores = [
            scored(known("a", "server_failure_recovery"), ["server_failure_recovery"], 1),
            scored(unknown("zone"), ["capacity_management"], None),
        ]

        self.assertEqual(summarize(scores), summarize(scores))

    def test_query_file_is_the_packaged_dataset(self) -> None:
        self.assertEqual(QUERY_PATH, Path(__file__).resolve().parents[1] / "evaluation" / "retrieval_queries.json")


if __name__ == "__main__":
    unittest.main()
