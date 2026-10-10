import unittest

from ai.rag import RagPipeline, build_grounded_prompt
from retrieval.knowledge import Chunk
from retrieval.vector_store import SearchHit


def make_chunk(chunk_id: str, text: str, *, document_id: str = "server_failure_recovery", topic: str = "recovery") -> Chunk:
    return Chunk(document_id=document_id, chunk_id=chunk_id, topic=topic, text=text)


def make_hit(chunk: Chunk, score: float = 0.5) -> SearchHit:
    return SearchHit(chunk=chunk, score=score)


class FakeRetriever:
    def __init__(self, hits: list[SearchHit] | Exception) -> None:
        self.hits = hits
        self.question: str | None = None
        self.top_k: int | None = None

    def retrieve(self, question: str, top_k: int) -> list[SearchHit]:
        self.question = question
        self.top_k = top_k
        if isinstance(self.hits, Exception):
            raise self.hits
        return self.hits


class FakeLLM:
    def __init__(self, answer: object = "grounded answer") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate_response(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.answer, Exception):
            raise self.answer
        return str(self.answer)


class RagPipelineTests(unittest.TestCase):
    def _pipeline(self, retriever: FakeRetriever, llm: FakeLLM, top_k: int = 3) -> RagPipeline:
        return RagPipeline(retriever, llm, top_k=top_k)  # type: ignore[arg-type]

    def test_question_is_retrieved_and_the_model_answer_is_returned(self) -> None:
        recovery = make_chunk("server_failure_recovery:1", "Identify workloads on the failed server.")
        retriever = FakeRetriever([make_hit(recovery)])
        llm = FakeLLM("Move them only after validation.")
        pipeline = self._pipeline(retriever, llm, top_k=2)

        answer = pipeline.answer_question("What should happen when a server fails?")

        self.assertEqual(answer, "Move them only after validation.")
        self.assertEqual(retriever.question, "What should happen when a server fails?")
        self.assertEqual(retriever.top_k, 2)
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("What should happen when a server fails?", llm.prompts[0])
        self.assertIn("Identify workloads on the failed server.", llm.prompts[0])
        self.assertIn("server_failure_recovery:1", llm.prompts[0])

    def test_multiple_chunks_are_labeled_in_the_prompt(self) -> None:
        first = make_chunk("server_failure_recovery:1", "Mark the failed server unavailable.", topic="detect")
        second = make_chunk(
            "server_failure_recovery:4",
            "Reject the plan when a workload does not fit.",
            topic="validate",
        )
        prompt = build_grounded_prompt(
            "How is recovery checked?",
            [make_hit(first, 0.9), make_hit(second, 0.4)],
        )

        self.assertLess(prompt.index("CONTEXT"), prompt.index("USER QUESTION"))
        self.assertLess(prompt.index("server_failure_recovery:1"), prompt.index("server_failure_recovery:4"))
        self.assertIn("document_id: server_failure_recovery", prompt)
        self.assertIn("topic: detect", prompt)
        self.assertIn("topic: validate", prompt)
        self.assertIn("Mark the failed server unavailable.", prompt)
        self.assertIn("Reject the plan when a workload does not fit.", prompt)
        self.assertIn("How is recovery checked?", prompt.split("USER QUESTION", maxsplit=1)[1])

    def test_empty_question_is_rejected_before_retrieval(self) -> None:
        retriever = FakeRetriever([])
        llm = FakeLLM()

        with self.assertRaises(ValueError):
            self._pipeline(retriever, llm).answer_question("  \n")
        self.assertIsNone(retriever.question)
        self.assertEqual(llm.prompts, [])

    def test_retrieval_failure_does_not_call_the_model(self) -> None:
        retriever = FakeRetriever(RuntimeError("vector store is empty"))
        llm = FakeLLM()

        with self.assertRaises(RuntimeError) as caught:
            self._pipeline(retriever, llm).answer_question("Where did the workloads go?")
        self.assertIn("retrieval failed", str(caught.exception))
        self.assertEqual(llm.prompts, [])

    def test_empty_retrieval_does_not_call_the_model(self) -> None:
        llm = FakeLLM()

        with self.assertRaises(RuntimeError) as caught:
            self._pipeline(FakeRetriever([]), llm).answer_question("Where did the workloads go?")
        self.assertIn("no operational knowledge", str(caught.exception))
        self.assertEqual(llm.prompts, [])

    def test_llm_failure_is_reported(self) -> None:
        retriever = FakeRetriever([make_hit(make_chunk("capacity_management:1", "Available CPU is capacity minus used CPU."))])
        llm = FakeLLM(RuntimeError("connection refused"))

        with self.assertRaises(RuntimeError) as caught:
            self._pipeline(retriever, llm).answer_question("How much CPU is free?")
        self.assertIn("answer generation failed", str(caught.exception))
        self.assertIn("connection refused", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
