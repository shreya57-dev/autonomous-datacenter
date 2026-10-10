import unittest

from ai.llm import DEFAULT_MODEL, LocalLLM, generate_response


class FakeClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.model: str | None = None
        self.messages: list[dict[str, str]] | None = None

    def chat(self, *, model: str, messages: list[dict[str, str]]) -> object:
        self.model = model
        self.messages = list(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class LocalLLMTests(unittest.TestCase):
    def test_returns_generated_text_and_sends_the_configured_model(self) -> None:
        client = FakeClient({"message": {"content": "  ready  "}})
        model = LocalLLM(client=client)

        self.assertEqual(model.generate_response("Reply with ready."), "ready")
        self.assertEqual(model.model, DEFAULT_MODEL)
        self.assertEqual(client.model, "qwen3:1.7b")
        self.assertEqual(client.messages, [{"role": "user", "content": "Reply with ready."}])

    def test_model_name_is_configurable(self) -> None:
        client = FakeClient({"message": {"content": "ok"}})

        text = generate_response("status", model="qwen3:1.7b-custom", client=client)

        self.assertEqual(text, "ok")
        self.assertEqual(client.model, "qwen3:1.7b-custom")

    def test_empty_prompt_is_rejected(self) -> None:
        client = FakeClient({"message": {"content": "should not be used"}})

        with self.assertRaises(ValueError):
            LocalLLM(client=client).generate_response(" \n\t")
        self.assertIsNone(client.model)

    def test_connection_failure_is_reported(self) -> None:
        client = FakeClient(ConnectionError("connection refused"))

        with self.assertRaises(RuntimeError) as caught:
            LocalLLM(client=client).generate_response("hello")
        self.assertIn("connection refused", str(caught.exception))

    def test_missing_or_empty_response_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            LocalLLM(client=FakeClient({})).generate_response("hello")
        with self.assertRaises(RuntimeError):
            LocalLLM(client=FakeClient({"message": {"content": "  "}})).generate_response("hello")


if __name__ == "__main__":
    unittest.main()
