"""Ask a local Ollama model for text.

Other modules call this adapter. They do not talk to the Ollama API.
The model is not contacted until generate_response is called.
"""

from collections.abc import Sequence
from typing import Protocol

import ollama

DEFAULT_MODEL = "qwen3:1.7b"


class ChatClient(Protocol):
    """The small part of the Ollama client this adapter uses."""

    def chat(self, *, model: str, messages: Sequence[dict[str, str]]) -> object:
        """Send one chat request and return the raw client response."""


class LocalLLM:
    """One configured local model."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        client: ChatClient | None = None,
        host: str | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if client is not None and host is not None:
            raise ValueError("pass either a client or a host, not both")
        self.model = model
        if client is not None:
            self._client = client
        elif host is None:
            self._client = ollama.Client()
        else:
            if not isinstance(host, str) or not host.strip():
                raise ValueError("host must be a non-empty string")
            self._client = ollama.Client(host=host)

    def generate_response(self, prompt: str) -> str:
        """Return the model's answer. Failures stay visible."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise RuntimeError(f"ollama request failed: {exc}") from exc
        return _generated_text(response)


def generate_response(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    client: ChatClient | None = None,
    host: str | None = None,
) -> str:
    """Send one prompt to the local model and return its answer."""
    return LocalLLM(model, client=client, host=host).generate_response(prompt)


def _generated_text(response: object) -> str:
    message = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
    if message is None:
        raise RuntimeError("ollama response did not include a message")
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("ollama response did not include generated text")
    return content.strip()
