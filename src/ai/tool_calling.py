"""One controlled tool call, then a final answer.

The model returns a single JSON object. A tool name is accepted only when
it is already in the registered map. The data center is supplied by the
caller and is never an argument the model can pass.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass

from models.datacenter import DataCenter
from tools.system_tools import ToolSpec, tool_registry

from .llm import LocalLLM


@dataclass(frozen=True)
class ToolSelection:
    """One requested tool call. arguments contain only validated strings."""

    name: str
    arguments: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class FinalAnswer:
    """A natural-language answer that does not call a tool."""

    text: str


class ToolCaller:
    """Ask the model once, run at most one registered tool, then answer."""

    def __init__(
        self,
        llm: LocalLLM,
        datacenter: DataCenter,
        tools: Sequence[ToolSpec] | None = None,
    ) -> None:
        if not hasattr(llm, "generate_response"):
            raise TypeError("llm must provide generate_response")
        if not isinstance(datacenter, DataCenter):
            raise TypeError("datacenter must be a DataCenter")
        self._llm = llm
        self._datacenter = datacenter
        self._tools = _specs(tool_registry() if tools is None else tools)

    def answer_question(self, question: str) -> str:
        """Return the model's final answer for one non-empty question."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        decision = self._decide(build_selection_prompt(question, tuple(self._tools.values())))
        if isinstance(decision, FinalAnswer):
            return decision.text
        result = _execute(self._tools, self._datacenter, decision)
        follow_up = self._decide(build_answer_prompt(question, decision.name, result_json(result)))
        if not isinstance(follow_up, FinalAnswer):
            raise RuntimeError("only one tool call is allowed")
        return follow_up.text

    def _decide(self, prompt: str) -> ToolSelection | FinalAnswer:
        try:
            text = self._llm.generate_response(prompt)
        except Exception as exc:
            raise RuntimeError(f"model request failed: {exc}") from exc
        return parse_model_decision(text)


def answer_with_tools(question: str, datacenter: DataCenter, llm: LocalLLM | None = None) -> str:
    """Answer one question with the local model and the registered read-only tools."""
    if not isinstance(datacenter, DataCenter):
        raise TypeError("datacenter must be a DataCenter")
    return ToolCaller(LocalLLM() if llm is None else llm, datacenter).answer_question(question)


def build_selection_prompt(question: str, tools: Sequence[ToolSpec]) -> str:
    """Tell the model which tools exist and require one JSON object."""
    catalog = [
        {
            "name": spec.name,
            "description": spec.description,
            "arguments": [
                {"name": parameter.name, "type": parameter.type_name, "description": parameter.description}
                for parameter in spec.parameters
            ],
        }
        for spec in tools
    ]
    return (
        "You answer questions about a data center.\n"
        "Reply with one JSON object and no other text.\n"
        "To call a tool, reply:\n"
        '{"action":"tool","tool":"<name>","arguments":{...}}\n'
        "Arguments must match the tool exactly. Do not include a data center argument.\n"
        "To answer without a tool, reply:\n"
        '{"action":"final","answer":"..."}\n'
        "Call at most one tool.\n"
        "\n"
        f"TOOLS\n{json.dumps(catalog, indent=2)}\n"
        "\n"
        f"USER QUESTION\n{question.strip()}\n"
    )


def build_answer_prompt(question: str, tool_name: str, payload: str) -> str:
    """Give the model the tool result and require a final JSON answer."""
    return (
        "The tool result below is the current system state.\n"
        "Reply with one JSON object and no other text:\n"
        '{"action":"final","answer":"..."}\n'
        "Do not call another tool.\n"
        "Use only the tool result for facts about the current system.\n"
        "\n"
        f"TOOL\n{tool_name}\n"
        "\n"
        f"TOOL RESULT\n{payload}\n"
        "\n"
        f"USER QUESTION\n{question.strip()}\n"
    )


def parse_model_decision(text: str) -> ToolSelection | FinalAnswer:
    """Accept one JSON object. Anything else is rejected."""
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("model output was not valid JSON")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("model output was not valid JSON") from exc
    if isinstance(payload, list):
        raise RuntimeError("only one tool call is allowed")
    if not isinstance(payload, dict):
        raise RuntimeError("model output must be one JSON object")
    action = payload.get("action")
    if action == "final":
        _reject_unexpected(payload, {"action", "answer"})
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("answer must be a non-empty string")
        return FinalAnswer(answer.strip())
    if action == "tool":
        _reject_unexpected(payload, {"action", "tool", "arguments"})
        name = payload.get("tool")
        arguments = payload.get("arguments")
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError("tool name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise RuntimeError("arguments must be a JSON object")
        return ToolSelection(name, tuple(_argument_pairs(arguments)))
    if "action" not in payload:
        raise RuntimeError("model output is missing action")
    raise RuntimeError(f"model action {action!r} is not allowed")


def result_json(value: object) -> str:
    """Serialize a tool dataclass. Other objects are not sent to the model."""
    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError("tool result must be a dataclass")
    return json.dumps(asdict(value), sort_keys=True)


def _execute(tools: Mapping[str, ToolSpec], datacenter: DataCenter, selection: ToolSelection) -> object:
    spec = tools.get(selection.name)
    if spec is None:
        raise RuntimeError(f"unknown tool {selection.name!r}")
    arguments = _validated_arguments(spec, selection.arguments)
    try:
        return spec.function(datacenter, **arguments)
    except Exception as exc:
        raise RuntimeError(f"tool execution failed: {exc}") from exc


def _validated_arguments(spec: ToolSpec, arguments: tuple[tuple[str, str], ...]) -> dict[str, str]:
    supplied = dict(arguments)
    required = tuple(parameter.name for parameter in spec.parameters)
    missing = [name for name in required if name not in supplied]
    if missing:
        raise RuntimeError(f"missing argument {missing[0]!r}")
    unexpected = [name for name in supplied if name not in required]
    if unexpected:
        raise RuntimeError(f"unexpected argument {unexpected[0]!r}")
    validated: dict[str, str] = {}
    for parameter in spec.parameters:
        if parameter.type_name != "str":
            raise RuntimeError(f"argument {parameter.name!r} has an unsupported type")
        value = supplied[parameter.name]
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"argument {parameter.name!r} must be a non-empty string")
        validated[parameter.name] = value
    return validated


def _argument_pairs(arguments: dict[object, object]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for name, value in arguments.items():
        if not isinstance(name, str):
            raise RuntimeError("argument names must be strings")
        if isinstance(value, bool) or not isinstance(value, str):
            raise RuntimeError(f"argument {name!r} must be a string")
        pairs.append((name, value))
    return pairs


def _specs(tools: Sequence[ToolSpec]) -> dict[str, ToolSpec]:
    if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
        raise TypeError("tools must be a sequence of ToolSpec")
    found: dict[str, ToolSpec] = {}
    for spec in tools:
        if not isinstance(spec, ToolSpec):
            raise TypeError("tools must be a sequence of ToolSpec")
        if spec.name in found:
            raise ValueError(f"duplicate tool name {spec.name!r}")
        found[spec.name] = spec
    return found


def _reject_unexpected(payload: Mapping[str, object], allowed: set[str]) -> None:
    unexpected = [key for key in payload if key not in allowed]
    if unexpected:
        raise RuntimeError(f"unexpected field {unexpected[0]!r}")
