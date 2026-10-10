import json
import unittest
from dataclasses import dataclass

from models.datacenter import DataCenter
from models.rack import Rack
from models.server import Server, ServerStatus
from models.workload import Workload
from tools.system_tools import ToolParameter, ToolSpec, tool_registry
from ai.tool_calling import ToolCaller


@dataclass(frozen=True)
class Note:
    text: str


class FakeLLM:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate_response(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("no scripted response")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return str(response)


class RecordingTool:
    def __init__(self) -> None:
        self.calls: list[tuple[DataCenter, str]] = []

    def __call__(self, datacenter: DataCenter, server_id: str) -> Note:
        self.calls.append((datacenter, server_id))
        return Note(text=f"status of {server_id}")


def tool_json(name: str, arguments: dict[str, object] | None = None) -> str:
    return json.dumps({"action": "tool", "tool": name, "arguments": {} if arguments is None else arguments})


def final_json(answer: str) -> str:
    return json.dumps({"action": "final", "answer": answer})


def make_datacenter() -> DataCenter:
    server = Server(
        server_id="server-1",
        rack_id="rack-1",
        cpu_capacity=10,
        memory_capacity=20,
        temperature=21,
        power_consumption=100,
        status=ServerStatus.OPERATIONAL,
    )
    server.allocate(
        Workload(
            workload_id="job-1",
            cpu_required=4,
            memory_required=8,
            latency_requirement=50,
            priority=1,
        )
    )
    return DataCenter(racks=[Rack(rack_id="rack-1", cooling_capacity=5, servers=[server])])


def status_spec(function: object) -> ToolSpec:
    return ToolSpec(
        name="get_server_status",
        description="Return one server.",
        parameters=(ToolParameter("server_id", "str", "Server id."),),
        function=function,  # type: ignore[arg-type]
    )


class ToolCallingTests(unittest.TestCase):
    def test_registered_tool_receives_the_argument_and_the_final_answer_is_returned(self) -> None:
        datacenter = make_datacenter()
        tool = RecordingTool()
        llm = FakeLLM(
            [
                tool_json("get_server_status", {"server_id": "server-1"}),
                final_json("Server server-1 is operational."),
            ]
        )
        caller = ToolCaller(llm, datacenter, tools=(status_spec(tool),))  # type: ignore[arg-type]

        answer = caller.answer_question("What is the status of server server-1?")

        self.assertEqual(answer, "Server server-1 is operational.")
        self.assertEqual(tool.calls, [(datacenter, "server-1")])
        self.assertIn("get_server_status", llm.prompts[0])
        self.assertIn("What is the status of server server-1?", llm.prompts[0])
        self.assertIn("status of server-1", llm.prompts[1])
        self.assertIn('"text": "status of server-1"', llm.prompts[1])
        self.assertNotIn('"name": "datacenter"', llm.prompts[0])

    def test_unknown_tool_is_rejected_before_execution(self) -> None:
        tool = RecordingTool()
        llm = FakeLLM([tool_json("delete_server", {"server_id": "server-1"})])

        with self.assertRaises(RuntimeError) as caught:
            ToolCaller(llm, make_datacenter(), tools=(status_spec(tool),)).answer_question(  # type: ignore[arg-type]
                "Shut down server-1."
            )
        self.assertIn("unknown tool", str(caught.exception))
        self.assertEqual(tool.calls, [])
        self.assertEqual(len(llm.prompts), 1)

    def test_missing_argument_is_rejected(self) -> None:
        tool = RecordingTool()
        llm = FakeLLM([tool_json("get_server_status", {})])

        with self.assertRaises(RuntimeError) as caught:
            ToolCaller(llm, make_datacenter(), tools=(status_spec(tool),)).answer_question(  # type: ignore[arg-type]
                "What is the status of server server-1?"
            )
        self.assertIn("missing argument 'server_id'", str(caught.exception))
        self.assertEqual(tool.calls, [])

    def test_unexpected_argument_is_rejected(self) -> None:
        tool = RecordingTool()
        llm = FakeLLM([tool_json("get_server_status", {"server_id": "server-1", "shutdown": "true"})])

        with self.assertRaises(RuntimeError) as caught:
            ToolCaller(llm, make_datacenter(), tools=(status_spec(tool),)).answer_question(  # type: ignore[arg-type]
                "What is the status of server server-1?"
            )
        self.assertIn("unexpected argument 'shutdown'", str(caught.exception))
        self.assertEqual(tool.calls, [])

    def test_malformed_output_is_rejected(self) -> None:
        tool = RecordingTool()
        llm = FakeLLM(["Call get_server_status for server-1."])

        with self.assertRaises(RuntimeError) as caught:
            ToolCaller(llm, make_datacenter(), tools=(status_spec(tool),)).answer_question(  # type: ignore[arg-type]
                "What is the status of server server-1?"
            )
        self.assertIn("not valid JSON", str(caught.exception))
        self.assertEqual(tool.calls, [])

    def test_final_answer_does_not_call_a_tool(self) -> None:
        tool = RecordingTool()
        llm = FakeLLM([final_json("A digital twin is a model of a system.")])

        answer = ToolCaller(llm, make_datacenter(), tools=(status_spec(tool),)).answer_question(  # type: ignore[arg-type]
            "What is a digital twin?"
        )

        self.assertEqual(answer, "A digital twin is a model of a system.")
        self.assertEqual(tool.calls, [])
        self.assertEqual(len(llm.prompts), 1)

    def test_a_second_tool_call_is_rejected(self) -> None:
        tool = RecordingTool()
        llm = FakeLLM(
            [
                tool_json("get_server_status", {"server_id": "server-1"}),
                tool_json("get_server_status", {"server_id": "server-1"}),
            ]
        )

        with self.assertRaises(RuntimeError) as caught:
            ToolCaller(llm, make_datacenter(), tools=(status_spec(tool),)).answer_question(  # type: ignore[arg-type]
                "What is the status of server server-1?"
            )
        self.assertIn("only one tool call is allowed", str(caught.exception))
        self.assertEqual(len(tool.calls), 1)

    def test_several_tool_calls_in_one_response_are_rejected(self) -> None:
        tool = RecordingTool()
        payload = json.dumps(
            [
                {"action": "tool", "tool": "get_server_status", "arguments": {"server_id": "server-1"}},
                {"action": "tool", "tool": "get_system_state", "arguments": {}},
            ]
        )
        llm = FakeLLM([payload])

        with self.assertRaises(RuntimeError) as caught:
            ToolCaller(llm, make_datacenter(), tools=(status_spec(tool),)).answer_question(  # type: ignore[arg-type]
                "Describe the facility."
            )
        self.assertIn("only one tool call is allowed", str(caught.exception))
        self.assertEqual(tool.calls, [])

    def test_tool_exception_is_reported_and_the_model_is_not_asked_again(self) -> None:
        def fail(datacenter: DataCenter, server_id: str) -> Note:
            raise ValueError(f"server {server_id!r} was not found")

        llm = FakeLLM([tool_json("get_server_status", {"server_id": "missing"})])

        with self.assertRaises(RuntimeError) as caught:
            ToolCaller(llm, make_datacenter(), tools=(status_spec(fail),)).answer_question(  # type: ignore[arg-type]
                "What is the status of server missing?"
            )
        self.assertIn("tool execution failed", str(caught.exception))
        self.assertIn("was not found", str(caught.exception))
        self.assertEqual(len(llm.prompts), 1)

    def test_empty_question_is_rejected_before_the_model(self) -> None:
        llm = FakeLLM([final_json("unused")])

        with self.assertRaises(ValueError):
            ToolCaller(llm, make_datacenter(), tools=tool_registry()).answer_question("  \n")  # type: ignore[arg-type]
        self.assertEqual(llm.prompts, [])

    def test_read_only_path_does_not_change_the_datacenter(self) -> None:
        datacenter = make_datacenter()
        server = datacenter.racks[0].servers[0]
        workload_list = server.workloads
        before = (server.status, server.temperature, server.power_consumption, tuple(server.workloads))
        llm = FakeLLM(
            [
                tool_json("get_server_status", {"server_id": "server-1"}),
                final_json("Server server-1 is operational and hosts job-1."),
            ]
        )

        answer = ToolCaller(llm, datacenter).answer_question("What is the status of server server-1?")  # type: ignore[arg-type]

        self.assertEqual(answer, "Server server-1 is operational and hosts job-1.")
        self.assertEqual(
            (server.status, server.temperature, server.power_consumption, tuple(server.workloads)),
            before,
        )
        self.assertIs(server.workloads, workload_list)
        self.assertIn("job-1", llm.prompts[1])
        self.assertIn('"status": "operational"', llm.prompts[1])
