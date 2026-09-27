import unittest

from agent.autonomous_agent import AgentState, AutonomousAgent
from agent.event_bus import EventBus
from agent.events import Event, EventType
from models.datacenter import DataCenter
from models.rack import Rack
from models.server import Server
from models.workload import Workload
from optimizer.placement_optimizer import PlacementOptimizer
from predictor.demand_predictor import DemandPredictor
from simulator.engine import SimulationEngine


def make_server(server_id: str, *, cpu_capacity: float = 10, memory_capacity: float = 16) -> Server:
    return Server(
        server_id=server_id,
        rack_id="rack-1",
        cpu_capacity=cpu_capacity,
        memory_capacity=memory_capacity,
        temperature=20,
        power_consumption=0,
    )


def make_datacenter(*servers: Server) -> DataCenter:
    return DataCenter(racks=[Rack(rack_id="rack-1", cooling_capacity=1, servers=list(servers))])


def arrival(cpu_required: float = 2, history: list[float] | None = None, workload_id: str = "job-1") -> Event:
    payload: dict[str, object] = {
        "cpu_required": cpu_required,
        "memory_required": 2,
        "latency_requirement": 50,
        "priority": 1,
    }
    if history is not None:
        payload["history"] = history
    return Event(
        event_type=EventType.WORKLOAD_ARRIVAL,
        tick=1,
        source_id=workload_id,
        payload=payload,
    )


def make_agent(datacenter: DataCenter, optimizer: PlacementOptimizer | None = None) -> AutonomousAgent:
    return AutonomousAgent(
        datacenter=datacenter,
        predictor=DemandPredictor(window=3),
        optimizer=optimizer or PlacementOptimizer(),
    )


class AutonomousAgentTests(unittest.TestCase):
    def test_initial_state_is_idle(self) -> None:
        agent = make_agent(make_datacenter(make_server("server-1")))

        self.assertEqual(agent.state, AgentState.IDLE)
        self.assertEqual(agent.state_history, (AgentState.IDLE,))

    def test_workload_arrival_places_on_the_selected_server(self) -> None:
        server_a = make_server("server-1")
        server_b = make_server("server-2")
        datacenter = make_datacenter(server_a, server_b)
        agent = make_agent(datacenter)
        bus = EventBus()
        agent.subscribe(bus)

        bus.publish(arrival())

        self.assertEqual(
            agent.state_history,
            (
                AgentState.IDLE,
                AgentState.OBSERVING,
                AgentState.PLANNING,
                AgentState.EVALUATING,
                AgentState.ACTING,
                AgentState.VERIFYING,
                AgentState.COMPLETED,
            ),
        )
        self.assertEqual(agent.decision.server_id if agent.decision else None, "server-1")
        self.assertEqual([workload.workload_id for workload in server_a.workloads], ["job-1"])
        self.assertEqual(server_b.workloads, [])
        self.assertEqual(server_a.workloads[0].cpu_required, 2)

    def test_prediction_increases_planning_cpu_without_changing_current_cpu(self) -> None:
        server = make_server("server-1")
        agent = make_agent(make_datacenter(server))
        history = [1.0, 2.0, 3.0, 4.0, 5.0]

        decision = agent.handle_workload_arrival(arrival(cpu_required=5, history=history))

        self.assertIsNotNone(decision.predicted_cpu)
        self.assertGreater(decision.planning_cpu, 5)
        self.assertEqual(server.workloads[0].cpu_required, 5)

    def test_short_history_uses_current_cpu(self) -> None:
        agent = make_agent(make_datacenter(make_server("server-1")))

        decision = agent.handle_workload_arrival(arrival(cpu_required=3, history=[1.0, 2.0]))

        self.assertIsNone(decision.predicted_cpu)
        self.assertEqual(decision.planning_cpu, 3)
        self.assertEqual(decision.state, AgentState.COMPLETED)

    def test_planning_and_evaluation_do_not_mutate_the_real_datacenter(self) -> None:
        server = make_server("server-1")
        datacenter = make_datacenter(server)
        real_servers = [server]

        class Guard(PlacementOptimizer):
            def optimize(self, workloads, servers):  # type: ignore[no-untyped-def]
                self.real_unchanged = [item.workload_id for item in server.workloads] == []
                self.used_copies = all(candidate is not server for candidate in servers)
                return super().optimize(workloads, servers)

        guard = Guard()

        def engine_factory(temporary: DataCenter) -> SimulationEngine:
            self.assertIsNot(temporary, datacenter)
            self.assertEqual(server.workloads, [])
            return SimulationEngine(temporary)

        agent = AutonomousAgent(datacenter, DemandPredictor(window=3), guard, engine_factory)
        agent.handle_workload_arrival(arrival())

        self.assertTrue(guard.real_unchanged)
        self.assertTrue(guard.used_copies)
        self.assertEqual([item.workload_id for item in server.workloads], ["job-1"])
        self.assertEqual(real_servers, [server])

    def test_invalid_proposal_is_rejected_without_placement(self) -> None:
        server = make_server("server-1", cpu_capacity=4)
        agent = make_agent(make_datacenter(server))

        with self.assertRaises(ValueError):
            agent.handle_workload_arrival(arrival(cpu_required=8))

        self.assertEqual(agent.state, AgentState.FAILED)
        self.assertEqual(server.workloads, [])
        self.assertEqual(agent.decision.reason if agent.decision else "", "no feasible server")

    def test_already_placed_workload_is_rejected(self) -> None:
        server = make_server("server-1")
        existing = Workload(
            workload_id="job-1",
            cpu_required=1,
            memory_required=1,
            latency_requirement=50,
            priority=1,
        )
        server.allocate(existing)
        agent = make_agent(make_datacenter(server))

        with self.assertRaises(ValueError):
            agent.handle_workload_arrival(arrival())

        self.assertEqual(agent.state, AgentState.FAILED)
        self.assertEqual(server.workloads, [existing])

    def test_unexpected_failure_is_visible(self) -> None:
        datacenter = make_datacenter(make_server("server-1"))

        class Broken(PlacementOptimizer):
            def optimize(self, workloads, servers):  # type: ignore[no-untyped-def]
                raise RuntimeError("optimizer exploded")

        agent = AutonomousAgent(datacenter, DemandPredictor(window=3), Broken())

        with self.assertRaises(RuntimeError):
            agent.handle_workload_arrival(arrival())

        self.assertEqual(agent.state, AgentState.FAILED)
        self.assertEqual(datacenter.racks[0].servers[0].workloads, [])

    def test_repeated_execution_makes_the_same_decision(self) -> None:
        def run_once() -> tuple[str | None, float, AgentState]:
            agent = make_agent(make_datacenter(make_server("server-1"), make_server("server-2")))
            decision = agent.handle_workload_arrival(arrival(cpu_required=4, history=[1.0, 2.0, 3.0, 4.0, 5.0]))
            return decision.server_id, decision.planning_cpu, decision.state

        self.assertEqual(run_once(), run_once())


if __name__ == "__main__":
    unittest.main()
