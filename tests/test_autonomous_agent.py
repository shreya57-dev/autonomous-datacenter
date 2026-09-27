import unittest

from agent.autonomous_agent import AgentState, AutonomousAgent
from agent.event_bus import EventBus
from agent.events import Event, EventType
from models.datacenter import DataCenter
from models.rack import Rack
from models.server import Server, ServerStatus
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


def job(workload_id: str, cpu_required: float, memory_required: float = 1) -> Workload:
    return Workload(
        workload_id=workload_id,
        cpu_required=cpu_required,
        memory_required=memory_required,
        latency_requirement=50,
        priority=1,
    )


def server_failure(server_id: str, histories: dict[str, list[float]] | None = None) -> Event:
    payload: dict[str, object] = {"server_id": server_id, "reason": "hardware_failure"}
    if histories is not None:
        payload["histories"] = histories
    return Event(
        event_type=EventType.SERVER_FAILURE,
        tick=10,
        source_id=server_id,
        payload=payload,
    )


class ServerFailureRecoveryTests(unittest.TestCase):
    def test_server_failure_recovers_affected_workloads(self) -> None:
        failed, first, third, idle, keep, other, job_a, job_b = self._running_datacenter()
        agent = make_agent(make_datacenter(first, failed, third, idle))
        bus = EventBus()
        agent.subscribe(bus)

        bus.publish(server_failure("server-2"))

        decision = agent.decision
        self.assertIsNotNone(decision)
        assert decision is not None
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
        self.assertEqual(decision.state, AgentState.COMPLETED)
        self.assertEqual(decision.failed_server_id, "server-2")
        self.assertEqual(decision.affected_workloads, ("job-a", "job-b"))
        self.assertEqual(decision.recovered_workloads, ("job-a", "job-b"))
        self.assertEqual(decision.unrecovered_workloads, ())
        self.assertEqual(dict(decision.placements), {"job-a": "server-3", "job-b": "server-1"})
        self.assertEqual(failed.status, ServerStatus.INACTIVE)
        self.assertEqual(failed.workloads, [])
        self.assertEqual(first.workloads, [keep, job_b])
        self.assertEqual(third.workloads, [other, job_a])
        self.assertEqual(idle.workloads, [])
        self.assertEqual(job_a.cpu_required, 5)
        self.assertEqual(job_b.cpu_required, 4)
        self.assertLessEqual(first.cpu_utilization(), 1)
        self.assertLessEqual(third.cpu_utilization(), 1)
        self.assertLessEqual(first.memory_utilization(), 1)
        self.assertLessEqual(third.memory_utilization(), 1)

    def test_nonexistent_server_is_rejected(self) -> None:
        server = make_server("server-1")
        agent = make_agent(make_datacenter(server))

        with self.assertRaises(ValueError):
            agent.handle_server_failure(server_failure("server-9"))

        self.assertEqual(agent.state, AgentState.FAILED)
        self.assertEqual(agent.decision.reason if agent.decision else "", "server does not exist")
        self.assertEqual(server.status, ServerStatus.OPERATIONAL)

    def test_failed_server_is_excluded_from_recovery_planning(self) -> None:
        failed, first, third, idle, *_ = self._running_datacenter()
        datacenter = make_datacenter(first, failed, third, idle)

        class Guard(PlacementOptimizer):
            def optimize(self, workloads, servers):  # type: ignore[no-untyped-def]
                self.server_ids = [server.server_id for server in servers]
                self.real_status = failed.status
                self.real_workloads = [workload.workload_id for workload in failed.workloads]
                self.used_copies = all(server is not failed and server is not first for server in servers)
                return super().optimize(workloads, servers)

        guard = Guard()
        agent = AutonomousAgent(datacenter, DemandPredictor(window=3), guard)
        agent.handle_server_failure(server_failure("server-2"))

        self.assertEqual(guard.server_ids, ["server-1", "server-3"])
        self.assertEqual(guard.real_status, ServerStatus.OPERATIONAL)
        self.assertEqual(guard.real_workloads, ["job-a", "job-b"])
        self.assertTrue(guard.used_copies)

    def test_recovery_rejects_insufficient_cpu_without_changes(self) -> None:
        failed = make_server("server-2", cpu_capacity=10)
        victim = job("job-a", 8)
        failed.allocate(victim)
        keeper_server = make_server("server-1", cpu_capacity=4)
        keeper = job("job-keep", 1)
        keeper_server.allocate(keeper)
        spare = make_server("server-3", cpu_capacity=4)
        agent = make_agent(make_datacenter(keeper_server, failed, spare))

        with self.assertRaises(ValueError):
            agent.handle_server_failure(server_failure("server-2"))

        self.assertEqual(
            agent.state_history,
            (
                AgentState.IDLE,
                AgentState.OBSERVING,
                AgentState.PLANNING,
                AgentState.EVALUATING,
                AgentState.FAILED,
            ),
        )
        self.assertEqual(agent.decision.unrecovered_workloads if agent.decision else (), ("job-a",))
        self.assertEqual(agent.decision.recovered_workloads if agent.decision else (), ())
        self.assertEqual(failed.status, ServerStatus.OPERATIONAL)
        self.assertEqual(failed.workloads, [victim])
        self.assertEqual(keeper_server.workloads, [keeper])
        self.assertEqual(spare.workloads, [])

    def test_recovery_rejects_insufficient_memory_without_changes(self) -> None:
        failed = make_server("server-2")
        victim = job("job-a", 1, memory_required=8)
        failed.allocate(victim)
        keeper_server = make_server("server-1", memory_capacity=3)
        keeper = job("job-keep", 1, memory_required=1)
        keeper_server.allocate(keeper)
        spare = make_server("server-3", memory_capacity=3)
        agent = make_agent(make_datacenter(keeper_server, failed, spare))

        with self.assertRaises(ValueError):
            agent.handle_server_failure(server_failure("server-2"))

        self.assertEqual(agent.state, AgentState.FAILED)
        self.assertEqual(agent.decision.unrecovered_workloads if agent.decision else (), ("job-a",))
        self.assertEqual(failed.workloads, [victim])
        self.assertEqual(keeper_server.workloads, [keeper])
        self.assertEqual(spare.workloads, [])

    def test_prediction_increases_planning_cpu_and_can_reject_recovery(self) -> None:
        failed = make_server("server-2", cpu_capacity=10)
        victim = job("job-a", 4)
        failed.allocate(victim)
        survivor = make_server("server-1", cpu_capacity=5)
        agent = make_agent(make_datacenter(survivor, failed))
        history = [1.0, 2.0, 3.0, 4.0, 5.0]

        with self.assertRaises(ValueError):
            agent.handle_server_failure(server_failure("server-2", {"job-a": history}))

        decision = agent.decision
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertGreater(decision.planning_cpus[0][1], 4)
        self.assertEqual(victim.cpu_required, 4)
        self.assertEqual(failed.workloads, [victim])
        self.assertEqual(survivor.workloads, [])
        self.assertEqual(decision.recovered_workloads, ())

    def test_repeated_recovery_makes_the_same_placement(self) -> None:
        def run_once() -> tuple[tuple[tuple[str, str], ...], AgentState]:
            failed, first, third, idle, *_ = self._running_datacenter()
            agent = make_agent(make_datacenter(first, failed, third, idle))
            decision = agent.handle_server_failure(server_failure("server-2"))
            return decision.placements, decision.state

        self.assertEqual(run_once(), run_once())

    def _running_datacenter(self) -> tuple[Server, Server, Server, Server, Workload, Workload, Workload, Workload]:
        first = make_server("server-1")
        failed = make_server("server-2")
        third = make_server("server-3")
        idle = make_server("server-4")
        idle.status = ServerStatus.INACTIVE
        keep = job("job-keep", 6)
        other = job("job-other", 4)
        job_a = job("job-a", 5)
        job_b = job("job-b", 4)
        first.allocate(keep)
        third.allocate(other)
        failed.allocate(job_a)
        failed.allocate(job_b)
        return failed, first, third, idle, keep, other, job_a, job_b


if __name__ == "__main__":
    unittest.main()
