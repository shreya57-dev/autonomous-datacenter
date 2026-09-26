import unittest

from models.server import Server, ServerStatus
from models.workload import Workload
from optimizer.placement_optimizer import PlacementOptimizer


def make_server(
    server_id: str,
    *,
    cpu_capacity: float = 10,
    memory_capacity: float = 20,
    status: ServerStatus = ServerStatus.OPERATIONAL,
) -> Server:
    return Server(
        server_id=server_id,
        rack_id="rack-1",
        cpu_capacity=cpu_capacity,
        memory_capacity=memory_capacity,
        temperature=30,
        power_consumption=100,
        status=status,
    )


def make_workload(
    workload_id: str,
    *,
    cpu_required: float = 2,
    memory_required: float = 2,
) -> Workload:
    return Workload(
        workload_id=workload_id,
        cpu_required=cpu_required,
        memory_required=memory_required,
        latency_requirement=50,
        priority=1,
    )


def cpu_used_by_server(placements: dict[str, str], workloads: list[Workload]) -> dict[str, float]:
    requirements = {workload.workload_id: workload.cpu_required for workload in workloads}
    used: dict[str, float] = {}
    for workload_id, server_id in placements.items():
        used[server_id] = used.get(server_id, 0.0) + requirements[workload_id]
    return used


def first_fit_peak(workloads: list[Workload], servers: list[Server]) -> float:
    """Assign each workload, in order, to the first server that still has CPU."""
    remaining = {server.server_id: server.cpu_capacity for server in servers}
    used = {server.server_id: 0.0 for server in servers}
    for workload in workloads:
        for server in servers:
            if workload.cpu_required <= remaining[server.server_id]:
                remaining[server.server_id] -= workload.cpu_required
                used[server.server_id] += workload.cpu_required
                break
    return max(used[server.server_id] / server.cpu_capacity for server in servers)


class PlacementOptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.optimizer = PlacementOptimizer()

    def test_assigns_a_simple_feasible_workload(self) -> None:
        server = make_server("server-1")
        workload = make_workload("job-1", cpu_required=2, memory_required=4)

        result = self.optimizer.optimize([workload], [server])

        self.assertEqual(result.placements, {"job-1": "server-1"})
        self.assertEqual(result.unplaced_workloads, [])
        self.assertAlmostEqual(result.objective_value, 0.2)

    def test_cpu_capacity_is_never_exceeded(self) -> None:
        server = make_server("server-1", cpu_capacity=5, memory_capacity=10)
        workloads = [
            make_workload("job-1", cpu_required=2.5, memory_required=1),
            make_workload("job-2", cpu_required=2.5, memory_required=1),
            make_workload("job-3", cpu_required=0.2, memory_required=1),
        ]

        result = self.optimizer.optimize(workloads, [server])

        used = sum(cpu_used_by_server(result.placements, workloads).values())
        self.assertEqual(len(result.placements), 2)
        self.assertEqual(len(result.unplaced_workloads), 1)
        self.assertLessEqual(used, 5)

    def test_memory_capacity_is_never_exceeded(self) -> None:
        server = make_server("server-1", cpu_capacity=10, memory_capacity=5)
        workloads = [
            make_workload("job-1", cpu_required=1, memory_required=2.5),
            make_workload("job-2", cpu_required=1, memory_required=2.5),
            make_workload("job-3", cpu_required=1, memory_required=0.2),
        ]

        result = self.optimizer.optimize(workloads, [server])

        used = 0.0
        requirements = {workload.workload_id: workload.memory_required for workload in workloads}
        for workload_id in result.placements:
            used += requirements[workload_id]
        self.assertEqual(len(result.placements), 2)
        self.assertEqual(len(result.unplaced_workloads), 1)
        self.assertLessEqual(used, 5)

    def test_inactive_servers_are_never_selected(self) -> None:
        inactive = make_server("inactive", cpu_capacity=100, memory_capacity=100, status=ServerStatus.INACTIVE)
        active = make_server("active", cpu_capacity=10, memory_capacity=10)
        fits_active = make_workload("fits-active", cpu_required=4, memory_required=2)
        fits_only_inactive = make_workload("too-big", cpu_required=50, memory_required=2)

        result = self.optimizer.optimize([fits_active, fits_only_inactive], [inactive, active])

        self.assertEqual(result.placements, {"fits-active": "active"})
        self.assertEqual(result.unplaced_workloads, ["too-big"])
        self.assertNotIn("inactive", result.placements.values())

    def test_workload_that_fits_nowhere_is_unplaced(self) -> None:
        server = make_server("server-1", cpu_capacity=4, memory_capacity=4)
        workload = make_workload("job-1", cpu_required=5, memory_required=1)

        result = self.optimizer.optimize([workload], [server])

        self.assertEqual(result.placements, {})
        self.assertEqual(result.unplaced_workloads, ["job-1"])

    def test_multiple_workloads_are_assigned_consistently(self) -> None:
        servers = [make_server("server-1", cpu_capacity=10), make_server("server-2", cpu_capacity=10)]
        workloads = [
            make_workload("job-1", cpu_required=3, memory_required=2),
            make_workload("job-2", cpu_required=3, memory_required=2),
            make_workload("job-3", cpu_required=3, memory_required=2),
            make_workload("job-4", cpu_required=3, memory_required=2),
        ]

        result = self.optimizer.optimize(workloads, servers)

        self.assertEqual(sorted(result.placements), ["job-1", "job-2", "job-3", "job-4"])
        self.assertEqual(result.unplaced_workloads, [])
        for server in servers:
            used = cpu_used_by_server(result.placements, workloads).get(server.server_id, 0.0)
            self.assertLessEqual(used, server.cpu_capacity)
        self.assertEqual(len(set(result.placements.values())), 2)

    def test_peak_objective_beats_input_order_assignment(self) -> None:
        # No pair of these fits on one server of capacity 10.
        # Input order keeps the large job and leaves a higher peak.
        # The optimizer leaves the large job unplaced and keeps the lower peak.
        servers = [make_server("server-1"), make_server("server-2")]
        workloads = [
            make_workload("big", cpu_required=9, memory_required=1),
            make_workload("medium", cpu_required=6, memory_required=1),
            make_workload("small", cpu_required=5, memory_required=1),
        ]

        result = self.optimizer.optimize(workloads, servers)
        optimizer_peak = max(cpu_used_by_server(result.placements, workloads).values()) / 10

        self.assertEqual(result.unplaced_workloads, ["big"])
        self.assertEqual(set(result.placements), {"medium", "small"})
        self.assertEqual(len(set(result.placements.values())), 2)
        self.assertLess(optimizer_peak, first_fit_peak(workloads, servers))

    def test_placements_do_not_mutate_servers(self) -> None:
        server = make_server("server-1", cpu_capacity=10)
        existing = make_workload("existing", cpu_required=2, memory_required=2)
        server.allocate(existing)
        before = (
            [workload.workload_id for workload in server.workloads],
            server.cpu_utilization(),
            server.temperature,
            server.power_consumption,
            server.status,
        )

        result = self.optimizer.optimize([make_workload("job-1", cpu_required=2)], [server])

        after = (
            [workload.workload_id for workload in server.workloads],
            server.cpu_utilization(),
            server.temperature,
            server.power_consumption,
            server.status,
        )
        self.assertEqual(after, before)
        self.assertEqual(result.placements, {"job-1": "server-1"})
        self.assertEqual(server.workloads, [existing])

    def test_duplicate_workload_ids_are_rejected(self) -> None:
        server = make_server("server-1")
        workloads = [make_workload("job-1"), make_workload("job-1")]

        with self.assertRaises(ValueError):
            self.optimizer.optimize(workloads, [server])
        self.assertEqual(server.workloads, [])

    def test_duplicate_server_ids_are_rejected(self) -> None:
        servers = [make_server("server-1"), make_server("server-1")]

        with self.assertRaises(ValueError):
            self.optimizer.optimize([make_workload("job-1")], servers)
        self.assertEqual(servers[0].workloads, [])
        self.assertEqual(servers[1].workloads, [])

    def test_already_assigned_workloads_are_rejected(self) -> None:
        server = make_server("server-1")
        workload = make_workload("job-1")
        server.allocate(workload)

        with self.assertRaises(ValueError):
            self.optimizer.optimize([workload], [server])
        self.assertEqual(server.workloads, [workload])

    def test_solver_status_is_optimal_for_a_feasible_case(self) -> None:
        result = self.optimizer.optimize(
            [make_workload("job-1", cpu_required=2)],
            [make_server("server-1")],
        )

        self.assertEqual(result.solver_status, "OPTIMAL")

    def test_repeated_execution_returns_the_same_result(self) -> None:
        workloads = [
            make_workload("job-1", cpu_required=4, memory_required=2),
            make_workload("job-2", cpu_required=3, memory_required=2),
            make_workload("job-3", cpu_required=2, memory_required=2),
        ]
        servers = [make_server("server-1"), make_server("server-2")]

        first = self.optimizer.optimize(workloads, servers)
        second = self.optimizer.optimize(workloads, servers)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
