import unittest

from models.server import Server, ServerStatus
from models.workload import Workload
from scheduler.baseline_scheduler import BaselineScheduler


def make_server(
    server_id: str,
    *,
    cpu_capacity: float = 10,
    memory_capacity: float = 16,
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


class BaselineSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = BaselineScheduler()

    def test_places_a_feasible_workload(self) -> None:
        server = make_server("server-1")
        workload = make_workload("job-1")

        result = self.scheduler.schedule([workload], [server])

        self.assertEqual(result.placements, {"job-1": "server-1"})
        self.assertEqual(result.unplaced_workloads, [])
        self.assertEqual(server.workloads, [workload])

    def test_selects_the_least_cpu_utilized_feasible_server(self) -> None:
        busy = make_server("busy")
        idle = make_server("idle")
        busy.allocate(make_workload("existing", cpu_required=6))
        incoming = make_workload("job-1", cpu_required=2)

        result = self.scheduler.schedule([incoming], [busy, idle])

        self.assertEqual(result.placements, {"job-1": "idle"})
        self.assertEqual(result.unplaced_workloads, [])

    def test_skips_a_server_with_insufficient_cpu(self) -> None:
        small = make_server("small", cpu_capacity=2)
        large = make_server("large", cpu_capacity=10)
        large.allocate(make_workload("existing", cpu_required=4))
        incoming = make_workload("job-1", cpu_required=3)

        result = self.scheduler.schedule([incoming], [small, large])

        self.assertEqual(result.placements, {"job-1": "large"})
        self.assertEqual([workload.workload_id for workload in small.workloads], [])

    def test_skips_a_server_with_insufficient_memory(self) -> None:
        small = make_server("small", memory_capacity=2)
        large = make_server("large", memory_capacity=16)
        large.allocate(make_workload("existing", cpu_required=4, memory_required=2))
        incoming = make_workload("job-1", cpu_required=1, memory_required=4)

        result = self.scheduler.schedule([incoming], [small, large])

        self.assertEqual(result.placements, {"job-1": "large"})
        self.assertEqual([workload.workload_id for workload in small.workloads], [])

    def test_skips_an_inactive_server(self) -> None:
        inactive = make_server("inactive", status=ServerStatus.INACTIVE)
        active = make_server("active")
        active.allocate(make_workload("existing", cpu_required=4))
        incoming = make_workload("job-1", cpu_required=2)

        result = self.scheduler.schedule([incoming], [inactive, active])

        self.assertEqual(result.placements, {"job-1": "active"})
        self.assertEqual(inactive.workloads, [])

    def test_reports_an_unplaceable_workload(self) -> None:
        server = make_server("server-1", cpu_capacity=4)
        workload = make_workload("job-1", cpu_required=5)

        result = self.scheduler.schedule([workload], [server])

        self.assertEqual(result.placements, {})
        self.assertEqual(result.unplaced_workloads, ["job-1"])
        self.assertEqual(server.workloads, [])

    def test_later_placements_use_updated_utilization(self) -> None:
        first_choice = make_server("server-1")
        second_choice = make_server("server-2")
        second_choice.allocate(make_workload("existing", cpu_required=3))
        first = make_workload("job-1", cpu_required=4)
        second = make_workload("job-2", cpu_required=2)

        result = self.scheduler.schedule([first, second], [first_choice, second_choice])

        self.assertEqual(result.placements, {"job-1": "server-1", "job-2": "server-2"})
        self.assertEqual(result.unplaced_workloads, [])

    def test_same_initial_scenario_is_deterministic(self) -> None:
        first = self.scheduler.schedule(
            [make_workload("job-1", cpu_required=4), make_workload("job-2", cpu_required=2)],
            _two_servers_with_existing_load(),
        )
        second = self.scheduler.schedule(
            [make_workload("job-1", cpu_required=4), make_workload("job-2", cpu_required=2)],
            _two_servers_with_existing_load(),
        )

        self.assertEqual(first, second)

    def test_failed_placement_does_not_mutate_server_state(self) -> None:
        server = make_server("server-1", cpu_capacity=10)
        placed = make_workload("job-1", cpu_required=6)
        rejected = make_workload("job-2", cpu_required=5)

        result = self.scheduler.schedule([placed, rejected], [server])

        self.assertEqual(result.placements, {"job-1": "server-1"})
        self.assertEqual(result.unplaced_workloads, ["job-2"])
        self.assertEqual(server.workloads, [placed])
        self.assertEqual(server.cpu_utilization(), 0.6)


def _two_servers_with_existing_load() -> list[Server]:
    lighter = make_server("server-1")
    heavier = make_server("server-2")
    heavier.allocate(make_workload("existing", cpu_required=3))
    return [lighter, heavier]


if __name__ == "__main__":
    unittest.main()
