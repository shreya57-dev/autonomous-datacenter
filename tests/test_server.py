import unittest

from models.server import Server, ServerStatus
from models.workload import Workload


def make_server(
    *,
    cpu_capacity: float = 8,
    memory_capacity: float = 16,
    status: ServerStatus = ServerStatus.OPERATIONAL,
) -> Server:
    return Server(
        server_id="server-1",
        rack_id="rack-1",
        cpu_capacity=cpu_capacity,
        memory_capacity=memory_capacity,
        temperature=30,
        power_consumption=100,
        status=status,
    )


def make_workload(
    *,
    workload_id: str = "job-1",
    cpu_required: float = 2,
    memory_required: float = 4,
) -> Workload:
    return Workload(
        workload_id=workload_id,
        cpu_required=cpu_required,
        memory_required=memory_required,
        latency_requirement=50,
        priority=1,
    )


class ServerHostingTests(unittest.TestCase):
    def test_can_host_when_cpu_and_memory_are_available(self) -> None:
        server = make_server()
        workload = make_workload()

        self.assertTrue(server.can_host(workload))

    def test_rejects_workload_when_cpu_capacity_would_be_exceeded(self) -> None:
        server = make_server(cpu_capacity=4)
        server.allocate(make_workload(workload_id="job-1", cpu_required=3, memory_required=1))
        extra = make_workload(workload_id="job-2", cpu_required=2, memory_required=1)

        self.assertFalse(server.can_host(extra))

    def test_rejects_workload_when_memory_capacity_would_be_exceeded(self) -> None:
        server = make_server(memory_capacity=4)
        server.allocate(make_workload(workload_id="job-1", cpu_required=1, memory_required=3))
        extra = make_workload(workload_id="job-2", cpu_required=1, memory_required=2)

        self.assertFalse(server.can_host(extra))

    def test_inactive_server_cannot_host_a_workload(self) -> None:
        server = make_server(status=ServerStatus.INACTIVE)
        workload = make_workload()

        self.assertFalse(server.can_host(workload))
        with self.assertRaises(ValueError):
            server.allocate(workload)
        self.assertEqual(server.workloads, [])

    def test_allocation_updates_utilization(self) -> None:
        server = make_server(cpu_capacity=10, memory_capacity=20)
        server.allocate(make_workload(cpu_required=4, memory_required=5))

        self.assertEqual(server.cpu_utilization(), 0.4)
        self.assertEqual(server.memory_utilization(), 0.25)
        self.assertEqual(len(server.workloads), 1)

    def test_release_updates_utilization(self) -> None:
        server = make_server(cpu_capacity=10, memory_capacity=20)
        first = make_workload(workload_id="job-1", cpu_required=4, memory_required=5)
        second = make_workload(workload_id="job-2", cpu_required=2, memory_required=5)
        server.allocate(first)
        server.allocate(second)

        server.release(first)

        self.assertEqual(server.cpu_utilization(), 0.2)
        self.assertEqual(server.memory_utilization(), 0.25)
        self.assertEqual([workload.workload_id for workload in server.workloads], ["job-2"])

    def test_invalid_allocation_does_not_change_server_state(self) -> None:
        server = make_server(cpu_capacity=4, memory_capacity=8)
        placed = make_workload(workload_id="job-1", cpu_required=3, memory_required=4)
        server.allocate(placed)
        rejected = make_workload(workload_id="job-2", cpu_required=2, memory_required=1)

        with self.assertRaises(ValueError):
            server.allocate(rejected)

        self.assertEqual([workload.workload_id for workload in server.workloads], ["job-1"])
        self.assertEqual(server.cpu_utilization(), 0.75)
        self.assertEqual(server.memory_utilization(), 0.5)

    def test_release_of_unassigned_workload_leaves_server_unchanged(self) -> None:
        server = make_server()
        placed = make_workload(workload_id="job-1")
        server.allocate(placed)

        with self.assertRaises(ValueError):
            server.release(make_workload(workload_id="missing"))

        self.assertEqual([workload.workload_id for workload in server.workloads], ["job-1"])
        self.assertEqual(server.cpu_utilization(), 2 / 8)
        self.assertEqual(server.memory_utilization(), 4 / 16)


if __name__ == "__main__":
    unittest.main()
