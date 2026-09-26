import unittest

from models.server import Server, ServerStatus
from models.workload import Workload
from simulator.metrics import calculate_system_metrics


def make_server(
    server_id: str,
    *,
    cpu_capacity: float = 10,
    memory_capacity: float = 20,
    temperature: float = 20,
    power_consumption: float = 100,
    status: ServerStatus = ServerStatus.OPERATIONAL,
) -> Server:
    return Server(
        server_id=server_id,
        rack_id="rack-1",
        cpu_capacity=cpu_capacity,
        memory_capacity=memory_capacity,
        temperature=temperature,
        power_consumption=power_consumption,
        status=status,
    )


def make_workload(
    workload_id: str,
    *,
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


class SystemMetricsTests(unittest.TestCase):
    def test_cpu_utilization_uses_operational_servers_only(self) -> None:
        first = make_server("server-1", cpu_capacity=10)
        second = make_server("server-2", cpu_capacity=10)
        inactive = make_server("server-3", cpu_capacity=10)
        first.allocate(make_workload("job-1", cpu_required=4, memory_required=1))
        second.allocate(make_workload("job-2", cpu_required=2, memory_required=1))
        inactive.allocate(make_workload("job-3", cpu_required=9, memory_required=1))
        inactive.status = ServerStatus.INACTIVE

        metrics = calculate_system_metrics([first, second, inactive])

        self.assertEqual(metrics.total_cpu_utilization, 0.3)

    def test_memory_utilization_uses_operational_servers_only(self) -> None:
        first = make_server("server-1", memory_capacity=16)
        second = make_server("server-2", memory_capacity=16)
        inactive = make_server("server-3", memory_capacity=16)
        first.allocate(make_workload("job-1", cpu_required=1, memory_required=8))
        second.allocate(make_workload("job-2", cpu_required=1, memory_required=4))
        inactive.allocate(make_workload("job-3", cpu_required=1, memory_required=16))
        inactive.status = ServerStatus.INACTIVE

        metrics = calculate_system_metrics([first, second, inactive])

        self.assertEqual(metrics.total_memory_utilization, 0.375)

    def test_total_power_sums_operational_servers(self) -> None:
        servers = [
            make_server("server-1", power_consumption=100),
            make_server("server-2", power_consumption=150),
            make_server("server-3", power_consumption=999, status=ServerStatus.INACTIVE),
        ]

        metrics = calculate_system_metrics(servers)

        self.assertEqual(metrics.total_power_consumption, 250)

    def test_average_temperature_uses_operational_servers(self) -> None:
        servers = [
            make_server("server-1", temperature=20),
            make_server("server-2", temperature=30),
            make_server("server-3", temperature=100, status=ServerStatus.INACTIVE),
        ]

        metrics = calculate_system_metrics(servers)

        self.assertEqual(metrics.average_server_temperature, 25)

    def test_workload_count_includes_every_assigned_workload(self) -> None:
        first = make_server("server-1")
        inactive = make_server("server-2")
        first.allocate(make_workload("job-1"))
        first.allocate(make_workload("job-2"))
        inactive.allocate(make_workload("job-3"))
        inactive.status = ServerStatus.INACTIVE

        metrics = calculate_system_metrics([first, inactive])

        self.assertEqual(metrics.workload_count, 3)

    def test_active_and_inactive_server_counts(self) -> None:
        servers = [
            make_server("server-1"),
            make_server("server-2"),
            make_server("server-3", status=ServerStatus.INACTIVE),
        ]

        metrics = calculate_system_metrics(servers)

        self.assertEqual(metrics.active_server_count, 2)
        self.assertEqual(metrics.inactive_server_count, 1)

    def test_empty_server_list(self) -> None:
        metrics = calculate_system_metrics([])

        self.assertEqual(metrics.total_cpu_utilization, 0.0)
        self.assertEqual(metrics.total_memory_utilization, 0.0)
        self.assertEqual(metrics.total_power_consumption, 0.0)
        self.assertEqual(metrics.average_server_temperature, 0.0)
        self.assertEqual(metrics.workload_count, 0)
        self.assertEqual(metrics.active_server_count, 0)
        self.assertEqual(metrics.inactive_server_count, 0)

    def test_all_servers_inactive(self) -> None:
        server = make_server(
            "server-1",
            temperature=40,
            power_consumption=80,
        )
        server.allocate(make_workload("job-1", cpu_required=4, memory_required=4))
        server.status = ServerStatus.INACTIVE

        metrics = calculate_system_metrics([server])

        self.assertEqual(metrics.total_cpu_utilization, 0.0)
        self.assertEqual(metrics.total_memory_utilization, 0.0)
        self.assertEqual(metrics.total_power_consumption, 0.0)
        self.assertEqual(metrics.average_server_temperature, 0.0)
        self.assertEqual(metrics.workload_count, 1)
        self.assertEqual(metrics.active_server_count, 0)
        self.assertEqual(metrics.inactive_server_count, 1)

    def test_zero_utilization_operational_server(self) -> None:
        server = make_server("server-1", temperature=22, power_consumption=110)

        metrics = calculate_system_metrics([server])

        self.assertEqual(metrics.total_cpu_utilization, 0.0)
        self.assertEqual(metrics.total_memory_utilization, 0.0)
        self.assertEqual(metrics.total_power_consumption, 110)
        self.assertEqual(metrics.average_server_temperature, 22)
        self.assertEqual(metrics.workload_count, 0)
        self.assertEqual(metrics.active_server_count, 1)

    def test_metrics_change_after_workload_allocation(self) -> None:
        server = make_server("server-1", cpu_capacity=10, memory_capacity=20)
        before = calculate_system_metrics([server])

        server.allocate(make_workload("job-1", cpu_required=5, memory_required=10))
        after = calculate_system_metrics([server])

        self.assertEqual(before.total_cpu_utilization, 0.0)
        self.assertEqual(before.total_memory_utilization, 0.0)
        self.assertEqual(before.workload_count, 0)
        self.assertEqual(after.total_cpu_utilization, 0.5)
        self.assertEqual(after.total_memory_utilization, 0.5)
        self.assertEqual(after.workload_count, 1)


if __name__ == "__main__":
    unittest.main()
