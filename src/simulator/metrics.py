"""Current-state measurements for a set of servers.

These values are derived when requested. Nothing here places workloads
or keeps a second copy of server state.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from models.server import Server, ServerStatus


@dataclass(frozen=True)
class SystemMetrics:
    """Measurements taken from the servers at one moment."""

    total_cpu_utilization: float
    total_memory_utilization: float
    total_power_consumption: float
    average_server_temperature: float
    workload_count: int
    active_server_count: int
    inactive_server_count: int


def calculate_system_metrics(servers: Sequence[Server]) -> SystemMetrics:
    """Measure utilization, power, temperature, and counts from `servers`."""
    if isinstance(servers, (str, bytes)) or not isinstance(servers, Sequence):
        raise ValueError("servers must be a sequence of Server")
    if any(not isinstance(server, Server) for server in servers):
        raise ValueError("servers must be a sequence of Server")

    operational = [server for server in servers if server.status is ServerStatus.OPERATIONAL]
    inactive = [server for server in servers if server.status is ServerStatus.INACTIVE]
    cpu_capacity = sum(server.cpu_capacity for server in operational)
    memory_capacity = sum(server.memory_capacity for server in operational)
    return SystemMetrics(
        total_cpu_utilization=_ratio(
            sum(server.cpu_utilization() * server.cpu_capacity for server in operational),
            cpu_capacity,
        ),
        total_memory_utilization=_ratio(
            sum(server.memory_utilization() * server.memory_capacity for server in operational),
            memory_capacity,
        ),
        total_power_consumption=sum(server.power_consumption for server in operational),
        average_server_temperature=_average(server.temperature for server in operational),
        workload_count=sum(len(server.workloads) for server in servers),
        active_server_count=len(operational),
        inactive_server_count=len(inactive),
    )


def _ratio(used: float, capacity: float) -> float:
    if capacity == 0:
        return 0.0
    return used / capacity


def _average(values: Iterable[float]) -> float:
    readings = list(values)
    if not readings:
        return 0.0
    return sum(readings) / len(readings)
