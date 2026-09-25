from dataclasses import dataclass, field
from enum import Enum

from .workload import Workload


class ServerStatus(Enum):
    """Whether a server can accept workloads."""

    OPERATIONAL = "operational"
    INACTIVE = "inactive"


@dataclass
class Server:
    """A machine with fixed CPU and memory capacity and the workloads placed on it."""

    server_id: str
    rack_id: str
    cpu_capacity: float
    memory_capacity: float
    temperature: float
    power_consumption: float
    status: ServerStatus = ServerStatus.OPERATIONAL
    workloads: list[Workload] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not str(self.server_id).strip():
            raise ValueError("server_id must be non-empty")
        if not str(self.rack_id).strip():
            raise ValueError("rack_id must be non-empty")
        _require_positive("cpu_capacity", self.cpu_capacity)
        _require_positive("memory_capacity", self.memory_capacity)
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise TypeError("temperature must be a number")
        _require_non_negative("power_consumption", self.power_consumption)
        if not isinstance(self.status, ServerStatus):
            raise TypeError("status must be a ServerStatus")

    def can_host(self, workload: Workload) -> bool:
        """Return whether this server can accept the workload without breaking capacity."""
        return self._hosting_problem(workload) is None

    def allocate(self, workload: Workload) -> None:
        """Place the workload on this server, or raise if that would be invalid."""
        if self._find_workload(workload.workload_id) is not None:
            raise ValueError(
                f"workload {workload.workload_id!r} is already assigned to server {self.server_id!r}"
            )
        problem = self._hosting_problem(workload)
        if problem is not None:
            raise ValueError(problem)
        self.workloads.append(workload)

    def release(self, workload: Workload) -> None:
        """Remove an assigned workload, or raise if it is not on this server."""
        index = self._find_workload(workload.workload_id)
        if index is None:
            raise ValueError(
                f"workload {workload.workload_id!r} is not assigned to server {self.server_id!r}"
            )
        del self.workloads[index]

    def cpu_utilization(self) -> float:
        """Fraction of CPU capacity used by assigned workloads, from 0 to 1."""
        return self._cpu_used() / self.cpu_capacity

    def memory_utilization(self) -> float:
        """Fraction of memory capacity used by assigned workloads, from 0 to 1."""
        return self._memory_used() / self.memory_capacity

    def _cpu_used(self) -> float:
        return sum(workload.cpu_required for workload in self.workloads)

    def _memory_used(self) -> float:
        return sum(workload.memory_required for workload in self.workloads)

    def _find_workload(self, workload_id: str) -> int | None:
        for index, workload in enumerate(self.workloads):
            if workload.workload_id == workload_id:
                return index
        return None

    def _hosting_problem(self, workload: Workload) -> str | None:
        if self.status is not ServerStatus.OPERATIONAL:
            return f"server {self.server_id!r} is not operational"
        if self._cpu_used() + workload.cpu_required > self.cpu_capacity:
            return (
                f"server {self.server_id!r} does not have enough CPU for workload {workload.workload_id!r}"
            )
        if self._memory_used() + workload.memory_required > self.memory_capacity:
            return (
                f"server {self.server_id!r} does not have enough memory for workload {workload.workload_id!r}"
            )
        return None


def _require_positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
