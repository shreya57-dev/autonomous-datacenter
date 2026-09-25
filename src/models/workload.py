from dataclasses import dataclass


@dataclass
class Workload:
    """A unit of demand that needs CPU, memory, and a latency target."""

    workload_id: str
    cpu_required: float
    memory_required: float
    latency_requirement: float
    priority: int

    def __post_init__(self) -> None:
        if not str(self.workload_id).strip():
            raise ValueError("workload_id must be non-empty")
        _require_non_negative("cpu_required", self.cpu_required)
        _require_non_negative("memory_required", self.memory_required)
        _require_non_negative("latency_requirement", self.latency_requirement)
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("priority must be an int")
        if self.priority < 0:
            raise ValueError("priority cannot be negative")


def _require_non_negative(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
