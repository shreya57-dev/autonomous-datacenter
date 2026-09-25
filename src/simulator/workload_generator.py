"""Create reproducible batches of workloads for simulation experiments."""

import random

from models.workload import Workload


class WorkloadGenerator:
    """Build Workload objects from configurable ranges and a fixed seed.

    Each generator keeps its own random.Random, so it does not change
    Python's global random state.
    """

    def __init__(
        self,
        seed: int,
        *,
        min_cpu: float = 1,
        max_cpu: float = 8,
        min_memory: float = 1,
        max_memory: float = 16,
        min_latency: float = 10,
        max_latency: float = 100,
        min_priority: int = 1,
        max_priority: int = 5,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an int")
        _require_resource_range("cpu", min_cpu, max_cpu)
        _require_resource_range("memory", min_memory, max_memory)
        _require_resource_range("latency", min_latency, max_latency)
        _require_priority_range(min_priority, max_priority)

        self._random = random.Random(seed)
        self._min_cpu = min_cpu
        self._max_cpu = max_cpu
        self._min_memory = min_memory
        self._max_memory = max_memory
        self._min_latency = min_latency
        self._max_latency = max_latency
        self._min_priority = min_priority
        self._max_priority = max_priority

    def generate(self, count: int) -> list[Workload]:
        """Return `count` workloads with ids workload-1, workload-2, and so on."""
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError("number of workloads must be an int")
        if count < 0:
            raise ValueError("number of workloads cannot be negative")

        workloads: list[Workload] = []
        for number in range(1, count + 1):
            workloads.append(
                Workload(
                    workload_id=f"workload-{number}",
                    cpu_required=self._random.uniform(self._min_cpu, self._max_cpu),
                    memory_required=self._random.uniform(self._min_memory, self._max_memory),
                    latency_requirement=self._random.uniform(self._min_latency, self._max_latency),
                    priority=self._random.randint(self._min_priority, self._max_priority),
                )
            )
        return workloads


def _require_resource_range(name: str, minimum: float, maximum: float) -> None:
    _require_non_negative_number(f"minimum {name}", minimum)
    _require_non_negative_number(f"maximum {name}", maximum)
    if minimum > maximum:
        raise ValueError(f"minimum {name} cannot be greater than maximum {name}")


def _require_priority_range(minimum: int, maximum: int) -> None:
    _require_non_negative_int("minimum priority", minimum)
    _require_non_negative_int("maximum priority", maximum)
    if minimum > maximum:
        raise ValueError("minimum priority cannot be greater than maximum priority")


def _require_non_negative_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")


def _require_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
