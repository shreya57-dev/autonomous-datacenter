from collections.abc import Iterable
from dataclasses import dataclass, field

from models.datacenter import DataCenter
from models.rack import Rack
from models.server import Server, ServerStatus

from .config import SimulationConfig


@dataclass(frozen=True)
class TickRecord:
    """Datacenter totals after one simulation tick."""

    tick: int
    total_cpu_utilization: float
    total_memory_utilization: float
    total_power_consumption: float
    average_server_temperature: float


@dataclass
class SimulationEngine:
    """Advances a data center one tick at a time and records each result.

    The engine updates power and temperature from the workloads already
    placed on servers. It does not choose where workloads run.
    """

    datacenter: DataCenter
    config: SimulationConfig = field(default_factory=SimulationConfig)
    tick: int = 0
    history: list[TickRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.datacenter, DataCenter):
            raise TypeError("datacenter must be a DataCenter")
        if not isinstance(self.config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig")
        if isinstance(self.tick, bool) or not isinstance(self.tick, int):
            raise TypeError("tick must be an int")
        if self.tick < 0:
            raise ValueError("tick cannot be negative")

    def advance_tick(self) -> TickRecord:
        """Move the simulation forward by one tick and record the new state."""
        self.tick += 1
        for rack in self.datacenter.racks:
            for server in rack.servers:
                self._update_server(server, rack)
        record = self._record_state()
        self.history.append(record)
        return record

    def _update_server(self, server: Server, rack: Rack) -> None:
        server.power_consumption = self._power(server)
        heat = self.config.temperature_increase_factor * server.cpu_utilization()
        cooling = self.config.cooling_factor * rack.cooling_capacity
        server.temperature = server.temperature + heat - cooling

    def _power(self, server: Server) -> float:
        if server.status is not ServerStatus.OPERATIONAL:
            return 0.0
        return self.config.idle_power + server.cpu_utilization() * (
            self.config.max_power - self.config.idle_power
        )

    def _record_state(self) -> TickRecord:
        servers = [server for rack in self.datacenter.racks for server in rack.servers]
        return TickRecord(
            tick=self.tick,
            total_cpu_utilization=_capacity_ratio(
                sum(server.cpu_utilization() * server.cpu_capacity for server in servers),
                sum(server.cpu_capacity for server in servers),
            ),
            total_memory_utilization=_capacity_ratio(
                sum(server.memory_utilization() * server.memory_capacity for server in servers),
                sum(server.memory_capacity for server in servers),
            ),
            total_power_consumption=sum(server.power_consumption for server in servers),
            average_server_temperature=_average(server.temperature for server in servers),
        )


def _capacity_ratio(used: float, capacity: float) -> float:
    if capacity == 0:
        return 0.0
    return used / capacity


def _average(values: Iterable[float]) -> float:
    readings = list(values)
    if not readings:
        return 0.0
    return sum(readings) / len(readings)
