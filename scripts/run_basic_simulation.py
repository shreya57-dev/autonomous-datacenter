"""Run the same server under a light workload and a heavy workload.

Both scenarios use one rack, one fresh server, and the same simulation
settings. Only the workload's CPU demand changes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.datacenter import DataCenter
from models.rack import Rack
from models.server import Server
from models.workload import Workload
from simulator.config import SimulationConfig
from simulator.engine import SimulationEngine, TickRecord

TICKS = 5
CPU_CAPACITY = 10
MEMORY_CAPACITY = 10
STARTING_TEMPERATURE = 20
COOLING_CAPACITY = 1
LOW_CPU = 2
HIGH_CPU = 8


def build_engine(server_id: str, workload_id: str, cpu_required: float) -> SimulationEngine:
    server = Server(
        server_id=server_id,
        rack_id="rack-1",
        cpu_capacity=CPU_CAPACITY,
        memory_capacity=MEMORY_CAPACITY,
        temperature=STARTING_TEMPERATURE,
        power_consumption=0,
    )
    server.allocate(
        Workload(
            workload_id=workload_id,
            cpu_required=cpu_required,
            memory_required=1,
            latency_requirement=50,
            priority=1,
        )
    )
    datacenter = DataCenter(
        racks=[Rack(rack_id="rack-1", cooling_capacity=COOLING_CAPACITY, servers=[server])]
    )
    return SimulationEngine(datacenter, config=SimulationConfig())


def run_scenario(name: str, cpu_required: float) -> list[TickRecord]:
    engine = build_engine(server_id=name, workload_id=f"{name}-job", cpu_required=cpu_required)
    return [engine.advance_tick() for _ in range(TICKS)]


def print_comparison(low: list[TickRecord], high: list[TickRecord]) -> None:
    print("One rack, one server, same cooling and configuration.")
    print(f"Low load: {LOW_CPU} CPU. High load: {HIGH_CPU} CPU. Capacity: {CPU_CAPACITY} CPU.")
    print()
    header = (
        f"{'tick':>4}  "
        f"{'low cpu':>8}  {'low power':>9}  {'low temp':>8}  "
        f"{'high cpu':>8}  {'high power':>10}  {'high temp':>9}"
    )
    print(header)
    print("-" * len(header))
    for low_tick, high_tick in zip(low, high):
        print(
            f"{low_tick.tick:4d}  "
            f"{low_tick.total_cpu_utilization:8.2f}  "
            f"{low_tick.total_power_consumption:9.1f}  "
            f"{low_tick.average_server_temperature:8.1f}  "
            f"{high_tick.total_cpu_utilization:8.2f}  "
            f"{high_tick.total_power_consumption:10.1f}  "
            f"{high_tick.average_server_temperature:9.1f}"
        )


def main() -> None:
    print_comparison(
        run_scenario("low", LOW_CPU),
        run_scenario("high", HIGH_CPU),
    )


if __name__ == "__main__":
    main()
