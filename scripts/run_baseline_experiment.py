"""Run one reproducible baseline scheduling and simulation pass.

Each run builds a new data center, so a repeated seed does not reuse
servers from a previous pass. This script is a demonstration, not an
evaluation framework.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.datacenter import DataCenter
from models.rack import Rack
from models.server import Server
from models.workload import Workload
from scheduler.baseline_scheduler import BaselineScheduler, ScheduleResult
from simulator.engine import SimulationEngine
from simulator.metrics import SystemMetrics, calculate_system_metrics
from simulator.workload_generator import WorkloadGenerator

SEED = 42
WORKLOAD_COUNT = 6
TICKS = 5
RACK_COUNT = 1
SERVERS_PER_RACK = 3
CPU_CAPACITY = 20
MEMORY_CAPACITY = 40
COOLING_CAPACITY = 1
STARTING_TEMPERATURE = 20


def generate_workloads() -> list[Workload]:
    return WorkloadGenerator(
        seed=SEED,
        min_cpu=1,
        max_cpu=4,
        min_memory=1,
        max_memory=8,
    ).generate(WORKLOAD_COUNT)


def build_environment() -> tuple[DataCenter, list[Server]]:
    """Return a new data center and the same server objects it contains."""
    racks: list[Rack] = []
    servers: list[Server] = []
    for rack_number in range(1, RACK_COUNT + 1):
        rack_id = f"rack-{rack_number}"
        rack_servers = [
            Server(
                server_id=f"server-{rack_number}-{server_number}",
                rack_id=rack_id,
                cpu_capacity=CPU_CAPACITY,
                memory_capacity=MEMORY_CAPACITY,
                temperature=STARTING_TEMPERATURE,
                power_consumption=0,
            )
            for server_number in range(1, SERVERS_PER_RACK + 1)
        ]
        servers.extend(rack_servers)
        racks.append(Rack(rack_id=rack_id, cooling_capacity=COOLING_CAPACITY, servers=rack_servers))
    return DataCenter(racks=racks), servers


def schedule_on_fresh_servers(workloads: list[Workload]) -> tuple[DataCenter, list[Server], ScheduleResult]:
    datacenter, servers = build_environment()
    result = BaselineScheduler().schedule(workloads, servers)
    return datacenter, servers, result


def print_placement(workloads: list[Workload], result: ScheduleResult) -> None:
    print(f"Baseline experiment  seed={SEED}  workloads={WORKLOAD_COUNT}  servers={RACK_COUNT * SERVERS_PER_RACK}")
    print()
    print(f"{'workload':<12} {'server':<14} {'cpu':>6} {'memory':>8}")
    print("-" * 44)
    for workload in workloads:
        server_id = result.placements.get(workload.workload_id, "unplaced")
        print(
            f"{workload.workload_id:<12} "
            f"{server_id:<14} "
            f"{workload.cpu_required:6.2f} "
            f"{workload.memory_required:8.2f}"
        )
    unplaced = ", ".join(result.unplaced_workloads) if result.unplaced_workloads else "none"
    print()
    print(f"placed: {len(result.placements)}   unplaced: {unplaced}")


def print_metrics(label: str, metrics: SystemMetrics) -> None:
    print(
        f"{label:>4}  "
        f"{metrics.active_server_count:6d}  "
        f"{metrics.total_cpu_utilization:6.2f}  "
        f"{metrics.total_memory_utilization:8.2f}  "
        f"{metrics.total_power_consumption:8.1f}  "
        f"{metrics.average_server_temperature:7.1f}  "
        f"{metrics.workload_count:9d}"
    )


def print_simulation(datacenter: DataCenter, servers: list[Server]) -> None:
    print()
    print(f"{'tick':>4}  {'active':>6}  {'cpu':>6}  {'memory':>8}  {'power':>8}  {'temp':>7}  {'workloads':>9}")
    print("-" * 68)
    print_metrics("0", calculate_system_metrics(servers))
    engine = SimulationEngine(datacenter)
    for _ in range(TICKS):
        record = engine.advance_tick()
        print_metrics(str(record.tick), calculate_system_metrics(servers))


def main() -> None:
    workloads = generate_workloads()
    datacenter, servers, result = schedule_on_fresh_servers(workloads)
    print_placement(workloads, result)
    print_simulation(datacenter, servers)

    _, _, repeated = schedule_on_fresh_servers(generate_workloads())
    print()
    print(f"same seed placement repeated: {repeated == result}")


if __name__ == "__main__":
    main()
