"""Compare baseline scheduling and the placement optimizer on one scenario.

Both runs start from a new data center and the same generated workloads.
Power and temperature are observed after the same number of ticks.
They are not an energy-efficiency result.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.datacenter import DataCenter
from models.rack import Rack
from models.server import Server, ServerStatus
from models.workload import Workload
from optimizer.placement_optimizer import OptimizationResult, PlacementOptimizer
from scheduler.baseline_scheduler import BaselineScheduler, ScheduleResult
from simulator.engine import SimulationEngine
from simulator.metrics import calculate_system_metrics
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
    """Return a new data center and the server objects inside it."""
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


def apply_placements(servers: list[Server], workloads: list[Workload], placements: dict[str, str]) -> None:
    workload_by_id = {workload.workload_id: workload for workload in workloads}
    server_by_id = {server.server_id: server for server in servers}
    for workload_id, server_id in placements.items():
        server_by_id[server_id].allocate(workload_by_id[workload_id])


def run_ticks(datacenter: DataCenter) -> None:
    engine = SimulationEngine(datacenter)
    for _ in range(TICKS):
        engine.advance_tick()


def peak_cpu_utilization(servers: list[Server]) -> float:
    operational = [server for server in servers if server.status is ServerStatus.OPERATIONAL]
    if not operational:
        return 0.0
    return max(server.cpu_utilization() for server in operational)


def ids_or_none(workload_ids: list[str]) -> str:
    if not workload_ids:
        return "none"
    return ", ".join(workload_ids)


def print_metric(name: str, baseline_value: float, optimizer_value: float) -> None:
    print(f"{name:<28} {baseline_value:12.4f} {optimizer_value:12.4f}")


def print_comparison(
    workloads: list[Workload],
    baseline: ScheduleResult,
    optimized: OptimizationResult,
    baseline_servers: list[Server],
    optimizer_servers: list[Server],
) -> None:
    baseline_metrics = calculate_system_metrics(baseline_servers)
    optimizer_metrics = calculate_system_metrics(optimizer_servers)
    print(
        f"Comparison  seed={SEED}  workloads={WORKLOAD_COUNT}  "
        f"servers={RACK_COUNT * SERVERS_PER_RACK}  ticks={TICKS}"
    )
    print()
    print(f"{'workload':<12} {'baseline':<14} {'optimizer':<14}")
    print("-" * 42)
    for workload in workloads:
        print(
            f"{workload.workload_id:<12} "
            f"{baseline.placements.get(workload.workload_id, 'unplaced'):<14} "
            f"{optimized.placements.get(workload.workload_id, 'unplaced'):<14}"
        )
    print()
    print(f"baseline unplaced:  {ids_or_none(baseline.unplaced_workloads)}")
    print(f"optimizer unplaced: {ids_or_none(optimized.unplaced_workloads)}")
    print()
    print(f"{'metric':<28} {'baseline':>12} {'optimizer':>12}")
    print("-" * 54)
    print_metric(
        "peak cpu utilization",
        peak_cpu_utilization(baseline_servers),
        peak_cpu_utilization(optimizer_servers),
    )
    print_metric(
        "total cpu utilization",
        baseline_metrics.total_cpu_utilization,
        optimizer_metrics.total_cpu_utilization,
    )
    print_metric(
        "memory utilization",
        baseline_metrics.total_memory_utilization,
        optimizer_metrics.total_memory_utilization,
    )
    print_metric(
        "total power",
        baseline_metrics.total_power_consumption,
        optimizer_metrics.total_power_consumption,
    )
    print_metric(
        "average temperature",
        baseline_metrics.average_server_temperature,
        optimizer_metrics.average_server_temperature,
    )
    print()
    print(f"optimizer solver status:    {optimized.solver_status}")
    print(f"optimizer rejection count:  {len(optimized.unplaced_workloads)}")
    print(f"optimizer objective value:  {optimized.objective_value:.4f}")
    print()
    print("Power and temperature are observed after the same simulation. They are not an energy comparison.")


def main() -> None:
    workloads = generate_workloads()

    baseline_datacenter, baseline_servers = build_environment()
    baseline_result = BaselineScheduler().schedule(workloads, baseline_servers)

    optimizer_datacenter, optimizer_servers = build_environment()
    optimized = PlacementOptimizer().optimize(workloads, optimizer_servers)
    apply_placements(optimizer_servers, workloads, optimized.placements)

    run_ticks(baseline_datacenter)
    run_ticks(optimizer_datacenter)
    print_comparison(workloads, baseline_result, optimized, baseline_servers, optimizer_servers)


if __name__ == "__main__":
    main()
