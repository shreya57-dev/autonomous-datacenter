"""Place generated workloads on two servers and run a few simulation ticks.

Placement is round-robin only so the generator, models, and engine can be
checked together. It is not a scheduler.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.datacenter import DataCenter
from models.rack import Rack
from models.server import Server
from simulator.engine import SimulationEngine
from simulator.workload_generator import WorkloadGenerator

SEED = 42
WORKLOAD_COUNT = 5
TICKS = 5
CPU_CAPACITY = 20
MEMORY_CAPACITY = 40
COOLING_CAPACITY = 1


def build_servers() -> list[Server]:
    return [
        Server(
            server_id=f"server-{number}",
            rack_id="rack-1",
            cpu_capacity=CPU_CAPACITY,
            memory_capacity=MEMORY_CAPACITY,
            temperature=20,
            power_consumption=0,
        )
        for number in (1, 2)
    ]


def place_round_robin(servers: list[Server], generator: WorkloadGenerator) -> None:
    workloads = generator.generate(WORKLOAD_COUNT)
    for index, workload in enumerate(workloads):
        servers[index % len(servers)].allocate(workload)


def print_placement(servers: list[Server]) -> None:
    print(f"Seed {SEED}. Round-robin placement onto {len(servers)} servers.")
    print(f"{'workload':<12} {'cpu':>6} {'memory':>8} {'server':>10}")
    print("-" * 40)
    for server in servers:
        for workload in server.workloads:
            print(
                f"{workload.workload_id:<12} "
                f"{workload.cpu_required:6.2f} "
                f"{workload.memory_required:8.2f} "
                f"{server.server_id:>10}"
            )


def print_ticks(engine: SimulationEngine) -> None:
    print()
    print(f"{'tick':>4} {'server':<10} {'cpu':>6} {'memory':>8} {'power':>8} {'temp':>8}")
    print("-" * 48)
    for _ in range(TICKS):
        record = engine.advance_tick()
        for server in engine.datacenter.racks[0].servers:
            print(
                f"{record.tick:4d} "
                f"{server.server_id:<10} "
                f"{server.cpu_utilization():6.2f} "
                f"{server.memory_utilization():8.2f} "
                f"{server.power_consumption:8.1f} "
                f"{server.temperature:8.1f}"
            )


def main() -> None:
    servers = build_servers()
    place_round_robin(servers, WorkloadGenerator(seed=SEED, min_cpu=1, max_cpu=4, min_memory=1, max_memory=8))
    datacenter = DataCenter(
        racks=[Rack(rack_id="rack-1", cooling_capacity=COOLING_CAPACITY, servers=servers)]
    )
    engine = SimulationEngine(datacenter)

    print_placement(servers)
    print_ticks(engine)


if __name__ == "__main__":
    main()
