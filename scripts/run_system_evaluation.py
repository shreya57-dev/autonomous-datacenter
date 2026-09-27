"""Evaluate placement and server-failure recovery on a fixed set of seeds.

Each seed builds a new data center and a new workload batch. The script
does not change the agent, optimizer, or predictor. A recovery that the
agent rejects is reported as a failed recovery.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.autonomous_agent import AgentState, AutonomousAgent
from agent.event_bus import EventBus
from agent.events import Event, EventType
from models.datacenter import DataCenter
from models.rack import Rack
from models.server import Server, ServerStatus
from models.workload import Workload
from optimizer.placement_optimizer import PlacementOptimizer
from predictor.demand_predictor import DemandPredictor
from simulator.metrics import calculate_system_metrics
from simulator.workload_generator import WorkloadGenerator

SEEDS = [1, 2, 3, 4, 5]
WORKLOAD_COUNT = 8
SERVER_COUNT = 4
CPU_CAPACITY = 20
MEMORY_CAPACITY = 40
MIN_CPU = 1
MAX_CPU = 4
MIN_MEMORY = 1
MAX_MEMORY = 8


def build_datacenter() -> tuple[DataCenter, list[Server]]:
    servers = [
        Server(
            server_id=f"server-{number}",
            rack_id="rack-1",
            cpu_capacity=CPU_CAPACITY,
            memory_capacity=MEMORY_CAPACITY,
            temperature=20,
            power_consumption=0,
        )
        for number in range(1, SERVER_COUNT + 1)
    ]
    datacenter = DataCenter(racks=[Rack(rack_id="rack-1", cooling_capacity=1, servers=servers)])
    return datacenter, servers


def place_initially(workloads: list[Workload], servers: list[Server]) -> tuple[dict[str, str], list[str]]:
    result = PlacementOptimizer().optimize(workloads, servers)
    by_id = {server.server_id: server for server in servers}
    for workload in workloads:
        server_id = result.placements.get(workload.workload_id)
        if server_id is not None:
            by_id[server_id].allocate(workload)
    return dict(result.placements), list(result.unplaced_workloads)


def choose_failed_server(servers: list[Server]) -> Server:
    """Pick one operational server without using random.

    Prefer the server hosting the most workloads. Equal counts use the
    larger CPU assignment, then the earlier server id.
    """
    operational = [server for server in servers if server.status is ServerStatus.OPERATIONAL]
    occupied = [server for server in operational if server.workloads]
    candidates = occupied or operational
    return min(candidates, key=lambda server: (-len(server.workloads), -_cpu_used(server), server.server_id))


def peak_cpu(servers: list[Server]) -> float:
    operational = [server for server in servers if server.status is ServerStatus.OPERATIONAL]
    if not operational:
        return 0.0
    return max(server.cpu_utilization() for server in operational)


def evaluate_seed(seed: int) -> dict[str, object]:
    workloads = WorkloadGenerator(
        seed,
        min_cpu=MIN_CPU,
        max_cpu=MAX_CPU,
        min_memory=MIN_MEMORY,
        max_memory=MAX_MEMORY,
    ).generate(WORKLOAD_COUNT)
    datacenter, servers = build_datacenter()
    placements, unplaced = place_initially(workloads, servers)
    initial_peak = peak_cpu(servers)
    failed = choose_failed_server(servers)
    affected = [workload.workload_id for workload in failed.workloads]
    unaffected = {
        workload.workload_id: server.server_id
        for server in servers
        if server is not failed
        for workload in server.workloads
    }
    before = _snapshot(servers)

    agent = AutonomousAgent(
        datacenter=datacenter,
        predictor=DemandPredictor(window=3),
        optimizer=PlacementOptimizer(),
    )
    bus = EventBus()
    agent.subscribe(bus)
    error = ""
    try:
        bus.publish(
            Event(
                event_type=EventType.SERVER_FAILURE,
                tick=1,
                source_id=failed.server_id,
                payload={"server_id": failed.server_id, "reason": "hardware_failure"},
            )
        )
    except ValueError as exc:
        error = str(exc)

    decision = agent.decision
    metrics = calculate_system_metrics(servers)
    violations = _violations(
        servers,
        failed_server_id=failed.server_id,
        unaffected=unaffected,
        initially_unplaced=unplaced,
        state=agent.state,
        affected=tuple(decision.affected_workloads) if decision else tuple(affected),
        recovered=tuple(decision.recovered_workloads) if decision else (),
    )
    return {
        "seed": seed,
        "generated": len(workloads),
        "placed": len(placements),
        "unplaced": unplaced,
        "initial_peak": initial_peak,
        "failed_server": failed.server_id,
        "affected": len(decision.affected_workloads) if decision else len(affected),
        "recovered": len(decision.recovered_workloads) if decision else 0,
        "unrecovered": len(decision.unrecovered_workloads) if decision else len(affected),
        "state": agent.state.name,
        "reason": decision.reason if decision else error,
        "final_peak": peak_cpu(servers),
        "final_cpu": metrics.total_cpu_utilization,
        "final_memory": metrics.total_memory_utilization,
        "active": metrics.active_server_count,
        "inactive": metrics.inactive_server_count,
        "violations": violations,
        "free_cpu": _free_cpu(before, failed.server_id),
        "free_memory": _free_memory(before, failed.server_id),
        "affected_cpu": _assigned_cpu(before, failed.server_id),
        "affected_memory": _assigned_memory(before, failed.server_id),
    }


def main() -> None:
    rows = [evaluate_seed(seed) for seed in SEEDS]
    print(
        "scenario: "
        f"{WORKLOAD_COUNT} workloads, cpu {MIN_CPU}-{MAX_CPU}, memory {MIN_MEMORY}-{MAX_MEMORY}; "
        f"{SERVER_COUNT} servers, cpu capacity {CPU_CAPACITY}, memory capacity {MEMORY_CAPACITY}"
    )
    print(
        "failure rule: operational server with the most workloads, "
        "then the most assigned CPU, then the earliest server id"
    )
    print()
    header = (
        f"{'seed':>4} {'gen':>4} {'placed':>6} {'unplaced':>8} {'init_peak':>9} "
        f"{'failed':>10} {'affected':>8} {'recovered':>9} {'unrecov':>8} "
        f"{'state':>10} {'final_peak':>10} {'cpu':>8} {'memory':>8} {'active':>6} {'inactive':>8} {'checks':>7}"
    )
    print(header)
    for row in rows:
        unplaced = row["unplaced"]
        assert isinstance(unplaced, list)
        print(
            f"{row['seed']:>4} {row['generated']:>4} {row['placed']:>6} {len(unplaced):>8} "
            f"{row['initial_peak']:>9.4f} {row['failed_server']:>10} {row['affected']:>8} "
            f"{row['recovered']:>9} {row['unrecovered']:>8} {row['state']:>10} "
            f"{row['final_peak']:>10.4f} {row['final_cpu']:>8.4f} {row['final_memory']:>8.4f} "
            f"{row['active']:>6} {row['inactive']:>8} {('ok' if not row['violations'] else 'FAIL'):>7}"
        )
        if unplaced:
            print(f"     initial unplaced: {unplaced}")

    print()
    print(f"scenarios:           {len(rows)}")
    print(f"successful recoveries: {sum(row['state'] == AgentState.COMPLETED.name for row in rows)}")
    print(f"failed recoveries:   {sum(row['state'] == AgentState.FAILED.name for row in rows)}")
    affected_total = sum(int(row["affected"]) for row in rows)  # type: ignore[arg-type]
    recovered_total = sum(int(row["recovered"]) for row in rows)  # type: ignore[arg-type]
    unrecovered_total = sum(int(row["unrecovered"]) for row in rows)  # type: ignore[arg-type]
    print(f"affected workloads:  {affected_total}")
    print(f"recovered workloads: {recovered_total}")
    print(f"unrecovered workloads: {unrecovered_total}")
    violations = [item for row in rows for item in row["violations"]]  # type: ignore[union-attr]
    print(f"constraint violations: {len(violations)}")
    for item in violations:
        print(f"  {item}")

    print()
    for row in rows:
        if row["state"] != AgentState.FAILED.name:
            continue
        print(
            f"seed {row['seed']}: recovery failed ({row['reason']}). "
            f"Affected CPU {row['affected_cpu']:.4f} and memory {row['affected_memory']:.4f} "
            f"needed a place on the remaining servers, which had "
            f"{row['free_cpu']:.4f} free CPU and {row['free_memory']:.4f} free memory "
            "before the failure. The agent did not apply a partial move."
        )


def _violations(
    servers: list[Server],
    *,
    failed_server_id: str,
    unaffected: dict[str, str],
    initially_unplaced: list[str],
    state: AgentState,
    affected: tuple[str, ...],
    recovered: tuple[str, ...],
) -> list[str]:
    problems: list[str] = []
    locations: dict[str, list[str]] = {}
    for server in servers:
        if _cpu_used(server) > server.cpu_capacity + 1e-9:
            problems.append(f"{server.server_id} exceeds CPU capacity")
        if _memory_used(server) > server.memory_capacity + 1e-9:
            problems.append(f"{server.server_id} exceeds memory capacity")
        for workload in server.workloads:
            locations.setdefault(workload.workload_id, []).append(server.server_id)
            if workload.workload_id in recovered and server.server_id == failed_server_id:
                problems.append(f"{workload.workload_id} remains on failed server {failed_server_id}")
    for workload_id, hosts in locations.items():
        if len(hosts) > 1:
            problems.append(f"{workload_id} is on {hosts}")
    for workload_id, server_id in unaffected.items():
        if locations.get(workload_id) != [server_id]:
            problems.append(f"unaffected {workload_id} is no longer on {server_id}")
    for workload_id in initially_unplaced:
        if workload_id in locations:
            problems.append(f"initially unplaced {workload_id} was assigned during recovery")
    if state is AgentState.COMPLETED:
        if set(recovered) != set(affected):
            problems.append("COMPLETED does not recover every affected workload")
        for workload_id in affected:
            hosts = locations.get(workload_id, [])
            if hosts == [failed_server_id] or len(hosts) != 1:
                problems.append(f"COMPLETED left {workload_id} unrestored")
    return problems


def _snapshot(servers: list[Server]) -> dict[str, tuple[float, float, float, float]]:
    return {
        server.server_id: (
            _cpu_used(server),
            _memory_used(server),
            server.cpu_capacity,
            server.memory_capacity,
        )
        for server in servers
    }


def _free_cpu(snapshot: dict[str, tuple[float, float, float, float]], failed_server_id: str) -> float:
    return sum(capacity - used for server_id, (used, _, capacity, _) in snapshot.items() if server_id != failed_server_id)


def _free_memory(snapshot: dict[str, tuple[float, float, float, float]], failed_server_id: str) -> float:
    return sum(
        capacity - used for server_id, (_, used, _, capacity) in snapshot.items() if server_id != failed_server_id
    )


def _assigned_cpu(snapshot: dict[str, tuple[float, float, float, float]], server_id: str) -> float:
    return snapshot[server_id][0]


def _assigned_memory(snapshot: dict[str, tuple[float, float, float, float]], server_id: str) -> float:
    return snapshot[server_id][1]


def _cpu_used(server: Server) -> float:
    return sum(workload.cpu_required for workload in server.workloads)


def _memory_used(server: Server) -> float:
    return sum(workload.memory_required for workload in server.workloads)


if __name__ == "__main__":
    main()
