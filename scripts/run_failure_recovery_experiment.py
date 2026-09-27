"""Show one server failure recovered by the autonomous agent.

This demonstrates fault recovery. It is not a benchmark.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.autonomous_agent import AutonomousAgent
from agent.event_bus import EventBus
from agent.events import Event, EventType
from models.datacenter import DataCenter
from models.rack import Rack
from models.server import Server
from models.workload import Workload
from optimizer.placement_optimizer import PlacementOptimizer
from predictor.demand_predictor import DemandPredictor
from simulator.metrics import calculate_system_metrics


def build_server(server_id: str) -> Server:
    return Server(
        server_id=server_id,
        rack_id="rack-1",
        cpu_capacity=10,
        memory_capacity=16,
        temperature=20,
        power_consumption=0,
    )


def build_workload(workload_id: str, cpu_required: float) -> Workload:
    return Workload(
        workload_id=workload_id,
        cpu_required=cpu_required,
        memory_required=1,
        latency_requirement=50,
        priority=1,
    )


def print_assignments(servers: list[Server]) -> None:
    for server in servers:
        names = [workload.workload_id for workload in server.workloads]
        print(f"  {server.server_id} ({server.status.value}): {names}")


def main() -> None:
    server_1 = build_server("server-1")
    server_2 = build_server("server-2")
    server_3 = build_server("server-3")
    job_keep = build_workload("job-keep", 6)
    job_other = build_workload("job-other", 4)
    job_a = build_workload("job-a", 5)
    job_b = build_workload("job-b", 4)
    server_1.allocate(job_keep)
    server_3.allocate(job_other)
    server_2.allocate(job_a)
    server_2.allocate(job_b)
    datacenter = DataCenter(
        racks=[Rack(rack_id="rack-1", cooling_capacity=1, servers=[server_1, server_2, server_3])]
    )
    servers = [server_1, server_2, server_3]

    print("initial placement:")
    print_assignments(servers)

    agent = AutonomousAgent(
        datacenter=datacenter,
        predictor=DemandPredictor(window=3),
        optimizer=PlacementOptimizer(),
    )
    bus = EventBus()
    agent.subscribe(bus)
    bus.publish(
        Event(
            event_type=EventType.SERVER_FAILURE,
            tick=10,
            source_id="server-2",
            payload={"server_id": "server-2", "reason": "hardware_failure"},
        )
    )
    decision = agent.decision
    if decision is None:
        raise RuntimeError("agent did not record a decision")

    print(" -> ".join(state.name for state in agent.state_history))
    print(f"failed server:    {decision.failed_server_id}")
    print(f"affected:         {list(decision.affected_workloads)}")
    print(f"recovered:        {list(decision.recovered_workloads)}")
    print(f"unrecovered:      {list(decision.unrecovered_workloads)}")
    for workload_id, server_id in decision.placements:
        print(f"placement:        {workload_id} -> {server_id}")
    print(f"final state:      {agent.state.name}")

    print("final placement:")
    print_assignments(servers)
    metrics = calculate_system_metrics(servers)
    print(f"cpu utilization:  {metrics.total_cpu_utilization:.4f}")
    print(f"memory util:      {metrics.total_memory_utilization:.4f}")
    print(f"power:            {metrics.total_power_consumption:.4f}")
    print(f"temperature:      {metrics.average_server_temperature:.4f}")
    print(f"workload count:   {metrics.workload_count}")
    print(f"active servers:   {metrics.active_server_count}")
    print(f"inactive servers: {metrics.inactive_server_count}")

    unaffected = server_1.workloads[0] is job_keep and server_3.workloads[0] is job_other
    recovered = {workload.workload_id for workload in server_1.workloads + server_3.workloads}
    print(f"unaffected unchanged: {unaffected}")
    print(f"all affected recovered: {set(decision.affected_workloads) <= recovered and not decision.unrecovered_workloads}")


if __name__ == "__main__":
    main()
