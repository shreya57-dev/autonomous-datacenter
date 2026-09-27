"""Show one workload arrival handled by the autonomous agent.

This demonstrates the observe-to-verify loop. It is not a benchmark.
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
from optimizer.placement_optimizer import PlacementOptimizer
from predictor.demand_predictor import DemandPredictor
from simulator.metrics import calculate_system_metrics

WORKLOAD_ID = "job-1"
CURRENT_CPU = 4.0
HISTORY = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def build_datacenter() -> DataCenter:
    servers = [
        Server(
            server_id=server_id,
            rack_id="rack-1",
            cpu_capacity=10,
            memory_capacity=16,
            temperature=20,
            power_consumption=0,
        )
        for server_id in ("server-1", "server-2")
    ]
    return DataCenter(racks=[Rack(rack_id="rack-1", cooling_capacity=1, servers=servers)])


def main() -> None:
    datacenter = build_datacenter()
    agent = AutonomousAgent(
        datacenter=datacenter,
        predictor=DemandPredictor(window=3),
        optimizer=PlacementOptimizer(),
    )
    bus = EventBus()
    agent.subscribe(bus)
    bus.publish(
        Event(
            event_type=EventType.WORKLOAD_ARRIVAL,
            tick=1,
            source_id=WORKLOAD_ID,
            payload={
                "cpu_required": CURRENT_CPU,
                "memory_required": 2,
                "latency_requirement": 50,
                "priority": 1,
                "history": HISTORY,
            },
        )
    )
    decision = agent.decision
    if decision is None:
        raise RuntimeError("agent did not record a decision")

    print(" -> ".join(state.name for state in agent.state_history))
    print(f"workload:        {decision.workload_id}")
    print(f"current cpu:     {CURRENT_CPU:.4f}")
    predicted = "unavailable" if decision.predicted_cpu is None else f"{decision.predicted_cpu:.4f}"
    print(f"predicted cpu:   {predicted}")
    print(f"planning cpu:    {decision.planning_cpu:.4f}")
    print(f"selected server: {decision.server_id}")
    print(f"final state:     {agent.state.name}")

    servers = [server for rack in datacenter.racks for server in rack.servers]
    metrics = calculate_system_metrics(servers)
    print(f"cpu utilization: {metrics.total_cpu_utilization:.4f}")
    print(f"memory util:     {metrics.total_memory_utilization:.4f}")
    print(f"power:           {metrics.total_power_consumption:.4f}")
    print(f"temperature:     {metrics.average_server_temperature:.4f}")
    print(f"workload count:  {metrics.workload_count}")

    selected = next(server for server in servers if server.server_id == decision.server_id)
    present = any(workload.workload_id == WORKLOAD_ID for workload in selected.workloads)
    print(f"present on {decision.server_id}: {present}")


if __name__ == "__main__":
    main()
