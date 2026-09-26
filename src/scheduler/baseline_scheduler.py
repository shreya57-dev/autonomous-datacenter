"""Greedy baseline placement: lowest current CPU utilization."""

from collections.abc import Sequence
from dataclasses import dataclass

from models.server import Server
from models.workload import Workload


@dataclass(frozen=True)
class ScheduleResult:
    """Where each workload was placed, in input order.

    placements maps workload id to server id.
    unplaced_workloads lists ids that fit on no server.
    """

    placements: dict[str, str]
    unplaced_workloads: list[str]


class BaselineScheduler:
    """Place workloads one by one onto the least-utilized feasible server.

    Feasibility comes only from Server.can_host. When CPU utilization is tied,
    the earlier server in the given list is chosen. No global search is done.
    """

    def schedule(self, workloads: Sequence[Workload], servers: Sequence[Server]) -> ScheduleResult:
        """Allocate each workload in order, or record it as unplaced."""
        _validate_inputs(workloads, servers)
        placements: dict[str, str] = {}
        unplaced: list[str] = []
        for workload in workloads:
            server = _least_loaded_feasible_server(workload, servers)
            if server is None:
                unplaced.append(workload.workload_id)
                continue
            server.allocate(workload)
            placements[workload.workload_id] = server.server_id
        return ScheduleResult(placements=placements, unplaced_workloads=unplaced)


def _least_loaded_feasible_server(workload: Workload, servers: Sequence[Server]) -> Server | None:
    feasible = [
        (index, server)
        for index, server in enumerate(servers)
        if server.can_host(workload)
    ]
    if not feasible:
        return None
    _, selected = min(feasible, key=lambda item: (item[1].cpu_utilization(), item[0]))
    return selected


def _validate_inputs(workloads: Sequence[Workload], servers: Sequence[Server]) -> None:
    if isinstance(workloads, (str, bytes)) or not isinstance(workloads, Sequence):
        raise ValueError("workloads must be a sequence of Workload")
    if isinstance(servers, (str, bytes)) or not isinstance(servers, Sequence):
        raise ValueError("servers must be a sequence of Server")
    if any(not isinstance(workload, Workload) for workload in workloads):
        raise ValueError("workloads must be a sequence of Workload")
    if any(not isinstance(server, Server) for server in servers):
        raise ValueError("servers must be a sequence of Server")

    workload_ids = [workload.workload_id for workload in workloads]
    if len(workload_ids) != len(set(workload_ids)):
        raise ValueError("workload ids must be unique")
    server_ids = [server.server_id for server in servers]
    if len(server_ids) != len(set(server_ids)):
        raise ValueError("server ids must be unique")

    already_assigned = {
        assigned.workload_id for server in servers for assigned in server.workloads
    }
    for workload_id in workload_ids:
        if workload_id in already_assigned:
            raise ValueError(f"workload {workload_id!r} is already assigned to a server")
