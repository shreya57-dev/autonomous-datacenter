"""Constraint-based placement using OR-Tools CP-SAT.

This proposes a placement. It does not call Server.allocate().

Integer scaling: 1.0 CPU or memory unit = RESOURCE_SCALE integer units.
Demand is rounded up and capacity is rounded down, so a solution cannot
exceed the real CPU or memory capacity.

Objective, in reported units:
    unplaced_count * REJECTION_PENALTY + peak CPU utilization

Peak CPU utilization is the highest utilization among operational servers
after the proposed placement, from 0 to 1. REJECTION_PENALTY is larger
than that maximum, so a feasible workload is placed before the solver
tries to lower the peak. This is not a physical energy model.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ortools.sat.python import cp_model

from models.server import Server, ServerStatus
from models.workload import Workload

# 1.0 CPU or memory = 100 integer units.
RESOURCE_SCALE = 100
# Peak utilization is an integer in [0, PEAK_SCALE], where PEAK_SCALE means 100%.
PEAK_SCALE = 10_000
# One unplaced workload costs more than any possible peak (the peak is at most 1).
REJECTION_PENALTY = 10


@dataclass(frozen=True)
class OptimizationResult:
    """A proposed placement. Servers are not modified.

    placements maps workload id to server id, in workload input order.
    unplaced_workloads lists ids the solver left unassigned, in input order.
    objective_value is unplaced_count * REJECTION_PENALTY + peak CPU utilization.
    solver_status is OPTIMAL, FEASIBLE, INFEASIBLE, or UNKNOWN.
    """

    placements: dict[str, str]
    unplaced_workloads: list[str]
    objective_value: float
    solver_status: str


class PlacementOptimizer:
    """Choose a placement that respects capacity and lowers peak CPU use."""

    def __init__(self, time_limit_seconds: float = 10.0) -> None:
        if isinstance(time_limit_seconds, bool) or not isinstance(time_limit_seconds, (int, float)):
            raise ValueError("time_limit_seconds must be a number")
        if time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        self.time_limit_seconds = float(time_limit_seconds)

    def optimize(self, workloads: Sequence[Workload], servers: Sequence[Server]) -> OptimizationResult:
        """Return a proposed placement for these workloads and servers."""
        _validate_inputs(workloads, servers)
        return _solve(list(workloads), list(servers), self.time_limit_seconds)


def _solve(workloads: list[Workload], servers: list[Server], time_limit_seconds: float) -> OptimizationResult:
    model = cp_model.CpModel()

    # x[workload, server] = 1 means that workload is assigned to that server.
    assigned = {
        (workload.workload_id, server.server_id): model.NewBoolVar(
            f"x_{workload.workload_id}_{server.server_id}"
        )
        for workload in workloads
        for server in servers
    }
    rejected = []
    for workload in workloads:
        choices = [assigned[workload.workload_id, server.server_id] for server in servers]
        is_rejected = model.NewBoolVar(f"rejected_{workload.workload_id}")
        rejected.append(is_rejected)
        if choices:
            # At most one server. The rejected flag is 1 when none is chosen.
            model.Add(sum(choices) <= 1)
            model.Add(sum(choices) + is_rejected == 1)
        else:
            model.Add(is_rejected == 1)

    for server in servers:
        if server.status is ServerStatus.OPERATIONAL:
            continue
        # Inactive servers are not candidates.
        for workload in workloads:
            model.Add(assigned[workload.workload_id, server.server_id] == 0)

    # Peak is the maximum operational-server CPU utilization, in PEAK_SCALE units.
    peak = model.NewIntVar(0, PEAK_SCALE, "peak_cpu")
    operational = [server for server in servers if server.status is ServerStatus.OPERATIONAL]
    if not operational:
        model.Add(peak == 0)
    for server in operational:
        _add_server_constraints(model, server, workloads, assigned, peak)

    # Rejection dominates peak, and peak dominates this server-index tie-break.
    # The tie-break only picks among placements with the same peak.
    max_tie = len(workloads) * max(len(servers) - 1, 0)
    peak_weight = max_tie + 1
    rejection_weight = peak_weight * PEAK_SCALE + 1
    tie_break = [
        index * assigned[workload.workload_id, server.server_id]
        for index, server in enumerate(servers)
        for workload in workloads
    ]
    model.Minimize(
        rejection_weight * sum(rejected) + peak_weight * peak + sum(tie_break)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 1
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"placement model could not be solved (status={status.name})")

    placements: dict[str, str] = {}
    unplaced: list[str] = []
    for workload in workloads:
        selected = [
            server.server_id
            for server in servers
            if solver.Value(assigned[workload.workload_id, server.server_id]) == 1
        ]
        if selected:
            placements[workload.workload_id] = selected[0]
        else:
            unplaced.append(workload.workload_id)

    objective_value = len(unplaced) * REJECTION_PENALTY + solver.Value(peak) / PEAK_SCALE
    return OptimizationResult(
        placements=placements,
        unplaced_workloads=unplaced,
        objective_value=objective_value,
        solver_status=status.name,
    )


def _add_server_constraints(
    model: cp_model.CpModel,
    server: Server,
    workloads: list[Workload],
    assigned: dict[tuple[str, str], cp_model.IntVar],
    peak: cp_model.IntVar,
) -> None:
    cpu_capacity = _scale_capacity(server.cpu_capacity)
    memory_capacity = _scale_capacity(server.memory_capacity)
    if cpu_capacity <= 0 or memory_capacity <= 0:
        raise ValueError(f"server {server.server_id!r} capacity is too small to scale")
    if server.cpu_utilization() > 1 or server.memory_utilization() > 1:
        raise ValueError(f"server {server.server_id!r} is already over capacity")

    # Keep CPU and memory already on the server. Clamp only the rounding gap.
    existing_cpu = min(_scale_demand(server.cpu_utilization() * server.cpu_capacity), cpu_capacity)
    existing_memory = min(
        _scale_demand(server.memory_utilization() * server.memory_capacity),
        memory_capacity,
    )
    new_cpu = [
        _scale_demand(workload.cpu_required) * assigned[workload.workload_id, server.server_id]
        for workload in workloads
    ]
    new_memory = [
        _scale_demand(workload.memory_required) * assigned[workload.workload_id, server.server_id]
        for workload in workloads
    ]
    if new_cpu:
        # Scaled demand is rounded up and capacity down, so this cannot oversubscribe.
        model.Add(sum(new_cpu) <= cpu_capacity - existing_cpu)
        model.Add(sum(new_memory) <= memory_capacity - existing_memory)
        model.Add((existing_cpu + sum(new_cpu)) * PEAK_SCALE <= peak * cpu_capacity)
    else:
        model.Add(existing_cpu * PEAK_SCALE <= peak * cpu_capacity)


def _scale_demand(value: float) -> int:
    """Round a requirement up so the constraint never underestimates it."""
    _require_non_negative_number(value)
    return math.ceil(value * RESOURCE_SCALE - 1e-9)


def _scale_capacity(value: float) -> int:
    """Round a capacity down so the constraint never overestimates it."""
    _require_non_negative_number(value)
    return math.floor(value * RESOURCE_SCALE + 1e-9)


def _require_non_negative_number(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("resource values must be numbers")
    if value < 0:
        raise ValueError("resource values cannot be negative")


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
