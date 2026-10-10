"""Read-only views of the current data center.

These functions read racks, servers, and workloads that already exist.
They do not place work, change server status, or call the optimizer,
predictor, agent, or local model.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from models.datacenter import DataCenter
from models.rack import Rack
from models.server import Server, ServerStatus
from models.workload import Workload
from simulator.metrics import calculate_system_metrics


@dataclass(frozen=True)
class WorkloadSnapshot:
    """Demand fields copied from one workload."""

    workload_id: str
    cpu_required: float
    memory_required: float
    latency_requirement: float
    priority: int


@dataclass(frozen=True)
class ServerSnapshot:
    """One server's status and the workloads currently assigned to it."""

    server_id: str
    rack_id: str
    status: str
    cpu_capacity: float
    memory_capacity: float
    cpu_used: float
    memory_used: float
    cpu_utilization: float
    memory_utilization: float
    temperature: float
    power_consumption: float
    workloads: tuple[WorkloadSnapshot, ...]


@dataclass(frozen=True)
class RackSnapshot:
    """One rack and the server ids attached to it, in list order."""

    rack_id: str
    cooling_capacity: float
    server_ids: tuple[str, ...]


@dataclass(frozen=True)
class SystemState:
    """A copy of the facility. Unplaced workloads are not on any server."""

    racks: tuple[RackSnapshot, ...]
    servers: tuple[ServerSnapshot, ...]
    unplaced_workloads: tuple[WorkloadSnapshot, ...]


@dataclass(frozen=True)
class WorkloadPlacement:
    """Where one workload is running, or that it is present but unplaced."""

    workload: WorkloadSnapshot
    placed: bool
    server_id: str | None
    rack_id: str | None


@dataclass(frozen=True)
class ResourceUtilization:
    """Facility measurements from calculate_system_metrics, plus capacity.

    Utilization, power, temperature, and counts come from that function.
    Capacity and used amounts are sums of the same operational servers.
    Inactive servers stay in the counts and out of the utilization totals.
    """

    total_cpu_utilization: float
    total_memory_utilization: float
    total_power_consumption: float
    average_server_temperature: float
    workload_count: int
    active_server_count: int
    inactive_server_count: int
    operational_cpu_capacity: float
    operational_memory_capacity: float
    operational_cpu_used: float
    operational_memory_used: float


@dataclass(frozen=True)
class ToolParameter:
    """One argument a future caller must supply. The data center is not one of them."""

    name: str
    type_name: str
    description: str


@dataclass(frozen=True)
class ToolSpec:
    """A deterministic tool. function is the Python callable, not a model."""

    name: str
    description: str
    parameters: tuple[ToolParameter, ...]
    function: Callable[..., object]


def get_system_state(datacenter: DataCenter) -> SystemState:
    """Return racks, servers, and workloads that are not placed on a server."""
    servers = _servers(datacenter)
    placed: dict[str, Workload] = {}
    for server in servers:
        for workload in server.workloads:
            if workload.workload_id in placed:
                raise ValueError(f"workload {workload.workload_id!r} is assigned to more than one server")
            placed[workload.workload_id] = workload
    unplaced: list[WorkloadSnapshot] = []
    seen: set[str] = set()
    for workload in _environment_workloads(datacenter):
        assigned = placed.get(workload.workload_id)
        if assigned is not None:
            if not _same_demand(assigned, workload):
                raise ValueError(
                    f"workload {workload.workload_id!r} disagrees with the data center workload list"
                )
            continue
        if workload.workload_id in seen:
            raise ValueError(f"workload {workload.workload_id!r} is not unique")
        seen.add(workload.workload_id)
        unplaced.append(_workload_snapshot(workload))
    return SystemState(
        racks=tuple(_rack_snapshot(rack) for rack in _racks(datacenter)),
        servers=tuple(_server_snapshot(server) for server in servers),
        unplaced_workloads=tuple(unplaced),
    )


def get_server_status(datacenter: DataCenter, server_id: str) -> ServerSnapshot:
    """Return the server with this id, or raise if it is missing or duplicated."""
    return _server_snapshot(_server_by_id(datacenter, server_id))


def get_workload_placement(datacenter: DataCenter, workload_id: str) -> WorkloadPlacement:
    """Return the server hosting this workload, or the unplaced environment entry."""
    _require_id("workload_id", workload_id)
    hosts = [
        server
        for server in _servers(datacenter)
        if any(workload.workload_id == workload_id for workload in server.workloads)
    ]
    if len(hosts) > 1:
        raise ValueError(f"workload {workload_id!r} is assigned to more than one server")
    environment = [
        workload for workload in _environment_workloads(datacenter) if workload.workload_id == workload_id
    ]
    if len(environment) > 1:
        raise ValueError(f"workload {workload_id!r} is not unique")
    if hosts:
        placed = _workload_on_server(hosts[0], workload_id)
        if environment and not _same_demand(placed, environment[0]):
            raise ValueError(f"workload {workload_id!r} disagrees with the data center workload list")
        return WorkloadPlacement(
            workload=_workload_snapshot(placed),
            placed=True,
            server_id=hosts[0].server_id,
            rack_id=hosts[0].rack_id,
        )
    if not environment:
        raise ValueError(f"workload {workload_id!r} was not found")
    return WorkloadPlacement(
        workload=_workload_snapshot(environment[0]),
        placed=False,
        server_id=None,
        rack_id=None,
    )


def get_resource_utilization(datacenter: DataCenter) -> ResourceUtilization:
    """Measure the current servers. This does not advance the simulation."""
    servers = _servers(datacenter)
    metrics = calculate_system_metrics(servers)
    operational = [server for server in servers if server.status is ServerStatus.OPERATIONAL]
    return ResourceUtilization(
        total_cpu_utilization=metrics.total_cpu_utilization,
        total_memory_utilization=metrics.total_memory_utilization,
        total_power_consumption=metrics.total_power_consumption,
        average_server_temperature=metrics.average_server_temperature,
        workload_count=metrics.workload_count,
        active_server_count=metrics.active_server_count,
        inactive_server_count=metrics.inactive_server_count,
        operational_cpu_capacity=sum(server.cpu_capacity for server in operational),
        operational_memory_capacity=sum(server.memory_capacity for server in operational),
        operational_cpu_used=sum(workload.cpu_required for server in operational for workload in server.workloads),
        operational_memory_used=sum(
            workload.memory_required for server in operational for workload in server.workloads
        ),
    )


def tool_registry() -> tuple[ToolSpec, ...]:
    """Name, description, arguments, and callable for each read-only tool."""
    return (
        ToolSpec(
            name="get_system_state",
            description="Return the current racks, servers, assigned workloads, and unplaced workloads.",
            parameters=(),
            function=get_system_state,
        ),
        ToolSpec(
            name="get_server_status",
            description="Return status, capacity, utilization, and assigned workloads for one server.",
            parameters=(
                ToolParameter("server_id", "str", "Id of a server in the data center."),
            ),
            function=get_server_status,
        ),
        ToolSpec(
            name="get_workload_placement",
            description="Return the server and rack hosting one workload, or that it is unplaced.",
            parameters=(
                ToolParameter("workload_id", "str", "Id of a workload in the data center."),
            ),
            function=get_workload_placement,
        ),
        ToolSpec(
            name="get_resource_utilization",
            description="Return current utilization, power, temperature, counts, and operational capacity.",
            parameters=(),
            function=get_resource_utilization,
        ),
    )


def _server_snapshot(server: Server) -> ServerSnapshot:
    workloads = tuple(_workload_snapshot(workload) for workload in server.workloads)
    return ServerSnapshot(
        server_id=server.server_id,
        rack_id=server.rack_id,
        status=server.status.value,
        cpu_capacity=server.cpu_capacity,
        memory_capacity=server.memory_capacity,
        cpu_used=sum(workload.cpu_required for workload in workloads),
        memory_used=sum(workload.memory_required for workload in workloads),
        cpu_utilization=server.cpu_utilization(),
        memory_utilization=server.memory_utilization(),
        temperature=server.temperature,
        power_consumption=server.power_consumption,
        workloads=workloads,
    )


def _rack_snapshot(rack: Rack) -> RackSnapshot:
    return RackSnapshot(
        rack_id=rack.rack_id,
        cooling_capacity=rack.cooling_capacity,
        server_ids=tuple(server.server_id for server in rack.servers),
    )


def _workload_snapshot(workload: Workload) -> WorkloadSnapshot:
    return WorkloadSnapshot(
        workload_id=workload.workload_id,
        cpu_required=workload.cpu_required,
        memory_required=workload.memory_required,
        latency_requirement=workload.latency_requirement,
        priority=workload.priority,
    )


def _server_by_id(datacenter: DataCenter, server_id: str) -> Server:
    _require_id("server_id", server_id)
    matches = [server for server in _servers(datacenter) if server.server_id == server_id]
    if not matches:
        raise ValueError(f"server {server_id!r} was not found")
    return matches[0]


def _workload_on_server(server: Server, workload_id: str) -> Workload:
    matches = [workload for workload in server.workloads if workload.workload_id == workload_id]
    if len(matches) != 1:
        raise ValueError(f"workload {workload_id!r} is not unique on server {server.server_id!r}")
    return matches[0]


def _same_demand(left: Workload, right: Workload) -> bool:
    return (
        left.cpu_required == right.cpu_required
        and left.memory_required == right.memory_required
        and left.latency_requirement == right.latency_requirement
        and left.priority == right.priority
    )


def _racks(datacenter: DataCenter) -> list[Rack]:
    _require_datacenter(datacenter)
    if any(not isinstance(rack, Rack) for rack in datacenter.racks):
        raise TypeError("datacenter racks must be Rack instances")
    return list(datacenter.racks)


def _servers(datacenter: DataCenter) -> list[Server]:
    servers: list[Server] = []
    seen: set[str] = set()
    for rack in _racks(datacenter):
        if any(not isinstance(server, Server) for server in rack.servers):
            raise TypeError(f"rack {rack.rack_id!r} must contain Server instances")
        for server in rack.servers:
            if server.server_id in seen:
                raise ValueError(f"server {server.server_id!r} is not unique")
            seen.add(server.server_id)
            servers.append(server)
    return servers


def _environment_workloads(datacenter: DataCenter) -> Sequence[Workload]:
    if any(not isinstance(workload, Workload) for workload in datacenter.workloads):
        raise TypeError("datacenter workloads must be Workload instances")
    return datacenter.workloads


def _require_datacenter(datacenter: DataCenter) -> None:
    if not isinstance(datacenter, DataCenter):
        raise TypeError("datacenter must be a DataCenter")


def _require_id(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
