"""Place one arriving workload: observe, predict, plan, evaluate, act, verify.

Prediction is optional. Planning uses a copy of the data center. The real
servers change only in the acting step, through Server.allocate().
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum

from models.datacenter import DataCenter
from models.rack import Rack
from models.server import Server
from models.workload import Workload
from optimizer.placement_optimizer import PlacementOptimizer
from predictor.demand_predictor import DemandPrediction, DemandPredictor, build_planning_workloads
from simulator.engine import SimulationEngine
from simulator.metrics import calculate_system_metrics

from .event_bus import EventBus
from .events import Event, EventType


class AgentState(Enum):
    """Where the agent is in one workload-arrival decision."""

    IDLE = "idle"
    OBSERVING = "observing"
    PLANNING = "planning"
    EVALUATING = "evaluating"
    ACTING = "acting"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class AgentDecision:
    """What the agent decided for one arrival. It is not an event."""

    workload_id: str
    server_id: str | None
    predicted_cpu: float | None
    planning_cpu: float
    state: AgentState
    reason: str


class AutonomousAgent:
    """Orchestrate existing components for a WORKLOAD_ARRIVAL event.

    Forecasting happens while observing, because the state list has no
    separate predicting step. The optimizer is not reimplemented here.
    """

    def __init__(
        self,
        datacenter: DataCenter,
        predictor: DemandPredictor,
        optimizer: PlacementOptimizer,
        engine_factory: Callable[[DataCenter], SimulationEngine] | None = None,
    ) -> None:
        if not isinstance(datacenter, DataCenter):
            raise ValueError("datacenter must be a DataCenter")
        if not isinstance(predictor, DemandPredictor):
            raise ValueError("predictor must be a DemandPredictor")
        if not isinstance(optimizer, PlacementOptimizer):
            raise ValueError("optimizer must be a PlacementOptimizer")
        self._datacenter = datacenter
        self._predictor = predictor
        self._optimizer = optimizer
        self._engine_factory = engine_factory or SimulationEngine
        self._state = AgentState.IDLE
        self._history: list[AgentState] = [AgentState.IDLE]
        self._decision: AgentDecision | None = None

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def state_history(self) -> tuple[AgentState, ...]:
        return tuple(self._history)

    @property
    def decision(self) -> AgentDecision | None:
        return self._decision

    def subscribe(self, bus: EventBus) -> None:
        """Ask the bus to deliver workload arrivals. The bus does not decide."""
        bus.subscribe(EventType.WORKLOAD_ARRIVAL, self.handle_workload_arrival)

    def handle_workload_arrival(self, event: Event) -> AgentDecision:
        """Run one arrival. Failures stay visible and end in FAILED."""
        try:
            return self._decide(event)
        except Exception:
            if self._state is not AgentState.FAILED:
                self._set_state(AgentState.FAILED)
            raise

    def _decide(self, event: Event) -> AgentDecision:
        self._set_state(AgentState.OBSERVING)
        if event.event_type is not EventType.WORKLOAD_ARRIVAL:
            self._fail(event.source_id, "event is not a workload arrival")
        workload, history = self._observe(event)
        predicted = self._predict(history)
        planning_cpu = workload.cpu_required if predicted is None else max(workload.cpu_required, predicted)

        self._set_state(AgentState.PLANNING)
        server_id = self._plan(workload, planning_cpu, predicted)

        self._set_state(AgentState.EVALUATING)
        self._evaluate(workload, server_id, planning_cpu, predicted)

        self._set_state(AgentState.ACTING)
        server = self._server_by_id(server_id)
        try:
            server.allocate(workload)
        except Exception:
            self._fail(workload.workload_id, "allocation failed", predicted=predicted, planning_cpu=planning_cpu)

        self._set_state(AgentState.VERIFYING)
        try:
            self._verify(server, workload)
        except ValueError:
            self._rollback(server, workload)
            raise

        self._set_state(AgentState.COMPLETED)
        decision = AgentDecision(
            workload_id=workload.workload_id,
            server_id=server_id,
            predicted_cpu=predicted,
            planning_cpu=planning_cpu,
            state=AgentState.COMPLETED,
            reason="workload placed",
        )
        self._decision = decision
        return decision

    def _observe(self, event: Event) -> tuple[Workload, list[float]]:
        workload = _workload_from_event(event)
        if self._find_server(workload.workload_id) is not None:
            self._fail(workload.workload_id, "workload is already placed")
        return workload, _history_from_payload(event)

    def _predict(self, history: list[float]) -> float | None:
        if len(history) < self._predictor.window + 2:
            return None
        self._predictor.fit(history)
        return self._predictor.predict_next(history)

    def _plan(self, workload: Workload, planning_cpu: float, predicted: float | None) -> str:
        planning = build_planning_workloads(
            [workload],
            [DemandPrediction(workload.workload_id, planning_cpu)],
        )
        proposal = self._optimizer.optimize(planning, self._copied_servers())
        server_id = proposal.placements.get(workload.workload_id)
        if server_id is None:
            self._fail(
                workload.workload_id,
                "no feasible server",
                predicted=predicted,
                planning_cpu=planning_cpu,
            )
        return server_id

    def _evaluate(self, workload: Workload, server_id: str, planning_cpu: float, predicted: float | None) -> None:
        temporary, copies = self._temporary_datacenter()
        chosen = copies[server_id]
        planning_workload = replace(workload, cpu_required=planning_cpu)
        if not chosen.can_host(planning_workload):
            self._fail(
                workload.workload_id,
                "proposal violates capacity",
                predicted=predicted,
                planning_cpu=planning_cpu,
            )
        chosen.allocate(planning_workload)
        self._engine_factory(temporary).advance_tick()
        if chosen.cpu_utilization() > 1 or chosen.memory_utilization() > 1:
            self._fail(
                workload.workload_id,
                "evaluation left the copy over capacity",
                predicted=predicted,
                planning_cpu=planning_cpu,
            )

    def _verify(self, server: Server, workload: Workload) -> None:
        placed = any(item.workload_id == workload.workload_id for item in server.workloads)
        if not placed:
            raise ValueError("workload missing after allocation")
        if server.cpu_utilization() > 1 or server.memory_utilization() > 1:
            raise ValueError("server over capacity after allocation")
        metrics = calculate_system_metrics(self._servers())
        if metrics.workload_count < 1:
            raise ValueError("metrics did not see the placed workload")

    def _rollback(self, server: Server, workload: Workload) -> None:
        if any(item.workload_id == workload.workload_id for item in server.workloads):
            server.release(workload)

    def _fail(
        self,
        workload_id: str,
        reason: str,
        *,
        predicted: float | None = None,
        planning_cpu: float = 0.0,
    ) -> None:
        self._set_state(AgentState.FAILED)
        self._decision = AgentDecision(
            workload_id=workload_id,
            server_id=None,
            predicted_cpu=predicted,
            planning_cpu=planning_cpu,
            state=AgentState.FAILED,
            reason=reason,
        )
        raise ValueError(reason)

    def _set_state(self, state: AgentState) -> None:
        self._state = state
        self._history.append(state)

    def _servers(self) -> list[Server]:
        return [server for rack in self._datacenter.racks for server in rack.servers]

    def _copied_servers(self) -> list[Server]:
        return list(self._temporary_datacenter()[1].values())

    def _temporary_datacenter(self) -> tuple[DataCenter, dict[str, Server]]:
        copies: dict[str, Server] = {}
        racks: list[Rack] = []
        for rack in self._datacenter.racks:
            rack_servers = [_copy_server(server) for server in rack.servers]
            for server in rack_servers:
                copies[server.server_id] = server
            racks.append(Rack(rack_id=rack.rack_id, cooling_capacity=rack.cooling_capacity, servers=rack_servers))
        return DataCenter(racks=racks), copies

    def _server_by_id(self, server_id: str) -> Server:
        for server in self._servers():
            if server.server_id == server_id:
                return server
        self._fail(server_id, f"server {server_id!r} is not in the data center")
        raise AssertionError("unreachable")

    def _find_server(self, workload_id: str) -> Server | None:
        for server in self._servers():
            if any(workload.workload_id == workload_id for workload in server.workloads):
                return server
        return None


def _copy_server(server: Server) -> Server:
    return Server(
        server_id=server.server_id,
        rack_id=server.rack_id,
        cpu_capacity=server.cpu_capacity,
        memory_capacity=server.memory_capacity,
        temperature=server.temperature,
        power_consumption=server.power_consumption,
        status=server.status,
        workloads=list(server.workloads),
    )


def _workload_from_event(event: Event) -> Workload:
    payload = event.payload
    required = ("cpu_required", "memory_required", "latency_requirement", "priority")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"payload missing {missing[0]}")
    return Workload(
        workload_id=event.source_id,
        cpu_required=payload["cpu_required"],  # type: ignore[arg-type]
        memory_required=payload["memory_required"],  # type: ignore[arg-type]
        latency_requirement=payload["latency_requirement"],  # type: ignore[arg-type]
        priority=payload["priority"],  # type: ignore[arg-type]
    )


def _history_from_payload(event: Event) -> list[float]:
    raw = event.payload.get("history")
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("history must be a sequence of numbers")
    return [float(value) for value in raw]
