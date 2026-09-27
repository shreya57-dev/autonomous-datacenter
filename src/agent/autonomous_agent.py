"""Place one arriving workload: observe, predict, plan, evaluate, act, verify.

Prediction is optional. Planning uses a copy of the data center. The real
servers change only in the acting step, through Server.allocate().
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum

from models.datacenter import DataCenter
from models.rack import Rack
from models.server import Server, ServerStatus
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
    """What the agent decided for one event. It is not an event."""

    workload_id: str
    server_id: str | None
    predicted_cpu: float | None
    planning_cpu: float
    state: AgentState
    reason: str
    failed_server_id: str | None = None
    affected_workloads: tuple[str, ...] = ()
    recovered_workloads: tuple[str, ...] = ()
    unrecovered_workloads: tuple[str, ...] = ()
    placements: tuple[tuple[str, str], ...] = ()
    planning_cpus: tuple[tuple[str, float], ...] = ()
    predicted_cpus: tuple[tuple[str, float | None], ...] = ()


class AutonomousAgent:
    """Orchestrate existing components for arrival and server-failure events.

    Forecasting happens while observing, because the state list has no
    separate predicting step. The optimizer is not reimplemented here.
    Recovery planning uses copies. The real data center changes only while acting.
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
        """Ask the bus to deliver events. The bus does not decide."""
        bus.subscribe(EventType.WORKLOAD_ARRIVAL, self.handle_workload_arrival)
        bus.subscribe(EventType.SERVER_FAILURE, self.handle_server_failure)

    def handle_workload_arrival(self, event: Event) -> AgentDecision:
        """Run one arrival. Failures stay visible and end in FAILED."""
        try:
            return self._decide(event)
        except Exception:
            if self._state is not AgentState.FAILED:
                self._set_state(AgentState.FAILED)
            raise

    def handle_server_failure(self, event: Event) -> AgentDecision:
        """Recover workloads from one failed server. Failures stay visible."""
        try:
            return self._recover(event)
        except Exception:
            if self._state is not AgentState.FAILED:
                self._set_state(AgentState.FAILED)
            raise

    def _recover(self, event: Event) -> AgentDecision:
        self._set_state(AgentState.OBSERVING)
        if event.event_type is not EventType.SERVER_FAILURE:
            self._fail(event.source_id, "event is not a server failure")
        failed_server_id = _failed_server_id(event)
        failed = self._find_server_by_id(failed_server_id)
        if failed is None:
            self._fail(failed_server_id, "server does not exist", failed_server_id=failed_server_id)
        affected = list(failed.workloads)
        self._reject_duplicate_assignments(affected)
        forecasts = self._forecasts(affected, _histories_from_payload(event))

        self._set_state(AgentState.PLANNING)
        proposal = self._plan_recovery(affected, forecasts, failed_server_id)

        self._set_state(AgentState.EVALUATING)
        self._evaluate_recovery(affected, forecasts, proposal, failed_server_id)

        self._set_state(AgentState.ACTING)
        journal = self._apply_recovery(failed, proposal.placements, forecasts)

        self._set_state(AgentState.VERIFYING)
        try:
            self._verify_recovery(failed, proposal.placements)
        except ValueError as error:
            self._restore_recovery(journal)
            self._fail(
                failed_server_id,
                str(error),
                failed_server_id=failed_server_id,
                affected=tuple(workload.workload_id for workload in affected),
                unrecovered=tuple(workload.workload_id for workload in affected),
                placements=tuple(proposal.placements.items()),
                planning_cpus=tuple((item.workload_id, item.planning_cpu) for item in forecasts),
            )

        recovered = tuple(proposal.placements)
        decision = AgentDecision(
            workload_id=failed_server_id,
            server_id=None,
            predicted_cpu=None,
            planning_cpu=0.0,
            state=AgentState.COMPLETED,
            reason="workloads recovered",
            failed_server_id=failed_server_id,
            affected_workloads=tuple(workload.workload_id for workload in affected),
            recovered_workloads=recovered,
            unrecovered_workloads=(),
            placements=tuple(proposal.placements.items()),
            planning_cpus=tuple((item.workload_id, item.planning_cpu) for item in forecasts),
            predicted_cpus=tuple((item.workload_id, item.predicted_cpu) for item in forecasts),
        )
        self._set_state(AgentState.COMPLETED)
        self._decision = decision
        return decision

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

    def _forecasts(self, workloads: list[Workload], histories: dict[str, list[float]]) -> list["_RecoveryForecast"]:
        forecasts: list[_RecoveryForecast] = []
        for workload in workloads:
            predicted = self._predict(histories.get(workload.workload_id, []))
            planning_cpu = workload.cpu_required if predicted is None else max(workload.cpu_required, predicted)
            forecasts.append(_RecoveryForecast(workload, predicted, planning_cpu))
        return forecasts

    def _plan_recovery(
        self,
        affected: list[Workload],
        forecasts: list["_RecoveryForecast"],
        failed_server_id: str,
    ):
        planning = build_planning_workloads(
            affected,
            [DemandPrediction(item.workload.workload_id, item.planning_cpu) for item in forecasts],
        )
        candidates = [
            server
            for server in self._copied_servers()
            if server.server_id != failed_server_id and server.status is ServerStatus.OPERATIONAL
        ]
        return self._optimizer.optimize(planning, candidates)

    def _evaluate_recovery(
        self,
        affected: list[Workload],
        forecasts: list["_RecoveryForecast"],
        proposal,
        failed_server_id: str,
    ) -> None:
        affected_ids = tuple(workload.workload_id for workload in affected)
        planning_cpus = tuple((item.workload_id, item.planning_cpu) for item in forecasts)
        if proposal.unplaced_workloads or set(proposal.placements) != set(affected_ids):
            self._fail(
                failed_server_id,
                "recovery plan is infeasible",
                failed_server_id=failed_server_id,
                affected=affected_ids,
                unrecovered=tuple(proposal.unplaced_workloads) or affected_ids,
                placements=tuple(proposal.placements.items()),
                planning_cpus=planning_cpus,
            )
        temporary, copies = self._temporary_datacenter()
        by_id = {item.workload.workload_id: item for item in forecasts}
        for workload_id, server_id in proposal.placements.items():
            if server_id == failed_server_id or server_id not in copies:
                self._fail(
                    failed_server_id,
                    "recovery selected an unavailable server",
                    failed_server_id=failed_server_id,
                    affected=affected_ids,
                    unrecovered=affected_ids,
                    planning_cpus=planning_cpus,
                )
            chosen = copies[server_id]
            forecast = by_id[workload_id]
            planning_workload = replace(forecast.workload, cpu_required=forecast.planning_cpu)
            if not chosen.can_host(planning_workload):
                self._fail(
                    failed_server_id,
                    "proposal violates capacity",
                    failed_server_id=failed_server_id,
                    affected=affected_ids,
                    unrecovered=affected_ids,
                    placements=tuple(proposal.placements.items()),
                    planning_cpus=planning_cpus,
                )
            chosen.allocate(planning_workload)
        failed_copy = copies[failed_server_id]
        failed_copy.status = ServerStatus.INACTIVE
        self._engine_factory(temporary).advance_tick()
        for server in copies.values():
            if server.cpu_utilization() > 1 or server.memory_utilization() > 1:
                self._fail(
                    failed_server_id,
                    "evaluation left a copy over capacity",
                    failed_server_id=failed_server_id,
                    affected=affected_ids,
                    unrecovered=affected_ids,
                    planning_cpus=planning_cpus,
                )

    def _apply_recovery(self, failed: Server, placements: dict[str, str], forecasts: list["_RecoveryForecast"]) -> "_RecoveryJournal":
        original_status = failed.status
        released: list[Workload] = []
        moved: list[tuple[Server, Workload]] = []
        try:
            for workload in list(failed.workloads):
                destination = self._operational_server(placements[workload.workload_id])
                failed.release(workload)
                released.append(workload)
                destination.allocate(workload)
                moved.append((destination, workload))
            failed.status = ServerStatus.INACTIVE
        except Exception:
            journal = _RecoveryJournal(failed, original_status, released, moved)
            self._restore_recovery(journal)
            self._fail(
                failed.server_id,
                "recovery allocation failed",
                failed_server_id=failed.server_id,
                affected=tuple(item.workload.workload_id for item in forecasts),
                unrecovered=tuple(item.workload.workload_id for item in forecasts),
                placements=tuple(placements.items()),
                planning_cpus=tuple((item.workload_id, item.planning_cpu) for item in forecasts),
            )
        return _RecoveryJournal(failed, original_status, released, moved)

    def _verify_recovery(self, failed: Server, placements: dict[str, str]) -> None:
        if failed.status is not ServerStatus.INACTIVE:
            raise ValueError("failed server is still operational")
        hosted = {workload.workload_id for workload in failed.workloads}
        if hosted.intersection(placements):
            raise ValueError("failed server still hosts a recovered workload")
        locations: dict[str, str] = {}
        for server in self._servers():
            if server.cpu_utilization() > 1 or server.memory_utilization() > 1:
                raise ValueError("server over capacity after recovery")
            for workload in server.workloads:
                if workload.workload_id in locations:
                    raise ValueError("workload is assigned to multiple servers")
                locations[workload.workload_id] = server.server_id
        for workload_id, server_id in placements.items():
            if locations.get(workload_id) != server_id:
                raise ValueError("recovered workload is not on the selected server")
            if server_id == failed.server_id:
                raise ValueError("recovered workload is still on the failed server")

    def _restore_recovery(self, journal: "_RecoveryJournal") -> None:
        for server, workload in reversed(journal.moved):
            if any(item.workload_id == workload.workload_id for item in server.workloads):
                server.release(workload)
        journal.failed.status = ServerStatus.OPERATIONAL
        for workload in journal.released:
            if not any(item.workload_id == workload.workload_id for item in journal.failed.workloads):
                journal.failed.allocate(workload)
        journal.failed.status = journal.original_status

    def _reject_duplicate_assignments(self, affected: list[Workload]) -> None:
        affected_ids = {workload.workload_id for workload in affected}
        for server in self._servers():
            for workload in server.workloads:
                if workload.workload_id in affected_ids and workload not in affected:
                    self._fail(workload.workload_id, "workload is already placed on another server")

    def _operational_server(self, server_id: str) -> Server:
        for server in self._servers():
            if server.server_id == server_id:
                if server.status is not ServerStatus.OPERATIONAL:
                    raise ValueError(f"server {server_id!r} is not operational")
                return server
        raise ValueError(f"server {server_id!r} is not in the data center")

    def _find_server_by_id(self, server_id: str) -> Server | None:
        for server in self._servers():
            if server.server_id == server_id:
                return server
        return None

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
        failed_server_id: str | None = None,
        affected: tuple[str, ...] = (),
        recovered: tuple[str, ...] = (),
        unrecovered: tuple[str, ...] = (),
        placements: tuple[tuple[str, str], ...] = (),
        planning_cpus: tuple[tuple[str, float], ...] = (),
        predicted_cpus: tuple[tuple[str, float | None], ...] = (),
    ) -> None:
        self._set_state(AgentState.FAILED)
        self._decision = AgentDecision(
            workload_id=workload_id,
            server_id=None,
            predicted_cpu=predicted,
            planning_cpu=planning_cpu,
            state=AgentState.FAILED,
            reason=reason,
            failed_server_id=failed_server_id,
            affected_workloads=affected,
            recovered_workloads=recovered,
            unrecovered_workloads=unrecovered,
            placements=placements,
            planning_cpus=planning_cpus,
            predicted_cpus=predicted_cpus,
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


@dataclass(frozen=True)
class _RecoveryForecast:
    workload: Workload
    predicted_cpu: float | None
    planning_cpu: float

    @property
    def workload_id(self) -> str:
        return self.workload.workload_id


@dataclass(frozen=True)
class _RecoveryJournal:
    failed: Server
    original_status: ServerStatus
    released: list[Workload]
    moved: list[tuple[Server, Workload]]


def _failed_server_id(event: Event) -> str:
    if "server_id" not in event.payload:
        return event.source_id
    server_id = event.payload["server_id"]
    if not isinstance(server_id, str) or not server_id.strip():
        raise ValueError("server_id must be a non-empty string")
    if server_id != event.source_id:
        raise ValueError("server_id does not match source_id")
    return server_id


def _histories_from_payload(event: Event) -> dict[str, list[float]]:
    raw = event.payload.get("histories")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("histories must be a dictionary")
    histories: dict[str, list[float]] = {}
    for workload_id, values in raw.items():
        if not isinstance(workload_id, str) or not workload_id.strip():
            raise ValueError("history keys must be workload ids")
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError("history must be a sequence of numbers")
        histories[workload_id] = [float(value) for value in values]
    return histories


def _history_from_payload(event: Event) -> list[float]:
    raw = event.payload.get("history")
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("history must be a sequence of numbers")
    return [float(value) for value in raw]
