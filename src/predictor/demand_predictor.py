"""Predict the next tick of CPU demand and build a separate planning input.

Each sample uses the previous `window` observations as features and the
following observation as the target. Samples stay in time order. Training
uses only the earlier part of that sequence.

Workload.cpu_required is the current demand. A prediction never replaces it.
Planning demand is max(current CPU, predicted CPU), stored on a new Workload.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass, replace

from sklearn.linear_model import LinearRegression

from models.server import Server, ServerStatus
from models.workload import Workload
from optimizer.placement_optimizer import OptimizationResult, PlacementOptimizer

PATTERNS = ("stable", "increasing", "decreasing", "varying")


@dataclass(frozen=True)
class DemandPrediction:
    """A next-tick CPU forecast. It is not a change to the workload."""

    workload_id: str
    predicted_cpu: float


@dataclass(frozen=True)
class PredictionScore:
    """Temporal hold-out score for one demand history."""

    training_samples: int
    test_samples: int
    mae: float
    actual: tuple[float, ...]
    predicted: tuple[float, ...]
    next_cpu: float


@dataclass(frozen=True)
class PlanningComparison:
    """Current-demand placement versus prediction-aware placement."""

    current: OptimizationResult
    prediction_aware: OptimizationResult
    current_peak_cpu: float
    planned_peak_cpu: float


class DemandPredictor:
    """Linear regression from recent CPU observations to the next tick."""

    def __init__(self, window: int = 3) -> None:
        if isinstance(window, bool) or not isinstance(window, int):
            raise ValueError("window must be an int")
        if window < 1:
            raise ValueError("window must be positive")
        self.window = window
        self._model = LinearRegression()
        self._trained = False

    def fit(self, history: Sequence[float]) -> None:
        """Train on every window in `history`. Does not modify `history`."""
        values = _validated_history(history)
        features, targets = _windows(values, self.window)
        if len(targets) < 2:
            raise ValueError(
                f"insufficient history: need at least {self.window + 2} observations to train"
            )
        self._model.fit(features, targets)
        self._trained = True

    def predict_next(self, history: Sequence[float]) -> float:
        """Predict the observation that would follow `history`."""
        if not self._trained:
            raise ValueError("predictor has not been trained")
        values = _validated_history(history)
        if len(values) < self.window:
            raise ValueError(
                f"insufficient history: need at least {self.window} observations to predict"
            )
        prediction = self._model.predict([values[-self.window :]])[0]
        return float(prediction)

    def evaluate(self, history: Sequence[float], *, test_fraction: float = 0.3) -> PredictionScore:
        """Train on earlier windows and score later windows. Also predict the next tick."""
        values = _validated_history(history)
        features, targets = _windows(values, self.window)
        train_count, test_count = _split_counts(len(targets), test_fraction)
        self._model.fit(features[:train_count], targets[:train_count])
        self._trained = True
        test_features = features[train_count:]
        test_targets = targets[train_count:]
        predicted = [float(value) for value in self._model.predict(test_features)]
        mae = sum(abs(actual - forecast) for actual, forecast in zip(test_targets, predicted)) / test_count
        return PredictionScore(
            training_samples=train_count,
            test_samples=test_count,
            mae=mae,
            actual=tuple(test_targets),
            predicted=tuple(predicted),
            next_cpu=self.predict_next(values),
        )


def generate_demand_history(
    pattern: str,
    length: int,
    *,
    start: float,
    step: float = 0.5,
    variation: float = 0.0,
    seed: int = 0,
) -> list[float]:
    """Build one CPU-demand sequence. Varying noise uses `seed` and no global random state."""
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {', '.join(PATTERNS)}")
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("length must be a positive int")
    _require_non_negative_number("start", start)
    if pattern == "stable":
        return [float(start)] * length
    if pattern == "increasing":
        _require_positive_number("step", step)
        return [float(start + step * index) for index in range(length)]
    if pattern == "decreasing":
        _require_positive_number("step", step)
        values = [float(start - step * index) for index in range(length)]
        if values[-1] < 0:
            raise ValueError("decreasing history would become negative")
        return values
    _require_non_negative_number("variation", variation)
    if start < variation:
        raise ValueError("variation cannot exceed start")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")
    generator = random.Random(seed)
    return [float(start + generator.uniform(-variation, variation)) for _ in range(length)]


def build_planning_workloads(
    workloads: Sequence[Workload],
    predictions: Sequence[DemandPrediction],
) -> list[Workload]:
    """Return new workloads whose CPU is max(current, predicted).

    The workloads passed in are not modified.
    """
    if isinstance(workloads, (str, bytes)) or not isinstance(workloads, Sequence):
        raise ValueError("workloads must be a sequence of Workload")
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise ValueError("predictions must be a sequence of DemandPrediction")
    predicted_by_id: dict[str, float] = {}
    for prediction in predictions:
        if not isinstance(prediction, DemandPrediction):
            raise ValueError("predictions must be a sequence of DemandPrediction")
        if prediction.workload_id in predicted_by_id:
            raise ValueError("prediction ids must be unique")
        _require_number("predicted_cpu", prediction.predicted_cpu)
        predicted_by_id[prediction.workload_id] = float(prediction.predicted_cpu)

    planned: list[Workload] = []
    seen: set[str] = set()
    for workload in workloads:
        if not isinstance(workload, Workload):
            raise ValueError("workloads must be a sequence of Workload")
        if workload.workload_id in seen:
            raise ValueError("workload ids must be unique")
        seen.add(workload.workload_id)
        if workload.workload_id not in predicted_by_id:
            raise ValueError(f"missing prediction for workload {workload.workload_id!r}")
        planning_cpu = max(workload.cpu_required, predicted_by_id[workload.workload_id])
        planned.append(replace(workload, cpu_required=planning_cpu))
    extra = set(predicted_by_id) - seen
    if extra:
        raise ValueError(f"unexpected prediction for workload {sorted(extra)[0]!r}")
    return planned


def compare_placement_plans(
    workloads: Sequence[Workload],
    current_servers: Sequence[Server],
    planning_servers: Sequence[Server],
    predictions: Sequence[DemandPrediction],
) -> PlanningComparison:
    """Place the same workloads with current CPU and with planning CPU.

    Planning peak uses max(current, predicted) demand. It is not current usage.
    Neither server list is modified, because the optimizer only proposes a placement.
    """
    current_result = PlacementOptimizer().optimize(workloads, current_servers)
    planned = build_planning_workloads(workloads, predictions)
    predicted_result = PlacementOptimizer().optimize(planned, planning_servers)
    current_cpu = {workload.workload_id: workload.cpu_required for workload in workloads}
    planning_cpu = {workload.workload_id: workload.cpu_required for workload in planned}
    return PlanningComparison(
        current=current_result,
        prediction_aware=predicted_result,
        current_peak_cpu=_peak_cpu(current_result.placements, current_cpu, current_servers),
        planned_peak_cpu=_peak_cpu(predicted_result.placements, planning_cpu, planning_servers),
    )


def _windows(history: list[float], window: int) -> tuple[list[list[float]], list[float]]:
    features: list[list[float]] = []
    targets: list[float] = []
    for index in range(window, len(history)):
        features.append(history[index - window : index])
        targets.append(history[index])
    return features, targets


def _split_counts(sample_count: int, test_fraction: float) -> tuple[int, int]:
    _require_number("test_fraction", test_fraction)
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if sample_count < 3:
        raise ValueError("insufficient history for a temporal train/test split")
    test_count = int(sample_count * test_fraction)
    if test_count < 1:
        test_count = 1
    train_count = sample_count - test_count
    if train_count < 2:
        raise ValueError("insufficient history for a temporal train/test split")
    return train_count, test_count


def _peak_cpu(
    placements: dict[str, str],
    cpu_by_workload: dict[str, float],
    servers: Sequence[Server],
) -> float:
    operational = [server for server in servers if server.status is ServerStatus.OPERATIONAL]
    if not operational:
        return 0.0
    used = {server.server_id: 0.0 for server in operational}
    for workload_id, server_id in placements.items():
        used[server_id] += cpu_by_workload[workload_id]
    return max(used[server.server_id] / server.cpu_capacity for server in operational)


def _validated_history(history: Sequence[float]) -> list[float]:
    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise ValueError("history must be a sequence of numbers")
    if len(history) == 0:
        raise ValueError("history must not be empty")
    values: list[float] = []
    for value in history:
        _require_non_negative_number("cpu demand", value)
        values.append(float(value))
    return values


def _require_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")


def _require_non_negative_number(name: str, value: float) -> None:
    _require_number(name, value)
    if value < 0:
        raise ValueError(f"{name} cannot be negative")


def _require_positive_number(name: str, value: float) -> None:
    _require_number(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
