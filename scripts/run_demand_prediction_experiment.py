"""Compare two frozen placements on the actual next tick of CPU demand.

The histories are synthetic straight lines. A linear model can recover them,
which is not a claim about real demand.

Each plan is chosen once and is not revised after the next tick arrives.
Planned peak uses the demand the optimizer was given.
Future realized peak uses the actual next-tick demand on that same placement.
The comparison of interest is future realized peak.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.server import Server
from models.workload import Workload
from predictor.demand_predictor import (
    DemandPrediction,
    DemandPredictor,
    PredictionScore,
    build_planning_workloads,
    compare_placement_plans,
    generate_demand_history,
)

WINDOW = 3
TEST_FRACTION = 0.5
HISTORY_LENGTH = 10
CPU_CAPACITY = 10
SERIES = (
    ("workload-a", "stable", {"start": 5}),
    ("workload-b", "increasing", {"start": 0.25, "step": 0.5}),
    ("workload-c", "stable", {"start": 0.25}),
)


def scenario_series() -> tuple[dict[str, list[float]], dict[str, float]]:
    """Observed history, plus the next generated tick held back from planning."""
    histories: dict[str, list[float]] = {}
    future_cpu: dict[str, float] = {}
    for workload_id, pattern, options in SERIES:
        full = generate_demand_history(pattern, HISTORY_LENGTH + 1, **options)
        histories[workload_id] = full[:-1]
        future_cpu[workload_id] = full[-1]
    return histories, future_cpu


def scenario_histories() -> dict[str, list[float]]:
    histories, _future_cpu = scenario_series()
    return histories


def score_histories(histories: dict[str, list[float]]) -> dict[str, PredictionScore]:
    scores = {}
    for workload_id, history in histories.items():
        scores[workload_id] = DemandPredictor(window=WINDOW).evaluate(history, test_fraction=TEST_FRACTION)
    return scores


def make_servers() -> list[Server]:
    return [
        Server(
            server_id=server_id,
            rack_id="rack-1",
            cpu_capacity=CPU_CAPACITY,
            memory_capacity=20,
            temperature=20,
            power_consumption=0,
        )
        for server_id in ("server-1", "server-2")
    ]


def make_workloads(histories: dict[str, list[float]]) -> list[Workload]:
    return [
        Workload(
            workload_id=workload_id,
            cpu_required=history[-1],
            memory_required=1,
            latency_requirement=50,
            priority=1,
        )
        for workload_id, history in histories.items()
    ]


def print_scores(scores: dict[str, PredictionScore]) -> None:
    print("Temporal split on synthetic CPU histories. This is not a real-world accuracy claim.")
    print()
    for workload_id, score in scores.items():
        print(f"{workload_id}")
        print(f"  training samples: {score.training_samples}")
        print(f"  test samples:     {score.test_samples}")
        print(f"  MAE:              {score.mae:.6f}")
        print("  actual vs predicted:")
        for actual, predicted in zip(score.actual, score.predicted):
            print(f"    {actual:.4f}  {predicted:.4f}")
        print()


def peak_cpu(placements: dict[str, str], cpu_by_workload: dict[str, float], servers: list[Server]) -> float:
    """Highest server CPU utilization for one fixed placement and one demand map."""
    used = {server.server_id: 0.0 for server in servers}
    for workload_id, server_id in placements.items():
        used[server_id] += cpu_by_workload[workload_id]
    return max(used[server.server_id] / server.cpu_capacity for server in servers)


def print_decision(histories: dict[str, list[float]], future_cpu: dict[str, float], scores: dict[str, PredictionScore]) -> None:
    workloads = make_workloads(histories)
    predictions = [
        DemandPrediction(workload.workload_id, scores[workload.workload_id].next_cpu)
        for workload in workloads
    ]
    planned = build_planning_workloads(workloads, predictions)
    current_servers = make_servers()
    planning_servers = make_servers()
    comparison = compare_placement_plans(workloads, current_servers, planning_servers, predictions)
    planning_cpu = {workload.workload_id: workload.cpu_required for workload in planned}
    current_future_peak = peak_cpu(comparison.current.placements, future_cpu, current_servers)
    predicted_future_peak = peak_cpu(comparison.prediction_aware.placements, future_cpu, planning_servers)

    print(
        f"{'workload':<12} {'current':>8} {'predicted':>10} {'planning':>10} "
        f"{'future':>8} {'current plan':<12} {'predicted plan':<14}"
    )
    print("-" * 84)
    for workload in workloads:
        print(
            f"{workload.workload_id:<12} "
            f"{workload.cpu_required:8.4f} "
            f"{scores[workload.workload_id].next_cpu:10.4f} "
            f"{planning_cpu[workload.workload_id]:10.4f} "
            f"{future_cpu[workload.workload_id]:8.4f} "
            f"{comparison.current.placements.get(workload.workload_id, 'unplaced'):<12} "
            f"{comparison.prediction_aware.placements.get(workload.workload_id, 'unplaced'):<14}"
        )
    print()
    print(f"current unplaced:     {_ids(comparison.current.unplaced_workloads)}")
    print(f"prediction unplaced:  {_ids(comparison.prediction_aware.unplaced_workloads)}")
    print()
    print("Planned peak uses the demand the optimizer saw when it chose the placement.")
    print("Future realized peak keeps that placement and applies the actual next-tick demand.")
    print("The comparison of interest is future realized peak.")
    print()
    print(f"current peak cpu:                            {comparison.current_peak_cpu:.4f}")
    print(f"prediction-aware planned peak cpu:           {comparison.planned_peak_cpu:.4f}")
    print(f"current plan future realized peak:           {current_future_peak:.4f}")
    print(f"prediction-aware plan future realized peak:  {predicted_future_peak:.4f}")


def _ids(workload_ids: list[str]) -> str:
    if not workload_ids:
        return "none"
    return ", ".join(workload_ids)


def main() -> None:
    histories, future_cpu = scenario_series()
    scores = score_histories(histories)
    print_scores(scores)
    print_decision(histories, future_cpu, scores)


if __name__ == "__main__":
    main()
