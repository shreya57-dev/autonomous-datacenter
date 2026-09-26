import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_demand_prediction_experiment as experiment

from models.server import Server
from models.workload import Workload
from predictor.demand_predictor import (
    DemandPrediction,
    DemandPredictor,
    build_planning_workloads,
    compare_placement_plans,
    generate_demand_history,
)


def make_server(server_id: str, *, cpu_capacity: float = 10) -> Server:
    return Server(
        server_id=server_id,
        rack_id="rack-1",
        cpu_capacity=cpu_capacity,
        memory_capacity=20,
        temperature=20,
        power_consumption=0,
    )


def make_workload(workload_id: str, cpu_required: float) -> Workload:
    return Workload(
        workload_id=workload_id,
        cpu_required=cpu_required,
        memory_required=1,
        latency_requirement=50,
        priority=1,
    )


def increasing_history() -> list[float]:
    return generate_demand_history("increasing", 10, start=0.25, step=0.5)


class DemandPredictorTests(unittest.TestCase):
    def test_history_generation_is_deterministic(self) -> None:
        first = generate_demand_history("varying", 8, start=4, variation=0.5, seed=7)
        second = generate_demand_history("varying", 8, start=4, variation=0.5, seed=7)
        other = generate_demand_history("varying", 8, start=4, variation=0.5, seed=8)

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(
            generate_demand_history("increasing", 4, start=1, step=0.5),
            [1.0, 1.5, 2.0, 2.5],
        )
        self.assertEqual(generate_demand_history("stable", 3, start=2), [2.0, 2.0, 2.0])
        self.assertEqual(
            generate_demand_history("decreasing", 3, start=2, step=0.5),
            [2.0, 1.5, 1.0],
        )

    def test_invalid_history_is_rejected(self) -> None:
        predictor = DemandPredictor()
        with self.assertRaises(ValueError):
            predictor.fit([])
        with self.assertRaises(ValueError):
            predictor.fit([1.0, -0.1, 2.0])
        with self.assertRaises(ValueError):
            generate_demand_history("seasonal", 4, start=1)
        with self.assertRaises(ValueError):
            generate_demand_history("decreasing", 6, start=1, step=0.5)

    def test_insufficient_history_is_rejected(self) -> None:
        predictor = DemandPredictor(window=3)
        with self.assertRaises(ValueError):
            predictor.fit([1.0, 1.0, 1.0, 1.0])
        predictor.fit([1.0, 1.0, 1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            predictor.predict_next([1.0, 1.0])

    def test_predictor_trains_and_returns_a_forecast(self) -> None:
        history = increasing_history()
        predictor = DemandPredictor(window=3)
        predictor.fit(history)

        prediction = predictor.predict_next(history)

        self.assertGreater(prediction, history[-1])

    def test_temporal_split_and_mae(self) -> None:
        history = increasing_history()
        score = DemandPredictor(window=3).evaluate(history, test_fraction=0.5)

        self.assertEqual(score.training_samples + score.test_samples, len(history) - 3)
        self.assertGreaterEqual(score.training_samples, 2)
        self.assertGreaterEqual(score.test_samples, 1)
        self.assertEqual(score.actual, tuple(history[-score.test_samples :]))
        self.assertAlmostEqual(score.mae, 0.0, places=6)
        self.assertEqual(len(score.predicted), score.test_samples)
        self.assertAlmostEqual(score.next_cpu, history[-1] + 0.5, places=6)

    def test_prediction_does_not_mutate_history(self) -> None:
        history = increasing_history()
        original = list(history)
        predictor = DemandPredictor(window=3)
        predictor.evaluate(history)
        predictor.predict_next(history)

        self.assertEqual(history, original)

    def test_prediction_aware_planning_preserves_current_feasibility(self) -> None:
        workloads = [
            make_workload("workload-a", 5.0),
            make_workload("workload-b", 4.75),
            make_workload("workload-c", 0.25),
        ]
        original_cpu = [workload.cpu_required for workload in workloads]
        predictions = [
            DemandPrediction("workload-a", 5.0),
            DemandPrediction("workload-b", 5.25),
            DemandPrediction("workload-c", 0.25),
        ]
        planned = build_planning_workloads(workloads, predictions)
        servers = [make_server("server-1"), make_server("server-2")]

        result = compare_placement_plans(workloads, servers, [make_server("server-1"), make_server("server-2")], predictions)

        self.assertEqual([workload.cpu_required for workload in workloads], original_cpu)
        self.assertEqual([workload.cpu_required for workload in planned], [5.0, 5.25, 0.25])
        current_cpu = {workload.workload_id: workload.cpu_required for workload in workloads}
        used = {server.server_id: 0.0 for server in servers}
        for workload_id, server_id in result.prediction_aware.placements.items():
            used[server_id] += current_cpu[workload_id]
        for server in servers:
            self.assertLessEqual(used[server.server_id], server.cpu_capacity)
        self.assertEqual(servers[0].workloads, [])

    def test_repeated_prediction_is_deterministic(self) -> None:
        history = increasing_history()
        first = DemandPredictor(window=3).evaluate(history, test_fraction=0.5)
        second = DemandPredictor(window=3).evaluate(history, test_fraction=0.5)

        self.assertEqual(first, second)

    def test_repeated_experiment_is_deterministic_and_changes_placement(self) -> None:
        workloads = [
            make_workload("workload-a", 5.0),
            make_workload("workload-b", 4.75),
            make_workload("workload-c", 0.25),
        ]
        predictions = [
            DemandPrediction("workload-a", 5.0),
            DemandPrediction("workload-b", 5.25),
            DemandPrediction("workload-c", 0.25),
        ]

        def run_once() -> object:
            return compare_placement_plans(
                workloads,
                [make_server("server-1"), make_server("server-2")],
                [make_server("server-1"), make_server("server-2")],
                predictions,
            )

        first = run_once()
        second = run_once()

        self.assertEqual(first, second)
        self.assertNotEqual(first.current.placements, first.prediction_aware.placements)
        self.assertLess(first.current_peak_cpu, first.planned_peak_cpu)

    def test_prediction_aware_plan_has_lower_future_realized_peak(self) -> None:
        histories, future_cpu = experiment.scenario_series()
        scores = experiment.score_histories(histories)
        workloads = experiment.make_workloads(histories)
        predictions = [
            DemandPrediction(workload.workload_id, scores[workload.workload_id].next_cpu)
            for workload in workloads
        ]
        current_servers = experiment.make_servers()
        planning_servers = experiment.make_servers()
        comparison = compare_placement_plans(workloads, current_servers, planning_servers, predictions)

        current_future_peak = experiment.peak_cpu(comparison.current.placements, future_cpu, current_servers)
        predicted_future_peak = experiment.peak_cpu(
            comparison.prediction_aware.placements,
            future_cpu,
            planning_servers,
        )

        self.assertLess(predicted_future_peak, current_future_peak)
        self.assertEqual(comparison.current.unplaced_workloads, [])
        self.assertEqual(comparison.prediction_aware.unplaced_workloads, [])


if __name__ == "__main__":
    unittest.main()
