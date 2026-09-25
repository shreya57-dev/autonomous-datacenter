import unittest

from simulator.workload_generator import WorkloadGenerator


def tight_generator(seed: int) -> WorkloadGenerator:
    return WorkloadGenerator(
        seed=seed,
        min_cpu=2,
        max_cpu=4,
        min_memory=8,
        max_memory=12,
        min_latency=20,
        max_latency=30,
        min_priority=1,
        max_priority=3,
    )


class WorkloadGeneratorTests(unittest.TestCase):
    def test_generates_the_requested_number_of_workloads(self) -> None:
        workloads = WorkloadGenerator(seed=1).generate(5)

        self.assertEqual(len(workloads), 5)

    def test_workload_ids_are_unique(self) -> None:
        workloads = WorkloadGenerator(seed=1).generate(4)

        self.assertEqual(
            [workload.workload_id for workload in workloads],
            ["workload-1", "workload-2", "workload-3", "workload-4"],
        )

    def test_generated_values_are_within_configured_ranges(self) -> None:
        workloads = tight_generator(seed=7).generate(20)

        for workload in workloads:
            self.assertGreaterEqual(workload.cpu_required, 2)
            self.assertLessEqual(workload.cpu_required, 4)
            self.assertGreaterEqual(workload.memory_required, 8)
            self.assertLessEqual(workload.memory_required, 12)
            self.assertGreaterEqual(workload.latency_requirement, 20)
            self.assertLessEqual(workload.latency_requirement, 30)
            self.assertGreaterEqual(workload.priority, 1)
            self.assertLessEqual(workload.priority, 3)
            self.assertIsInstance(workload.priority, int)

    def test_same_seed_produces_identical_workloads(self) -> None:
        first = WorkloadGenerator(seed=42).generate(10)
        second = WorkloadGenerator(seed=42).generate(10)

        self.assertEqual(first, second)

    def test_different_seeds_produce_different_scenarios(self) -> None:
        first = WorkloadGenerator(seed=42).generate(10)
        second = WorkloadGenerator(seed=99).generate(10)

        self.assertNotEqual(first, second)

    def test_zero_workloads_returns_an_empty_list(self) -> None:
        self.assertEqual(WorkloadGenerator(seed=1).generate(0), [])

    def test_invalid_configuration_raises_value_error(self) -> None:
        invalid_generators = [
            {"min_cpu": 5, "max_cpu": 1},
            {"min_memory": 10, "max_memory": 2},
            {"min_latency": 40, "max_latency": 10},
            {"min_priority": 5, "max_priority": 1},
            {"min_cpu": -1},
            {"min_memory": -0.5},
            {"min_latency": -10},
            {"min_priority": -1},
            {"max_priority": -1},
        ]
        for values in invalid_generators:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    WorkloadGenerator(seed=1, **values)

        generator = WorkloadGenerator(seed=1)
        with self.assertRaises(ValueError):
            generator.generate(-1)


if __name__ == "__main__":
    unittest.main()
