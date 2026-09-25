import unittest

from models.datacenter import DataCenter
from models.rack import Rack
from models.server import Server, ServerStatus
from models.workload import Workload
from simulator.config import SimulationConfig
from simulator.engine import SimulationEngine


def make_server(
    server_id: str,
    *,
    cpu_capacity: float = 10,
    memory_capacity: float = 10,
    temperature: float = 20,
    status: ServerStatus = ServerStatus.OPERATIONAL,
) -> Server:
    return Server(
        server_id=server_id,
        rack_id="rack-1",
        cpu_capacity=cpu_capacity,
        memory_capacity=memory_capacity,
        temperature=temperature,
        power_consumption=0,
        status=status,
    )


def make_workload(
    workload_id: str,
    *,
    cpu_required: float,
    memory_required: float = 0,
) -> Workload:
    return Workload(
        workload_id=workload_id,
        cpu_required=cpu_required,
        memory_required=memory_required,
        latency_requirement=50,
        priority=1,
    )


def engine_with_server(
    server: Server,
    *,
    cooling_capacity: float = 0,
    config: SimulationConfig | None = None,
    tick: int = 0,
) -> SimulationEngine:
    datacenter = DataCenter(racks=[Rack(rack_id="rack-1", cooling_capacity=cooling_capacity, servers=[server])])
    return SimulationEngine(datacenter, config=config or SimulationConfig(), tick=tick)


class SimulationEngineTests(unittest.TestCase):
    def test_simulation_starts_at_tick_zero(self) -> None:
        engine = SimulationEngine(DataCenter())

        self.assertEqual(engine.tick, 0)
        self.assertEqual(engine.history, [])

    def test_advance_tick_increments_time_by_one(self) -> None:
        engine = SimulationEngine(DataCenter())

        engine.advance_tick()
        self.assertEqual(engine.tick, 1)

        engine.advance_tick()
        self.assertEqual(engine.tick, 2)

    def test_power_increases_when_utilization_increases(self) -> None:
        config = SimulationConfig(idle_power=100, max_power=300)
        light = engine_with_server(make_server("light"), config=config)
        busy = engine_with_server(make_server("busy"), config=config)
        light.datacenter.racks[0].servers[0].allocate(make_workload("light-job", cpu_required=1))
        busy.datacenter.racks[0].servers[0].allocate(make_workload("busy-job", cpu_required=8))

        light.advance_tick()
        busy.advance_tick()

        light_power = light.datacenter.racks[0].servers[0].power_consumption
        busy_power = busy.datacenter.racks[0].servers[0].power_consumption
        self.assertEqual(light_power, 120)
        self.assertEqual(busy_power, 260)
        self.assertGreater(busy_power, light_power)

    def test_temperature_increases_with_higher_utilization(self) -> None:
        config = SimulationConfig(temperature_increase_factor=10, cooling_factor=1)
        light = engine_with_server(make_server("light"), cooling_capacity=1, config=config)
        busy = engine_with_server(make_server("busy"), cooling_capacity=1, config=config)
        light.datacenter.racks[0].servers[0].allocate(make_workload("light-job", cpu_required=1))
        busy.datacenter.racks[0].servers[0].allocate(make_workload("busy-job", cpu_required=8))

        light.advance_tick()
        busy.advance_tick()

        light_temperature = light.datacenter.racks[0].servers[0].temperature
        busy_temperature = busy.datacenter.racks[0].servers[0].temperature
        self.assertEqual(light_temperature, 20)
        self.assertEqual(busy_temperature, 27)
        self.assertGreater(busy_temperature, light_temperature)

    def test_cooling_reduces_the_temperature_effect(self) -> None:
        config = SimulationConfig(temperature_increase_factor=10, cooling_factor=1)
        warm_rack = engine_with_server(make_server("warm"), cooling_capacity=0, config=config)
        cool_rack = engine_with_server(make_server("cool"), cooling_capacity=4, config=config)
        warm_rack.datacenter.racks[0].servers[0].allocate(make_workload("warm-job", cpu_required=5))
        cool_rack.datacenter.racks[0].servers[0].allocate(make_workload("cool-job", cpu_required=5))

        warm_rack.advance_tick()
        cool_rack.advance_tick()

        warm_temperature = warm_rack.datacenter.racks[0].servers[0].temperature
        cool_temperature = cool_rack.datacenter.racks[0].servers[0].temperature
        self.assertEqual(warm_temperature, 25)
        self.assertEqual(cool_temperature, 21)
        self.assertLess(cool_temperature, warm_temperature)

    def test_state_history_records_each_tick(self) -> None:
        server = make_server("server-1")
        server.allocate(make_workload("job-1", cpu_required=5, memory_required=2))
        config = SimulationConfig(idle_power=100, max_power=300, temperature_increase_factor=10, cooling_factor=1)
        engine = engine_with_server(server, cooling_capacity=2, config=config)

        record = engine.advance_tick()

        self.assertEqual(len(engine.history), 1)
        self.assertIs(engine.history[0], record)
        self.assertEqual(record.tick, 1)
        self.assertEqual(record.total_cpu_utilization, 0.5)
        self.assertEqual(record.total_memory_utilization, 0.2)
        self.assertEqual(record.total_power_consumption, 200)
        self.assertEqual(record.average_server_temperature, 23)

    def test_multiple_ticks_produce_multiple_history_entries(self) -> None:
        engine = engine_with_server(make_server("server-1"))

        engine.advance_tick()
        engine.advance_tick()
        engine.advance_tick()

        self.assertEqual([record.tick for record in engine.history], [1, 2, 3])

    def test_invalid_configuration_is_rejected(self) -> None:
        invalid_configs = [
            {"idle_power": 0},
            {"idle_power": -10},
            {"max_power": 0},
            {"max_power": -5},
            {"idle_power": 200, "max_power": 100},
            {"temperature_increase_factor": -1},
            {"cooling_factor": -0.5},
        ]
        for values in invalid_configs:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    SimulationConfig(**values)

        datacenter = DataCenter()
        with self.assertRaises(ValueError):
            SimulationEngine(datacenter, tick=-1)
        with self.assertRaises(TypeError):
            SimulationEngine(datacenter, tick=1.5)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
