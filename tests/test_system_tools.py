import unittest

from models.datacenter import DataCenter
from models.rack import Rack
from models.server import Server, ServerStatus
from models.workload import Workload
from tools.system_tools import (
    get_resource_utilization,
    get_server_status,
    get_system_state,
    get_workload_placement,
    tool_registry,
)


def make_workload(
    workload_id: str,
    *,
    cpu_required: float = 2,
    memory_required: float = 4,
    priority: int = 1,
) -> Workload:
    return Workload(
        workload_id=workload_id,
        cpu_required=cpu_required,
        memory_required=memory_required,
        latency_requirement=50,
        priority=priority,
    )


def make_server(
    server_id: str,
    *,
    rack_id: str = "rack-1",
    cpu_capacity: float = 10,
    memory_capacity: float = 20,
    temperature: float = 21,
    power_consumption: float = 100,
    status: ServerStatus = ServerStatus.OPERATIONAL,
) -> Server:
    return Server(
        server_id=server_id,
        rack_id=rack_id,
        cpu_capacity=cpu_capacity,
        memory_capacity=memory_capacity,
        temperature=temperature,
        power_consumption=power_consumption,
        status=status,
    )


def make_datacenter() -> tuple[DataCenter, Server, Server, Workload, Workload]:
    placed = make_workload("job-1", cpu_required=4, memory_required=8)
    waiting = make_workload("job-2", cpu_required=1, memory_required=2, priority=2)
    active = make_server("server-1")
    inactive = make_server(
        "server-2",
        rack_id="rack-2",
        temperature=30,
        power_consumption=50,
        status=ServerStatus.INACTIVE,
    )
    active.allocate(placed)
    datacenter = DataCenter(
        racks=[
            Rack(rack_id="rack-1", cooling_capacity=5, servers=[active]),
            Rack(rack_id="rack-2", cooling_capacity=0, servers=[inactive]),
        ],
        workloads=[placed, waiting],
    )
    return datacenter, active, inactive, placed, waiting


class SystemToolTests(unittest.TestCase):
    def test_system_state_reads_racks_servers_and_unplaced_work(self) -> None:
        datacenter, active, inactive, placed, waiting = make_datacenter()

        state = get_system_state(datacenter)

        self.assertEqual([rack.rack_id for rack in state.racks], ["rack-1", "rack-2"])
        self.assertEqual(state.racks[0].cooling_capacity, 5)
        self.assertEqual(state.racks[0].server_ids, ("server-1",))
        self.assertEqual([server.server_id for server in state.servers], ["server-1", "server-2"])
        self.assertEqual(state.servers[0].status, "operational")
        self.assertEqual(state.servers[0].cpu_used, 4)
        self.assertEqual(state.servers[0].cpu_utilization, 0.4)
        self.assertEqual(state.servers[0].memory_used, 8)
        self.assertEqual(state.servers[0].memory_utilization, 0.4)
        self.assertEqual(state.servers[0].workloads[0].workload_id, placed.workload_id)
        self.assertEqual(state.servers[1].status, "inactive")
        self.assertEqual(state.servers[1].temperature, inactive.temperature)
        self.assertEqual(state.servers[1].power_consumption, 50)
        self.assertEqual([item.workload_id for item in state.unplaced_workloads], [waiting.workload_id])
        self.assertEqual(state.unplaced_workloads[0].priority, 2)
        self.assertIsNot(state.servers[0].workloads, active.workloads)

    def test_server_status_returns_the_requested_server(self) -> None:
        datacenter, _, _, placed, _ = make_datacenter()

        status = get_server_status(datacenter, "server-1")

        self.assertEqual(status.server_id, "server-1")
        self.assertEqual(status.rack_id, "rack-1")
        self.assertEqual(status.status, "operational")
        self.assertEqual(status.cpu_capacity, 10)
        self.assertEqual(status.memory_capacity, 20)
        self.assertEqual(status.workloads[0].cpu_required, placed.cpu_required)
        self.assertEqual(status.workloads[0].latency_requirement, 50)

    def test_workload_placement_reports_the_hosting_server(self) -> None:
        datacenter, _, _, _, _ = make_datacenter()

        placement = get_workload_placement(datacenter, "job-1")

        self.assertTrue(placement.placed)
        self.assertEqual(placement.server_id, "server-1")
        self.assertEqual(placement.rack_id, "rack-1")
        self.assertEqual(placement.workload.workload_id, "job-1")
        self.assertEqual(placement.workload.cpu_required, 4)
        self.assertEqual(placement.workload.memory_required, 8)

    def test_unplaced_workload_has_no_server(self) -> None:
        datacenter, _, _, _, waiting = make_datacenter()

        placement = get_workload_placement(datacenter, "job-2")

        self.assertFalse(placement.placed)
        self.assertIsNone(placement.server_id)
        self.assertIsNone(placement.rack_id)
        self.assertEqual(placement.workload.priority, waiting.priority)

    def test_resource_utilization_uses_operational_servers(self) -> None:
        datacenter, _, _, _, _ = make_datacenter()

        utilization = get_resource_utilization(datacenter)

        self.assertEqual(utilization.total_cpu_utilization, 0.4)
        self.assertEqual(utilization.total_memory_utilization, 0.4)
        self.assertEqual(utilization.total_power_consumption, 100)
        self.assertEqual(utilization.average_server_temperature, 21)
        self.assertEqual(utilization.workload_count, 1)
        self.assertEqual(utilization.active_server_count, 1)
        self.assertEqual(utilization.inactive_server_count, 1)
        self.assertEqual(utilization.operational_cpu_capacity, 10)
        self.assertEqual(utilization.operational_memory_capacity, 20)
        self.assertEqual(utilization.operational_cpu_used, 4)
        self.assertEqual(utilization.operational_memory_used, 8)

    def test_empty_datacenter_is_an_explicit_empty_snapshot(self) -> None:
        datacenter = DataCenter()

        state = get_system_state(datacenter)
        utilization = get_resource_utilization(datacenter)

        self.assertEqual(state.racks, ())
        self.assertEqual(state.servers, ())
        self.assertEqual(state.unplaced_workloads, ())
        self.assertEqual(utilization.active_server_count, 0)
        self.assertEqual(utilization.inactive_server_count, 0)
        self.assertEqual(utilization.total_cpu_utilization, 0.0)
        self.assertEqual(utilization.operational_cpu_capacity, 0.0)
        self.assertEqual(utilization.workload_count, 0)

    def test_missing_server_and_workload_are_rejected(self) -> None:
        datacenter, _, _, _, _ = make_datacenter()

        with self.assertRaises(ValueError) as missing_server:
            get_server_status(datacenter, "server-9")
        with self.assertRaises(ValueError) as missing_workload:
            get_workload_placement(datacenter, "job-9")

        self.assertIn("was not found", str(missing_server.exception))
        self.assertIn("was not found", str(missing_workload.exception))

    def test_blank_ids_are_rejected(self) -> None:
        datacenter, _, _, _, _ = make_datacenter()

        with self.assertRaises(ValueError):
            get_server_status(datacenter, "  ")
        with self.assertRaises(ValueError):
            get_workload_placement(datacenter, "")
        with self.assertRaises(TypeError):
            get_server_status(datacenter, 1)  # type: ignore[arg-type]

    def test_duplicate_server_id_is_rejected(self) -> None:
        first = make_server("server-1")
        second = make_server("server-1", rack_id="rack-2")
        datacenter = DataCenter(
            racks=[
                Rack(rack_id="rack-1", cooling_capacity=1, servers=[first]),
                Rack(rack_id="rack-2", cooling_capacity=1, servers=[second]),
            ]
        )

        with self.assertRaises(ValueError) as caught:
            get_server_status(datacenter, "server-1")
        self.assertIn("not unique", str(caught.exception))

    def test_conflicting_workload_records_are_rejected(self) -> None:
        placed = make_workload("job-1", cpu_required=4)
        other = make_workload("job-1", cpu_required=9)
        server = make_server("server-1")
        server.allocate(placed)
        datacenter = DataCenter(
            racks=[Rack(rack_id="rack-1", cooling_capacity=1, servers=[server])],
            workloads=[other],
        )

        with self.assertRaises(ValueError) as caught:
            get_workload_placement(datacenter, "job-1")
        self.assertIn("disagrees", str(caught.exception))

    def test_read_only_tools_do_not_change_the_datacenter(self) -> None:
        datacenter, active, inactive, placed, waiting = make_datacenter()
        workload_list = active.workloads
        before = (
            active.status,
            active.temperature,
            active.power_consumption,
            active.cpu_capacity,
            tuple(workload.workload_id for workload in active.workloads),
            tuple(workload.cpu_required for workload in active.workloads),
            inactive.status,
            tuple(workload.workload_id for workload in datacenter.workloads),
            tuple(rack.rack_id for rack in datacenter.racks),
            tuple(len(rack.servers) for rack in datacenter.racks),
        )

        get_system_state(datacenter)
        get_server_status(datacenter, "server-1")
        get_workload_placement(datacenter, "job-1")
        get_workload_placement(datacenter, "job-2")
        get_resource_utilization(datacenter)

        after = (
            active.status,
            active.temperature,
            active.power_consumption,
            active.cpu_capacity,
            tuple(workload.workload_id for workload in active.workloads),
            tuple(workload.cpu_required for workload in active.workloads),
            inactive.status,
            tuple(workload.workload_id for workload in datacenter.workloads),
            tuple(rack.rack_id for rack in datacenter.racks),
            tuple(len(rack.servers) for rack in datacenter.racks),
        )
        self.assertEqual(after, before)
        self.assertIs(active.workloads, workload_list)
        self.assertIs(active.workloads[0], placed)
        self.assertIs(datacenter.workloads[1], waiting)

    def test_registry_lists_read_only_tools_and_their_arguments(self) -> None:
        specs = {spec.name: spec for spec in tool_registry()}

        self.assertEqual(
            set(specs),
            {
                "get_system_state",
                "get_server_status",
                "get_workload_placement",
                "get_resource_utilization",
            },
        )
        self.assertEqual(specs["get_system_state"].parameters, ())
        self.assertEqual(specs["get_resource_utilization"].parameters, ())
        self.assertEqual(specs["get_server_status"].parameters[0].name, "server_id")
        self.assertEqual(specs["get_server_status"].parameters[0].type_name, "str")
        self.assertEqual(specs["get_workload_placement"].parameters[0].name, "workload_id")
        self.assertIs(specs["get_server_status"].function, get_server_status)
        self.assertTrue(specs["get_workload_placement"].description)
        datacenter, _, _, _, _ = make_datacenter()
        status = specs["get_server_status"].function(datacenter, "server-2")
        self.assertEqual(status.status, "inactive")
