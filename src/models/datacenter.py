from dataclasses import dataclass, field

from .rack import Rack
from .workload import Workload


@dataclass
class DataCenter:
    """The simulated facility: racks of servers and the workloads in the environment."""

    racks: list[Rack] = field(default_factory=list)
    workloads: list[Workload] = field(default_factory=list)
