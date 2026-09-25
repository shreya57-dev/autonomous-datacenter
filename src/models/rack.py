from dataclasses import dataclass, field

from .server import Server


@dataclass
class Rack:
    """A group of servers that share a cooling capacity."""

    rack_id: str
    cooling_capacity: float
    servers: list[Server] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not str(self.rack_id).strip():
            raise ValueError("rack_id must be non-empty")
        if isinstance(self.cooling_capacity, bool) or not isinstance(self.cooling_capacity, (int, float)):
            raise TypeError("cooling_capacity must be a number")
        if self.cooling_capacity < 0:
            raise ValueError("cooling_capacity cannot be negative")
