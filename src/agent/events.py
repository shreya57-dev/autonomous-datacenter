"""Facts about the data center. An event does not say what to do next."""

from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    """What happened. The agent later decides how to respond."""

    WORKLOAD_ARRIVAL = "workload_arrival"
    DEMAND_CHANGE = "demand_change"
    SERVER_FAILURE = "server_failure"
    SERVER_RECOVERY = "server_recovery"
    RESOURCE_THRESHOLD_BREACH = "resource_threshold_breach"


@dataclass(frozen=True)
class Event:
    """One thing that happened at a simulation tick.

    payload is a copy of the dictionary passed in, so later changes to the
    caller's dictionary do not change this event.
    """

    event_type: EventType
    tick: int
    source_id: str
    payload: dict[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, EventType):
            raise ValueError("event_type must be an EventType")
        if isinstance(self.tick, bool) or not isinstance(self.tick, int):
            raise ValueError("tick must be an int")
        if self.tick < 0:
            raise ValueError("tick cannot be negative")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a dictionary")
        object.__setattr__(self, "payload", dict(self.payload))
