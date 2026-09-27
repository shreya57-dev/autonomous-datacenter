"""In-process delivery of events. Handlers run in subscription order."""

from collections.abc import Callable

from .events import Event, EventType


class EventBus:
    """Deliver published events to subscribers. There is no queue or network."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Callable[[Event], None]]] = {
            event_type: [] for event_type in EventType
        }
        self._history: list[Event] = []

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Register a handler. Earlier subscriptions run first."""
        if not isinstance(event_type, EventType):
            raise ValueError("event_type must be an EventType")
        if not callable(handler):
            raise ValueError("handler must be callable")
        self._handlers[event_type].append(handler)

    def publish(self, event: Event) -> None:
        """Record the event, then call handlers for its type."""
        if not isinstance(event, Event):
            raise ValueError("event must be an Event")
        self._history.append(event)
        for handler in self._handlers[event.event_type]:
            handler(event)

    def published_events(self) -> tuple[Event, ...]:
        """Events in the order they were published."""
        return tuple(self._history)
