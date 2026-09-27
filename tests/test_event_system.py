import unittest

from agent.event_bus import EventBus
from agent.events import Event, EventType


def make_event(**overrides: object) -> Event:
    values: dict[str, object] = {
        "event_type": EventType.SERVER_FAILURE,
        "tick": 3,
        "source_id": "server-1",
        "payload": {"reason": "offline"},
    }
    values.update(overrides)
    return Event(**values)  # type: ignore[arg-type]


class EventTests(unittest.TestCase):
    def test_valid_event_creation(self) -> None:
        event = Event(
            event_type=EventType.WORKLOAD_ARRIVAL,
            tick=0,
            source_id="workload-1",
            payload={"cpu": 2},
        )

        self.assertEqual(event.event_type, EventType.WORKLOAD_ARRIVAL)
        self.assertEqual(event.tick, 0)
        self.assertEqual(event.source_id, "workload-1")
        self.assertEqual(event.payload, {"cpu": 2})

    def test_invalid_event_type(self) -> None:
        with self.assertRaises(ValueError):
            make_event(event_type="SERVER_FAILURE")

    def test_invalid_tick(self) -> None:
        with self.assertRaises(ValueError):
            make_event(tick=-1)
        with self.assertRaises(ValueError):
            make_event(tick=1.5)

    def test_invalid_source_id(self) -> None:
        with self.assertRaises(ValueError):
            make_event(source_id="")
        with self.assertRaises(ValueError):
            make_event(source_id="   ")

    def test_invalid_payload(self) -> None:
        with self.assertRaises(ValueError):
            make_event(payload=["cpu", 2])

    def test_caller_payload_mutation_does_not_change_the_event(self) -> None:
        payload = {"cpu": 2}
        event = make_event(payload=payload)

        payload["cpu"] = 9

        self.assertEqual(event.payload, {"cpu": 2})


class EventBusTests(unittest.TestCase):
    def test_handler_receives_published_event(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(EventType.SERVER_FAILURE, received.append)
        event = make_event()

        bus.publish(event)

        self.assertEqual(received, [event])

    def test_multiple_handlers_execute_in_subscription_order(self) -> None:
        bus = EventBus()
        order: list[str] = []
        bus.subscribe(EventType.DEMAND_CHANGE, lambda _event: order.append("first"))
        bus.subscribe(EventType.DEMAND_CHANGE, lambda _event: order.append("second"))

        bus.publish(make_event(event_type=EventType.DEMAND_CHANGE, source_id="workload-1"))

        self.assertEqual(order, ["first", "second"])

    def test_unrelated_event_type_is_not_delivered(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(EventType.SERVER_RECOVERY, received.append)

        bus.publish(make_event(event_type=EventType.SERVER_FAILURE))

        self.assertEqual(received, [])

    def test_event_with_no_subscribers_is_published(self) -> None:
        bus = EventBus()
        event = make_event(event_type=EventType.RESOURCE_THRESHOLD_BREACH, source_id="server-2")

        bus.publish(event)

        self.assertEqual(bus.published_events(), (event,))

    def test_invalid_subscription_is_rejected(self) -> None:
        bus = EventBus()

        with self.assertRaises(ValueError):
            bus.subscribe("SERVER_FAILURE", lambda _event: None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            bus.subscribe(EventType.SERVER_FAILURE, "not-a-handler")  # type: ignore[arg-type]

    def test_handler_exception_propagates(self) -> None:
        bus = EventBus()

        def fail(_event: Event) -> None:
            raise RuntimeError("handler failed")

        bus.subscribe(EventType.SERVER_FAILURE, fail)

        with self.assertRaises(RuntimeError):
            bus.publish(make_event())

    def test_published_events_keep_publication_order(self) -> None:
        bus = EventBus()
        first = make_event(event_type=EventType.WORKLOAD_ARRIVAL, tick=1, source_id="workload-1")
        second = make_event(event_type=EventType.SERVER_RECOVERY, tick=2, source_id="server-1")

        bus.publish(first)
        bus.publish(second)

        self.assertEqual(bus.published_events(), (first, second))


if __name__ == "__main__":
    unittest.main()
