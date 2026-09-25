from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for the power and temperature updates.

    Defaults live here so the tick logic does not repeat magic numbers.
    """

    idle_power: float = 100.0
    max_power: float = 300.0
    temperature_increase_factor: float = 5.0
    cooling_factor: float = 1.0

    def __post_init__(self) -> None:
        _require_positive("idle_power", self.idle_power)
        _require_positive("max_power", self.max_power)
        if self.max_power < self.idle_power:
            raise ValueError("max_power must be greater than or equal to idle_power")
        _require_non_negative("temperature_increase_factor", self.temperature_increase_factor)
        _require_non_negative("cooling_factor", self.cooling_factor)


def _require_positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
