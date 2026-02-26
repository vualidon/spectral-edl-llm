from __future__ import annotations

from typing import Literal


TemperatureMode = Literal["fixed", "mixed"]


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def build_temperature_schedule(
    *,
    num_cots: int = 6,
    mode: TemperatureMode = "fixed",
    temperature: float = 0.7,
    delta: float = 0.2,
    min_temperature: float = 0.0,
    max_temperature: float = 2.0,
) -> list[float]:
    if num_cots <= 0:
        raise ValueError("num_cots must be positive")
    if min_temperature > max_temperature:
        raise ValueError("min_temperature must be <= max_temperature")

    base = _clamp(float(temperature), min_temperature, max_temperature)
    if mode == "fixed":
        return [base for _ in range(num_cots)]

    if mode != "mixed":
        raise ValueError(f"unsupported temperature mode: {mode}")

    if num_cots != 6:
        raise ValueError("mixed mode in v1 currently requires num_cots=6")

    offsets = [0.0, 0.0, 0.0, -delta, +delta, +(2.0 * delta)]
    return [
        round(_clamp(base + offset, min_temperature, max_temperature), 10)
        for offset in offsets
    ]
