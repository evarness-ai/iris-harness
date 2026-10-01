"""The skill's work: plain functions, one per tool in ``manifest.yaml``.

Each takes the tool's arguments as keyword arguments (already checked against the
manifest's ``args``) and returns a string or something JSON can encode -- the observation
the model reads. Nothing here knows about IRIS, so it is tested like any Python.
"""

from __future__ import annotations

_METRES_PER_UNIT = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "km": 1000.0,
    "in": 0.0254,
    "ft": 0.3048,
    "mi": 1609.344,
}


def convert_length(
    value: float, from_unit: str, to_unit: str, digits: int = 3
) -> dict[str, object]:
    metres = value * _METRES_PER_UNIT[from_unit]
    result = round(metres / _METRES_PER_UNIT[to_unit], digits)
    return {
        "value": result,
        "unit": to_unit,
        "text": f"{value:g} {from_unit} = {result:g} {to_unit}",
    }
