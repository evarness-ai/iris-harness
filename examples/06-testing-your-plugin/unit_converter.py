"""The plugin under test: a ``convert_units`` tool (length and temperature).

Deliberately small -- the point of this example is its test file. The work is a pure
function (:func:`convert`), so most of it is tested without IRIS at all; ``setup``
only adapts it to a tool.
"""

from __future__ import annotations

from typing import Any

from iris_harness.sdk import PluginAPI

# Length units as a factor to metres; temperatures go through Celsius.
_TO_METRES = {"m": 1.0, "km": 1000.0, "mi": 1609.344, "ft": 0.3048}
_TEMPERATURES = {"c", "f", "k"}


def _to_celsius(value: float, unit: str) -> float:
    if unit == "f":
        return (value - 32.0) * 5.0 / 9.0
    return value - 273.15 if unit == "k" else value


def _from_celsius(value: float, unit: str) -> float:
    if unit == "f":
        return value * 9.0 / 5.0 + 32.0
    return value + 273.15 if unit == "k" else value


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """``value`` in ``from_unit``, expressed in ``to_unit`` (both of one kind)."""
    a, b = from_unit.lower(), to_unit.lower()
    if a in _TO_METRES and b in _TO_METRES:
        return value * _TO_METRES[a] / _TO_METRES[b]
    if a in _TEMPERATURES and b in _TEMPERATURES:
        return _from_celsius(_to_celsius(value, a), b)
    raise ValueError(f"cannot convert {from_unit} to {to_unit}")


def convert_units(args: dict[str, Any]) -> str:
    """The tool: never raises into the loop, returns an error line instead."""
    try:
        value = float(args["value"])
        result = convert(value, str(args["from_unit"]), str(args["to_unit"]))
    except (KeyError, TypeError, ValueError) as exc:
        return f"error: {exc}"
    return f"{value:g} {args['from_unit']} = {result:.2f} {args['to_unit']}"


def setup(api: PluginAPI) -> None:
    api.register_tool(
        "convert_units",
        "Convert a value between units of one kind: length (m, km, mi, ft) or "
        'temperature (c, f, k). Args: {"value": number, "from_unit": str, "to_unit": str}.',
        convert_units,
    )
