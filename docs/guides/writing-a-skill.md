# Writing a skill

A skill is a package of tools the `SkillRegistry` discovers under `config/skills/`: a
directory with a `manifest.yaml` that declares the skill and its tools, and a
`tools.py` that implements them. It is the lightest way to give the model a new tool;
a plugin (`examples/02-governed-tool`, `docs/architecture/plugin-contract.md`) is the
way to add anything more -- a deterministic handler, an agent, a job, a channel.

```
my-skill/
  manifest.yaml   # declares the skill, its tools, args, and prerequisites
  tools.py        # implements the tools
```

## The manifest

- **`governor_route`** classifies the capability for the governance kernel.
  `system/read` = pure/read-only compute; `system/net` = leaves the box (egress);
  pick the lightest route that is honest about what the tool does.
- **`args[].type`** is one of `string`, `int`, `number`, `enum`, `bool`. `enum` args
  take `options`. Optional args take `required: false` + `default`.
- **`requires`** is checked before loading: a missing package (`packages`), environment
  variable (`env_vars`), file (`config_files`), vault credential or Python version leaves
  the skill blocked rather than crashing the runtime. Each block is reported the same way:
  one INFO line, said once, naming what is missing (`missing env FOO_KEY`, never its
  value), and `blocked: env:FOO_KEY` (or `package:`, `config:`, `credential:`) in
  `iris skills list`. `env_vars` are read from the process environment; `extra` names the
  `iris-harness[...]` extra to install when a package is what is missing.

## The tools

`tools.py` exposes one or more LangChain `BaseTool` subclasses and a module-level
`SKILL_TOOLS` list. Each tool has a Pydantic `args_schema` (validated before `_run`)
and returns a JSON-serializable result. Never raise into the agent loop: return an
error value instead.

## A worked example: `unit-converter`

The simplest useful skill: pure computation, typed args, no network, no extra
packages. (`tests/unit/iris_harness/tools/test_skills/test_skill_guide.py` builds this
skill from the two blocks below and runs it, so they stay correct.)

```yaml
# manifest.yaml
name: unit-converter
version: 0.1.0
description: Convert a value between units of the same kind (length or temperature).
author: you
license: Apache-2.0
tools:
  - name: convert_units
    description: Convert a numeric value from one unit to another of the same kind.
      Length units (m, km, mi, ft) and temperature units (c, f, k) are supported;
      converting across kinds (e.g. m -> c) is rejected.
    # "system/read" is the lightest route: pure computation, no side effects, no egress.
    governor_route: system/read
    args:
      - name: value
        description: The numeric value to convert.
        type: number
      - name: from_unit
        description: The unit the value is currently in.
        type: enum
        options: ["m", "km", "mi", "ft", "c", "f", "k"]
      - name: to_unit
        description: The unit to convert to (must be the same kind as from_unit).
        type: enum
        options: ["m", "km", "mi", "ft", "c", "f", "k"]
requires:
  python: ">=3.12"
  packages:
    - pydantic>=2.0
  env_vars: []
  config_files: []
  agents: []
```

```python
# tools.py
from __future__ import annotations

from typing import Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

Unit = Literal["m", "km", "mi", "ft", "c", "f", "k"]

# Length units as a factor to metres; temperatures go through Celsius.
_LENGTH_TO_M: dict[str, float] = {"m": 1.0, "km": 1000.0, "mi": 1609.344, "ft": 0.3048}
_TEMP_UNITS = {"c", "f", "k"}


class ConvertUnitsInput(BaseModel):
    value: float = Field(description="The numeric value to convert.")
    from_unit: Unit = Field(description="The unit the value is currently in.")
    to_unit: Unit = Field(description="The unit to convert to (same kind).")


def _to_celsius(value: float, unit: str) -> float:
    if unit == "f":
        return (value - 32.0) * 5.0 / 9.0
    return value - 273.15 if unit == "k" else value


def _from_celsius(celsius: float, unit: str) -> float:
    if unit == "f":
        return celsius * 9.0 / 5.0 + 32.0
    return celsius + 273.15 if unit == "k" else celsius


def _convert(value: float, from_unit: str, to_unit: str) -> float:
    if from_unit in _LENGTH_TO_M and to_unit in _LENGTH_TO_M:
        return value * _LENGTH_TO_M[from_unit] / _LENGTH_TO_M[to_unit]
    if from_unit in _TEMP_UNITS and to_unit in _TEMP_UNITS:
        return _from_celsius(_to_celsius(value, from_unit), to_unit)
    raise ValueError(f"cannot convert across unit kinds: {from_unit} -> {to_unit}")


class ConvertUnitsTool(BaseTool):
    name: str = "convert_units"
    description: str = (
        "Convert a numeric value between units of the same kind. "
        "Length: m, km, mi, ft. Temperature: c, f, k."
    )
    args_schema: type[BaseModel] = ConvertUnitsInput

    def _run(self, value: float, from_unit: Unit, to_unit: Unit) -> dict[str, object]:
        try:
            result = _convert(value, from_unit, to_unit)
        except ValueError as exc:
            return {"error": str(exc)}
        return {"value": round(result, 6), "unit": to_unit, "from": {"value": value, "unit": from_unit}}

    async def _arun(self, value: float, from_unit: Unit, to_unit: Unit) -> dict[str, object]:
        return self._run(value=value, from_unit=from_unit, to_unit=to_unit)


# The registry imports this list to discover the package's tools.
SKILL_TOOLS = [ConvertUnitsTool]
```

## Install and test it

```bash
mkdir -p config/skills/builtin/unit-converter
# save the two files above into it, then restart the API
# (or rely on hot reload when IRIS_SKILL_HOTRELOAD is on)
```

The `SkillRegistry` discovers every `manifest.yaml` under `config/skills/` (except the
quarantined `auto/` tree) on `IrisRuntime.startup()`. To test a skill, load it the way
the registry does and call each tool's `_run`:

```python
from iris_harness.tools.skills.loader import load_skill_manifest, load_skill_tool_classes

manifest = load_skill_manifest(skill_dir)
tool = load_skill_tool_classes(skill_dir)[0]()
assert tool._run(value=100, from_unit="km", to_unit="mi")["value"] == 62.137119
```
