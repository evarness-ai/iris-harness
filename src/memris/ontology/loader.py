"""Read an ontology directory and compile it.

A directory holds ``ontology.yaml`` (required), and optionally ``shapes.yaml`` and
``mappings.yaml``. A file that is not valid YAML, or that uses a key the DSL does not
know, becomes an :class:`Issue` like any other problem, so one run reports everything.

**Fragments** (ADR-0115: ``core ∪ plugin vocabulary``) are further directories of the
same three files that add to the base: prefixes, classes, relations and attributes in
``ontology.yaml`` (no ``ontology:`` header — the base owns identity; an optional
``fragment: {id, version}`` names it), properties per class in ``shapes.yaml``, rules in
``mappings.yaml``. A fragment only adds: redefining a name the base or another fragment
defines is an error, reported against the fragment. The merged whole is then compiled
and checked as one ontology.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter, ValidationError

from memris.ontology.compiler import (
    CompileResult,
    Issue,
    MappingRule,
    Ontology,
    compile_mappings,
    compile_ontology,
)
from memris.ontology.spec import MappingsSpec, OntologySpec, ShapeSpec

ONTOLOGY_FILE = "ontology.yaml"
SHAPES_FILE = "shapes.yaml"
MAPPINGS_FILE = "mappings.yaml"

_SHAPES = TypeAdapter(dict[str, ShapeSpec])


class OntologyError(ValueError):
    """Raised by :func:`load_or_raise` when the ontology has errors."""

    def __init__(self, issues: list[Issue]) -> None:
        self.issues = issues
        super().__init__("\n".join(str(i) for i in issues))


def _read(path: Path, issues: list[Issue]) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        issues.append(Issue("unreadable", str(exc).splitlines()[0], path.name))
        return None


def _schema_issues(exc: ValidationError, file: str) -> list[Issue]:
    return [
        Issue("schema", err["msg"], f"{file}:{'.'.join(str(p) for p in err['loc'])}")
        for err in exc.errors()
    ]


def load_ontology(directory: str | Path) -> CompileResult | None:
    """Compile the ontology in ``directory``.

    Returns None only when ``ontology.yaml`` itself cannot be read or validated; the
    reason is in the issues of :func:`load_or_raise`. Use :func:`check_directory` for a
    result that always carries its issues.
    """
    return check_directory(directory)[0]


_FRAGMENT_SECTIONS = ("prefixes", "classes", "relations", "attributes")


def _where(root: Path, file: str) -> str:
    return f"{root.name}/{file}"


def _conflict(message: str, root: Path, file: str) -> Issue:
    return Issue("fragment-conflict", message, _where(root, file))


def _merge_ontology(base: dict[str, Any], fragment: Any, root: Path, issues: list[Issue]) -> None:
    """Add a fragment's sections to ``base`` in place; a clash or a header is an issue."""
    if fragment is None:
        return
    if not isinstance(fragment, dict):
        issues.append(Issue("schema", "a fragment is a mapping", _where(root, ONTOLOGY_FILE)))
        return
    for key in fragment:
        if key == "ontology":
            issues.append(
                Issue(
                    "fragment-header",
                    "a fragment does not redefine the ontology header",
                    _where(root, ONTOLOGY_FILE),
                )
            )
        elif key not in (*_FRAGMENT_SECTIONS, "fragment"):
            issues.append(
                Issue("schema", f"unknown fragment section '{key}'", _where(root, ONTOLOGY_FILE))
            )
    for section in _FRAGMENT_SECTIONS:
        entries = fragment.get(section) or {}
        if not isinstance(entries, dict):
            issues.append(Issue("schema", f"'{section}' is a mapping", _where(root, ONTOLOGY_FILE)))
            continue
        target = base.get(section) or {}
        base[section] = target
        for name, value in entries.items():
            if name in target:
                issues.append(
                    _conflict(f"'{name}' in {section} is already defined", root, ONTOLOGY_FILE)
                )
                continue
            target[name] = value


def _merge_shapes(base: dict[str, Any], fragment: Any, root: Path, issues: list[Issue]) -> None:
    if not isinstance(fragment, dict):
        return
    for cls, spec in fragment.items():
        props = (spec.get("properties") or {}) if isinstance(spec, dict) else {}
        existing = base.get(cls) or {}
        base[cls] = existing
        target = existing.get("properties") or {}
        existing["properties"] = target
        for prop, constraint in props.items():
            if prop in target:
                issues.append(
                    _conflict(f"shape for '{cls}.{prop}' is already defined", root, SHAPES_FILE)
                )
                continue
            target[prop] = constraint


def _merge_mappings(base: dict[str, Any], fragment: Any, root: Path, issues: list[Issue]) -> None:
    if not isinstance(fragment, dict):
        return
    rules = base.get("mappings") or []
    base["mappings"] = rules
    taken = {r.get("id") for r in rules if isinstance(r, dict)}
    for rule in fragment.get("mappings") or []:
        rule_id = rule.get("id") if isinstance(rule, dict) else None
        if rule_id in taken:
            issues.append(_conflict(f"mapping '{rule_id}' is already defined", root, MAPPINGS_FILE))
            continue
        taken.add(rule_id)
        rules.append(rule)


def check_directory(
    directory: str | Path, fragments: Iterable[str | Path] = ()
) -> tuple[CompileResult | None, list[Issue]]:
    """Compile ``directory`` plus any ``fragments``; return the result and every issue."""
    root = Path(directory)
    issues: list[Issue] = []

    raw = _read(root / ONTOLOGY_FILE, issues)
    raw_shapes: Any = _read(root / SHAPES_FILE, issues) if (root / SHAPES_FILE).exists() else None
    raw_mappings: Any = (
        _read(root / MAPPINGS_FILE, issues) if (root / MAPPINGS_FILE).exists() else None
    )
    for part in (Path(f) for f in fragments):
        if isinstance(raw, dict) and (part / ONTOLOGY_FILE).exists():
            _merge_ontology(raw, _read(part / ONTOLOGY_FILE, issues), part, issues)
        if (part / SHAPES_FILE).exists():
            raw_shapes = raw_shapes if isinstance(raw_shapes, dict) else {}
            _merge_shapes(raw_shapes, _read(part / SHAPES_FILE, issues), part, issues)
        if (part / MAPPINGS_FILE).exists():
            raw_mappings = raw_mappings if isinstance(raw_mappings, dict) else {}
            _merge_mappings(raw_mappings, _read(part / MAPPINGS_FILE, issues), part, issues)
    if isinstance(raw, dict):
        raw.pop("fragment", None)

    try:
        spec = OntologySpec.model_validate(raw or {})
    except ValidationError as exc:
        return None, issues + _schema_issues(exc, ONTOLOGY_FILE)

    shapes: dict[str, ShapeSpec] = {}
    if raw_shapes is not None:
        try:
            shapes = _SHAPES.validate_python(raw_shapes or {})
        except ValidationError as exc:
            issues += _schema_issues(exc, SHAPES_FILE)

    mappings = MappingsSpec()
    if raw_mappings is not None:
        try:
            mappings = MappingsSpec.model_validate(raw_mappings or {})
        except ValidationError as exc:
            issues += _schema_issues(exc, MAPPINGS_FILE)

    result = compile_ontology(spec, shapes, mappings)
    result.issues[:0] = issues
    return result, result.issues


def load_mappings(path: str | Path, ontology: Ontology) -> list[MappingRule]:
    """A connector's own mappings file, checked against ``ontology``; raises on errors."""
    issues: list[Issue] = []
    raw = _read(Path(path), issues)
    if issues:
        raise OntologyError(issues)
    try:
        spec = MappingsSpec.model_validate(raw or {})
    except ValidationError as exc:
        raise OntologyError(_schema_issues(exc, Path(path).name)) from exc
    rules, found = compile_mappings(ontology, spec)
    errors = [i for i in found if i.severity == "error"]
    if errors:
        raise OntologyError(errors)
    return rules


def load_or_raise(directory: str | Path, fragments: Iterable[str | Path] = ()) -> Ontology:
    """Compile ``directory`` plus ``fragments`` or raise :class:`OntologyError`."""
    result, issues = check_directory(directory, fragments)
    errors = [i for i in issues if i.severity == "error"]
    if result is None or errors:
        raise OntologyError(errors)
    return result.ontology


__all__ = [
    "MAPPINGS_FILE",
    "ONTOLOGY_FILE",
    "SHAPES_FILE",
    "OntologyError",
    "check_directory",
    "load_mappings",
    "load_ontology",
    "load_or_raise",
]
