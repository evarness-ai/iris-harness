"""Compile the YAML sections into one checked ontology.

The compiler resolves every name to a qualified name (``prefix:local``), synthesises
inverse relations that are named but not declared, and reports each problem it finds
as an :class:`Issue` with the place it came from. It never stops at the first problem:
a file with three mistakes reports three.

Nothing here knows a class, relation or attribute by name (ADR-0115, decision 2). The
only names the engine knows are the DSL's own keys and the XSD datatypes, which are a
standard rather than vocabulary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

from memris.ontology.spec import (
    AttributeSpec,
    ClassSpec,
    MappingsSpec,
    OntologySpec,
    RelationSpec,
    ShapeSpec,
)

Severity = Literal["error", "warning"]
TermKind = Literal["class", "relation", "attribute"]

XSD = "http://www.w3.org/2001/XMLSchema#"
DATATYPES = frozenset(
    {"string", "anyURI", "boolean", "integer", "decimal", "double", "date", "dateTime"}
)


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    where: str
    severity: Severity = "error"

    def __str__(self) -> str:
        return f"{self.severity}: {self.where}: {self.message} [{self.code}]"


@dataclass(frozen=True)
class ClassTerm:
    name: str
    label: str
    parent: str | None
    abstract: bool
    system: bool
    maps_to: str | None
    deprecated: bool
    replaced_by: tuple[str, ...]


@dataclass(frozen=True)
class RelationTerm:
    name: str
    label: str
    domain: str
    range: str
    inverse: str | None
    symmetric: bool
    parent: str | None
    maps_to: str | None
    deprecated: bool
    replaced_by: tuple[str, ...]
    synthesized: bool = False


@dataclass(frozen=True)
class AttributeTerm:
    name: str
    label: str
    domain: str
    datatype: str
    parent: str | None
    maps_to: str | None
    deprecated: bool
    replaced_by: tuple[str, ...]


@dataclass(frozen=True)
class Constraint:
    min_count: int | None
    max_count: int | None
    object_class: str | None


@dataclass(frozen=True)
class MappingRule:
    id: str
    source_type: str
    keys: tuple[str, ...]
    subject: str
    predicate: str
    object_from: str | None
    object_class: str | None
    value: str | None


@dataclass
class Ontology:
    id: str
    version: str
    default_prefix: str
    prefixes: dict[str, str]
    owner_class: str | None = None
    classes: dict[str, ClassTerm] = field(default_factory=dict)
    relations: dict[str, RelationTerm] = field(default_factory=dict)
    attributes: dict[str, AttributeTerm] = field(default_factory=dict)
    shapes: dict[str, dict[str, Constraint]] = field(default_factory=dict)
    mappings: list[MappingRule] = field(default_factory=list)

    def qualify(self, name: str) -> str:
        """``Person`` → ``mem:Person``; an already prefixed name is returned as is."""
        return name if ":" in name else f"{self.default_prefix}:{name}"

    def expand(self, name: str) -> str | None:
        """The full IRI for a qualified name, or None when its prefix is unknown."""
        prefix, _, local = self.qualify(name).partition(":")
        base = self.prefixes.get(prefix)
        return None if base is None else base + local

    def kind_of(self, name: str) -> TermKind | None:
        if name in self.classes:
            return "class"
        if name in self.relations:
            return "relation"
        if name in self.attributes:
            return "attribute"
        return None

    def ancestors(self, name: str) -> list[str]:
        """The class itself, then each superclass up to the root (cycle-safe)."""
        chain: list[str] = []
        current: str | None = name
        while current is not None and current in self.classes and current not in chain:
            chain.append(current)
            current = self.classes[current].parent
        return chain

    def is_subclass(self, name: str, of: str) -> bool:
        return of in self.ancestors(name)

    def property_domain(self, name: str) -> str | None:
        if name in self.relations:
            return self.relations[name].domain
        if name in self.attributes:
            return self.attributes[name].domain
        return None


@dataclass
class CompileResult:
    ontology: Ontology
    issues: list[Issue]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors


class _Compiler:
    def __init__(self, spec: OntologySpec) -> None:
        self.spec = spec
        self.issues: list[Issue] = []
        header = spec.ontology
        self.onto = Ontology(
            id=header.id,
            version=header.version,
            default_prefix=header.default_prefix,
            prefixes=dict(spec.prefixes),
        )

    # -- helpers ---------------------------------------------------------------

    def _issue(self, code: str, where: str, message: str, severity: Severity = "error") -> None:
        self.issues.append(Issue(code, message, where, severity))

    def _qualify(self, name: str, where: str) -> str:
        qualified = self.onto.qualify(name)
        prefix = qualified.partition(":")[0]
        if prefix not in self.onto.prefixes:
            self._issue("unknown-prefix", where, f"prefix '{prefix}' of '{name}' is not declared")
        return qualified

    def _ref(self, name: str, want: TermKind, where: str) -> str:
        """Qualify a reference and report it when it names nothing of the wanted kind."""
        qualified = self._qualify(name, where)
        kind = self.onto.kind_of(qualified)
        if kind is None:
            self._issue(f"unknown-{want}", where, f"'{name}' is not a declared {want}")
        elif kind != want:
            self._issue("wrong-kind", where, f"'{name}' is a {kind}, not a {want}")
        return qualified

    def _external(self, name: str | None, where: str) -> None:
        """``maps_to`` points outside the ontology, so it must carry a declared prefix."""
        if name is None:
            return
        prefix, sep, _ = name.partition(":")
        if not sep or prefix not in self.onto.prefixes:
            self._issue("unknown-prefix", where, f"maps_to '{name}' needs a declared prefix")

    # -- passes ----------------------------------------------------------------

    def compile(self) -> Ontology:
        if self.onto.default_prefix not in self.onto.prefixes:
            self._issue(
                "unknown-prefix",
                "ontology.default_prefix",
                f"default prefix '{self.onto.default_prefix}' is not declared",
            )
        self._declare()
        self._link_owner_class()
        self._link_classes()
        self._link_properties()
        self._synthesize_inverses()
        self._check_subproperties()
        self._check_replacements()
        return self.onto

    def _declare(self) -> None:
        seen: dict[str, str] = {}
        sections: list[tuple[str, Mapping[str, object]]] = [
            ("classes", self.spec.classes),
            ("relations", self.spec.relations),
            ("attributes", self.spec.attributes),
        ]
        for section, entries in sections:
            for raw in entries:
                where = f"{section}.{raw}"
                name = self._qualify(raw, where)
                if name in seen:
                    self._issue(
                        "duplicate-term", where, f"'{name}' is also declared in {seen[name]}"
                    )
                    continue
                seen[name] = section
                spec = entries[raw]
                if isinstance(spec, ClassSpec):
                    self.onto.classes[name] = ClassTerm(
                        name,
                        spec.label or raw.split(":")[-1],
                        None,
                        spec.abstract,
                        spec.system,
                        spec.maps_to,
                        spec.deprecated,
                        (),
                    )
                elif isinstance(spec, RelationSpec):
                    self.onto.relations[name] = RelationTerm(
                        name,
                        spec.label or raw.split(":")[-1],
                        "",
                        "",
                        None,
                        spec.symmetric,
                        None,
                        spec.maps_to,
                        spec.deprecated,
                        (),
                    )
                elif isinstance(spec, AttributeSpec):
                    self.onto.attributes[name] = AttributeTerm(
                        name,
                        spec.label or raw.split(":")[-1],
                        "",
                        "",
                        None,
                        spec.maps_to,
                        spec.deprecated,
                        (),
                    )

    def _link_owner_class(self) -> None:
        raw = self.spec.ontology.owner_class
        if raw is None:
            return
        where = "ontology.owner_class"
        name = self._ref(raw, "class", where)
        spec = self.spec.classes.get(raw) or self.spec.classes.get(name)
        if spec is not None and (spec.abstract or spec.deprecated):
            self._issue("owner-class", where, f"'{raw}' must be a concrete, live class")
        self.onto.owner_class = name

    def _link_classes(self) -> None:
        for raw, spec in self.spec.classes.items():
            where = f"classes.{raw}"
            name = self.onto.qualify(raw)
            if self.onto.classes.get(name) is None:
                continue
            parent = (
                self._ref(spec.subclass_of, "class", where + ".subclass_of")
                if spec.subclass_of
                else None
            )
            self._external(spec.maps_to, where + ".maps_to")
            replaced = tuple(self._qualify(r, where + ".replaced_by") for r in spec.replaced_by)
            term = self.onto.classes[name]
            self.onto.classes[name] = ClassTerm(
                name,
                term.label,
                parent,
                term.abstract,
                term.system,
                term.maps_to,
                term.deprecated,
                replaced,
            )
        for name in self.onto.classes:
            chain: list[str] = []
            current: str | None = name
            while current is not None and current in self.onto.classes:
                if current in chain:
                    self._issue(
                        "class-cycle",
                        f"classes.{name}",
                        "subclass_of forms a cycle: " + " → ".join([*chain, current]),
                    )
                    break
                chain.append(current)
                current = self.onto.classes[current].parent

    def _link_properties(self) -> None:
        for raw, rspec in self.spec.relations.items():
            where = f"relations.{raw}"
            name = self.onto.qualify(raw)
            if name not in self.onto.relations:
                continue
            domain = self._ref(rspec.domain, "class", where + ".domain")
            range_ = self._ref(rspec.range, "class", where + ".range")
            inverse = self._qualify(rspec.inverse, where + ".inverse") if rspec.inverse else None
            parent = (
                self._qualify(rspec.subproperty_of, where + ".subproperty_of")
                if rspec.subproperty_of
                else None
            )
            self._external(rspec.maps_to, where + ".maps_to")
            if rspec.symmetric and rspec.inverse:
                self._issue(
                    "symmetric-with-inverse",
                    where,
                    "a symmetric relation is its own inverse; drop 'inverse'",
                )
            if rspec.symmetric and domain != range_:
                self._issue(
                    "symmetric-domain-range",
                    where,
                    "a symmetric relation needs the same domain and range",
                )
            replaced = tuple(self._qualify(r, where + ".replaced_by") for r in rspec.replaced_by)
            term = self.onto.relations[name]
            self.onto.relations[name] = RelationTerm(
                name,
                term.label,
                domain,
                range_,
                inverse,
                term.symmetric,
                parent,
                term.maps_to,
                term.deprecated,
                replaced,
            )
        for raw, aspec in self.spec.attributes.items():
            where = f"attributes.{raw}"
            name = self.onto.qualify(raw)
            if name not in self.onto.attributes:
                continue
            domain = self._ref(aspec.domain, "class", where + ".domain")
            if aspec.datatype not in DATATYPES:
                self._issue(
                    "unknown-datatype",
                    where + ".datatype",
                    f"'{aspec.datatype}' is not one of {sorted(DATATYPES)}",
                )
            parent = (
                self._qualify(aspec.subproperty_of, where + ".subproperty_of")
                if aspec.subproperty_of
                else None
            )
            self._external(aspec.maps_to, where + ".maps_to")
            replaced = tuple(self._qualify(r, where + ".replaced_by") for r in aspec.replaced_by)
            attr = self.onto.attributes[name]
            self.onto.attributes[name] = AttributeTerm(
                name,
                attr.label,
                domain,
                XSD + aspec.datatype,
                parent,
                attr.maps_to,
                attr.deprecated,
                replaced,
            )

    def _synthesize_inverses(self) -> None:
        for raw, rspec in self.spec.relations.items():
            name = self.onto.qualify(raw)
            term = self.onto.relations.get(name)
            if term is None or term.inverse is None:
                continue
            where = f"relations.{raw}.inverse"
            other = self.onto.relations.get(term.inverse)
            if other is None:
                if self.onto.kind_of(term.inverse) is not None:
                    self._issue("wrong-kind", where, f"inverse '{term.inverse}' is not a relation")
                    continue
                self.onto.relations[term.inverse] = RelationTerm(
                    term.inverse,
                    rspec.inverse_label or term.inverse.split(":")[-1],
                    term.range,
                    term.domain,
                    name,
                    False,
                    None,
                    None,
                    term.deprecated,
                    (),
                    synthesized=True,
                )
            elif not other.synthesized and (
                other.inverse != name or other.domain != term.range or other.range != term.domain
            ):
                self._issue(
                    "inverse-mismatch",
                    where,
                    f"'{term.inverse}' must declare inverse '{name}' with domain and range swapped",
                )

    def _check_subproperties(self) -> None:
        props: dict[str, RelationTerm | AttributeTerm] = {
            **self.onto.relations,
            **self.onto.attributes,
        }
        for name, term in props.items():
            if term.parent is None:
                continue
            where = f"{'relations' if name in self.onto.relations else 'attributes'}.{name}.subproperty_of"
            parent = props.get(term.parent)
            if parent is None:
                self._issue(
                    "unknown-property", where, f"'{term.parent}' is not a declared property"
                )
                continue
            if type(parent) is not type(term):
                self._issue(
                    "wrong-kind",
                    where,
                    "a relation and an attribute cannot be sub/super properties",
                )
                continue
            if (
                term.domain
                and parent.domain
                and not self.onto.is_subclass(term.domain, parent.domain)
            ):
                self._issue(
                    "subproperty-domain",
                    where,
                    f"domain must be '{parent.domain}' or a subclass of it",
                )
            if (
                isinstance(term, RelationTerm)
                and isinstance(parent, RelationTerm)
                and term.range
                and parent.range
                and not self.onto.is_subclass(term.range, parent.range)
            ):
                self._issue(
                    "subproperty-range",
                    where,
                    f"range must be '{parent.range}' or a subclass of it",
                )
            seen = [name]
            current: str | None = term.parent
            while current is not None and current in props:
                if current in seen:
                    self._issue("subproperty-cycle", where, "subproperty_of forms a cycle")
                    break
                seen.append(current)
                current = props[current].parent

    def _check_replacements(self) -> None:
        terms: dict[str, ClassTerm | RelationTerm | AttributeTerm] = {
            **self.onto.classes,
            **self.onto.relations,
            **self.onto.attributes,
        }
        for name, term in terms.items():
            where = f"{self.onto.kind_of(name)}.{name}.replaced_by"
            if term.replaced_by and not term.deprecated:
                self._issue(
                    "replaced-not-deprecated",
                    where,
                    "replaced_by only makes sense on a deprecated term",
                )
            for target in term.replaced_by:
                if target not in terms:
                    self._issue("unknown-replacement", where, f"'{target}' is not declared")
                elif self.onto.kind_of(target) != self.onto.kind_of(name):
                    self._issue("wrong-kind", where, f"'{target}' is a {self.onto.kind_of(target)}")
            if term.deprecated and _replacement_cycle(name, terms):
                self._issue("replacement-cycle", where, "replaced_by leads back to this term")


def _replacement_cycle(
    start: str, terms: Mapping[str, ClassTerm | RelationTerm | AttributeTerm]
) -> bool:
    stack = list(terms[start].replaced_by)
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current == start:
            return True
        if current in seen or current not in terms:
            continue
        seen.add(current)
        stack.extend(terms[current].replaced_by)
    return False


def _compile_shapes(onto: Ontology, shapes: Mapping[str, ShapeSpec], issues: list[Issue]) -> None:
    for raw_target, shape in shapes.items():
        where = f"shapes.{raw_target}"
        target = onto.qualify(raw_target)
        if onto.kind_of(target) != "class":
            issues.append(Issue("unknown-class", f"'{raw_target}' is not a declared class", where))
            continue
        constraints: dict[str, Constraint] = {}
        for raw_prop, c in shape.properties.items():
            pwhere = f"{where}.{raw_prop}"
            prop = onto.qualify(raw_prop)
            domain = onto.property_domain(prop)
            if domain is None:
                issues.append(
                    Issue("unknown-property", f"'{raw_prop}' is not a declared property", pwhere)
                )
                continue
            if not onto.is_subclass(target, domain):
                issues.append(
                    Issue(
                        "shape-not-applicable",
                        f"'{raw_prop}' has domain '{domain}', which '{target}' is not",
                        pwhere,
                    )
                )
            if c.min_count is not None and c.max_count is not None and c.min_count > c.max_count:
                issues.append(Issue("shape-counts", "min_count is greater than max_count", pwhere))
            object_class = None
            if c.class_ is not None:
                object_class = onto.qualify(c.class_)
                if prop not in onto.relations:
                    issues.append(Issue("wrong-kind", "'class' only applies to a relation", pwhere))
                elif onto.kind_of(object_class) != "class":
                    issues.append(
                        Issue("unknown-class", f"'{c.class_}' is not a declared class", pwhere)
                    )
                elif not onto.is_subclass(object_class, onto.relations[prop].range):
                    issues.append(
                        Issue(
                            "shape-class",
                            f"'{c.class_}' is outside the range '{onto.relations[prop].range}'",
                            pwhere,
                        )
                    )
            if onto.kind_of(prop) and _deprecated(onto, prop):
                issues.append(
                    Issue("deprecated-in-use", f"'{prop}' is deprecated", pwhere, "warning")
                )
            constraints[prop] = Constraint(c.min_count, c.max_count, object_class)
        onto.shapes[target] = constraints


def _deprecated(onto: Ontology, name: str) -> bool:
    for table in (onto.classes, onto.relations, onto.attributes):
        term = table.get(name)
        if term is not None:
            return term.deprecated
    return False


def _compile_mappings(onto: Ontology, spec: MappingsSpec, issues: list[Issue]) -> None:
    ids: set[str] = set()
    claimed: dict[tuple[str, str | None], str] = {}
    for m in spec.mappings:
        where = f"mappings.{m.id}"
        if m.id in ids:
            issues.append(Issue("duplicate-mapping", f"id '{m.id}' is used twice", where))
        ids.add(m.id)
        keys: list[str | None] = [*m.when.key] or [None]  # no key: the whole source type
        for key in keys:
            slot = (m.when.source_type, key)
            if slot in claimed:
                shown = f"{slot[0]}:{key}" if key else slot[0]
                issues.append(
                    Issue(
                        "mapping-overlap",
                        f"'{shown}' is already mapped by '{claimed[slot]}'",
                        where,
                    )
                )
            else:
                claimed[slot] = m.id
        predicate = onto.qualify(m.emit.predicate)
        kind = onto.kind_of(predicate)
        object_class = onto.qualify(m.emit.object.class_) if m.emit.object else None
        if kind not in ("relation", "attribute"):
            issues.append(
                Issue(
                    "unknown-property",
                    f"predicate '{m.emit.predicate}' is not a declared property",
                    where,
                )
            )
        elif kind == "relation":
            if m.emit.object is None or m.emit.value is not None:
                issues.append(
                    Issue(
                        "mapping-kind",
                        f"'{m.emit.predicate}' is a relation: emit an 'object', not a 'value'",
                        where,
                    )
                )
            elif object_class is not None:
                if onto.kind_of(object_class) != "class":
                    issues.append(
                        Issue(
                            "unknown-class",
                            f"'{m.emit.object.class_}' is not a declared class",
                            where,
                        )
                    )
                else:
                    if onto.classes[object_class].abstract:
                        issues.append(
                            Issue(
                                "abstract-class",
                                f"'{object_class}' is abstract and cannot be instantiated",
                                where,
                            )
                        )
                    if not onto.is_subclass(object_class, onto.relations[predicate].range):
                        issues.append(
                            Issue(
                                "mapping-range",
                                f"'{object_class}' is outside the range '{onto.relations[predicate].range}'",
                                where,
                            )
                        )
        elif m.emit.value is None or m.emit.object is not None:
            issues.append(
                Issue(
                    "mapping-kind",
                    f"'{m.emit.predicate}' is an attribute: emit a 'value', not an 'object'",
                    where,
                )
            )
        for used in filter(None, (predicate if kind else None, object_class)):
            if onto.kind_of(used) and _deprecated(onto, used):
                issues.append(
                    Issue("deprecated-in-use", f"'{used}' is deprecated", where, "warning")
                )
        onto.mappings.append(
            MappingRule(
                m.id,
                m.when.source_type,
                tuple(m.when.key),
                m.emit.subject,
                predicate,
                m.emit.object.from_ if m.emit.object else None,
                object_class,
                m.emit.value,
            )
        )


def compile_mappings(onto: Ontology, spec: MappingsSpec) -> tuple[list[MappingRule], list[Issue]]:
    """Check a separate mappings file against an already compiled ontology.

    A connector ships its own mappings (how ITS records become statements) against the
    core ontology; this compiles them with the same rules as the ontology's own
    ``mappings.yaml`` without touching ``onto``.
    """
    scratch = replace(onto, mappings=[])
    issues: list[Issue] = []
    _compile_mappings(scratch, spec, issues)
    return scratch.mappings, issues


def compile_ontology(
    spec: OntologySpec,
    shapes: Mapping[str, ShapeSpec] | None = None,
    mappings: MappingsSpec | None = None,
) -> CompileResult:
    """Compile the three sections and return the ontology with every issue found."""
    compiler = _Compiler(spec)
    onto = compiler.compile()
    issues = compiler.issues
    _compile_shapes(onto, shapes or {}, issues)
    _compile_mappings(onto, mappings or MappingsSpec(), issues)
    return CompileResult(onto, issues)


def check_usage(onto: Ontology, used: Iterable[str]) -> list[Issue]:
    """Decision 8's data check: every term stored data uses must still resolve.

    A term may be deprecated as long as ``replaced_by`` says where it went; a term that
    has vanished, or was deprecated with nowhere to go, fails.
    """
    issues: list[Issue] = []
    for raw in sorted(set(used)):
        name = onto.qualify(raw)
        kind = onto.kind_of(name)
        if kind is None:
            issues.append(
                Issue(
                    "term-vanished",
                    f"'{name}' is used by stored data but no longer declared",
                    "data",
                )
            )
            continue
        table: Mapping[str, ClassTerm | RelationTerm | AttributeTerm] = (
            onto.classes
            if kind == "class"
            else onto.relations if kind == "relation" else onto.attributes
        )
        term = table[name]
        if term.deprecated and not term.replaced_by:
            issues.append(
                Issue("term-vanished", f"'{name}' is deprecated with no replaced_by", "data")
            )
    return issues


__all__ = [
    "compile_mappings",
    "DATATYPES",
    "XSD",
    "AttributeTerm",
    "ClassTerm",
    "CompileResult",
    "Constraint",
    "Issue",
    "MappingRule",
    "Ontology",
    "RelationTerm",
    "check_usage",
    "compile_ontology",
]
