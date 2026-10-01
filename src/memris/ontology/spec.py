"""The YAML as written: one pydantic model per section, unknown keys refused.

These models hold names exactly as the author wrote them (``Person``, ``fin:Card``).
Resolving them to qualified names and checking that they refer to something is the
compiler's job, so a typo in a *reference* is reported with its location instead of
failing validation here. A typo in a *key* (``subclas_of``) is refused here, because
silently ignoring it would change the meaning of the file.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Header(_Strict):
    id: str
    version: str
    default_prefix: str
    # The class of the reserved owner entity — the person the memory belongs to.
    owner_class: str | None = None


class ClassSpec(_Strict):
    label: str | None = None
    subclass_of: str | None = None
    abstract: bool = False
    system: bool = False
    maps_to: str | None = None
    deprecated: bool = False
    replaced_by: list[str] = Field(default_factory=list)

    @field_validator("replaced_by", mode="before")
    @classmethod
    def _one_or_many(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value


class RelationSpec(_Strict):
    label: str | None = None
    domain: str
    range: str
    inverse: str | None = None
    inverse_label: str | None = None
    symmetric: bool = False
    subproperty_of: str | None = None
    maps_to: str | None = None
    deprecated: bool = False
    replaced_by: list[str] = Field(default_factory=list)

    @field_validator("replaced_by", mode="before")
    @classmethod
    def _one_or_many(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value


class AttributeSpec(_Strict):
    label: str | None = None
    domain: str
    datatype: str = "string"
    subproperty_of: str | None = None
    maps_to: str | None = None
    deprecated: bool = False
    replaced_by: list[str] = Field(default_factory=list)

    @field_validator("replaced_by", mode="before")
    @classmethod
    def _one_or_many(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value


class OntologySpec(_Strict):
    ontology: Header
    prefixes: dict[str, str] = Field(default_factory=dict)
    classes: dict[str, ClassSpec] = Field(default_factory=dict)
    relations: dict[str, RelationSpec] = Field(default_factory=dict)
    attributes: dict[str, AttributeSpec] = Field(default_factory=dict)


class PropertyConstraint(_Strict):
    min_count: int | None = Field(default=None, ge=0)
    max_count: int | None = Field(default=None, ge=0)
    # For a relation: the class every object must belong to (sh:class).
    class_: str | None = Field(default=None, alias="class")


class ShapeSpec(_Strict):
    properties: dict[str, PropertyConstraint] = Field(default_factory=dict)


class ObjectSpec(_Strict):
    from_: str = Field(alias="from")
    class_: str = Field(alias="class")


class EmitSpec(_Strict):
    subject: str
    predicate: str
    object: ObjectSpec | None = None
    value: str | None = None


class WhenSpec(_Strict):
    source_type: str
    key: list[str] = Field(default_factory=list)

    @field_validator("key", mode="before")
    @classmethod
    def _one_or_many(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value


class MappingSpec(_Strict):
    id: str
    when: WhenSpec
    emit: EmitSpec


class MappingsSpec(_Strict):
    mappings: list[MappingSpec] = Field(default_factory=list)


__all__ = [
    "AttributeSpec",
    "ClassSpec",
    "EmitSpec",
    "Header",
    "MappingSpec",
    "MappingsSpec",
    "ObjectSpec",
    "OntologySpec",
    "PropertyConstraint",
    "RelationSpec",
    "ShapeSpec",
    "WhenSpec",
]
