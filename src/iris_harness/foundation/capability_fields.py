"""The text-bearing fields of a capability method's return type: declare, extract, rebuild.

A capability call is governed like a tool call (docs/architecture/plugin-capabilities.md §4):
``POST_TOOL_USE`` sees what the provider returned and may transform it -- today, masking the
owner's identity literals out of it -- before the consumer gets it. A provider returns typed
data, not text, so each method's spec names the **field paths** that carry text, and the
harness moves exactly those through the hooks and writes the result back into a copy.

**Paths.** Segments joined by ``.``: a field name, or ``[]`` for every element of a list,
tuple or sequence. ``""`` is the value itself (a method returning ``str``). ``[].subject``
is the ``subject`` of every item of a returned list; ``items.[].body`` the ``body`` of every
item in the ``items`` field. A declared path is a *pattern*; the concrete paths the hooks
see carry indices (``[0].subject``).

**What can be declared.** Plain data only: ``str`` (text); ``int`` / ``float`` / ``bool`` /
``None`` / ``Decimal`` / dates and times / ``UUID`` / ``Enum`` / ``Literal`` (not text);
``list[X]``, ``tuple[X, ...]``, ``Sequence[X]``; unions of these; dataclasses, pydantic
models and ``TypedDict``\\ s whose fields are these. Anything else -- ``bytes``, ``dict``,
``Any``, a handle, an arbitrary object -- is refused when the spec is defined, because a
field the harness cannot see into is a field it cannot redact. A spec must declare every
``str`` leaf its return type has, and nothing else: an undeclared text field would escape
the transform.

**The copy rule.** The consumer never holds the provider's object. The result is rebuilt,
container by container, from the declared fields alone, with no user code run in the consumer's path: a dataclass
as ``object.__new__(tp)`` with each declared field set by ``object.__setattr__`` (frozen and
slots ones included -- copied, never mutated; no ``__init__``, hand-written or generated,
runs; a field declared ``init=False`` is refused at spec time), a pydantic model with
``model_construct`` (no validators, no private attributes), a ``TypedDict`` / list / tuple as
a new one of its kind. So no private attribute, extra or other state of the provider's
instance comes along; a pydantic model declaring private attributes is refused at spec time.
Leaves are exact types and are rebuilt fresh by their own type.

**Plain data, no construction hooks, no re-validation.** The consumer's copy is built
*without* re-validation -- ``model_construct`` runs no validators, and masked text (the
identity mask in place of the original) may no longer satisfy one -- so a record type must
be plain data: a dataclass with a ``__post_init__`` or an ``InitVar`` (required or optional),
or a pydantic model with a ``model_post_init``, is refused at spec time. Those hooks would
run again on masked values, twice per call (the extracting walk and the rebuild), repeating
their side effects in the consumer's path. Any other failure building the copy is refused
too (``ResultMismatch``, which reaches the consumer as ``CapabilityDenied``).

**No code runs on access.** A record is refused at spec time when reading or writing its
fields would run code: a field name that is a data descriptor (other than a slot), an
overridden ``__getattribute__`` / ``__getattr__`` / ``__setattr__`` (a frozen dataclass's
own guard excepted), a pydantic computed field, or a property that may return text. Reading
the provider's fields is guarded too: a field that cannot be read (a slots instance built
without ``__init__``, a ``default_factory`` never run) is refused, not a raw error.

**Numbers are exact.** A field declared ``float`` takes a ``float``, not an ``int``; one
declared ``int`` takes an ``int``, not a ``bool``; ``datetime`` / ``Decimal`` / ``UUID`` and
the rest likewise, never a subclass. Declare ``int | float`` where either is meant.

**Strict at runtime.** Redaction is only as good as the walk, so a returned value is walked
along its *declared* type, never its own shape, and anything the declaration does not
describe is refused (:class:`ResultMismatch`; the runner turns it into ``CapabilityDenied``):
a subclass of a declared dataclass or model (exact classes only -- a subclass can carry
fields nobody declared), pydantic extras, ``TypedDict`` keys the type does not name,
instance attributes that are not dataclass fields, a ``str`` where the type says ``int``,
a list where it says tuple, and an object of any other kind.
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
import inspect
import json
import types
import typing
import uuid
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

ITEM = "[]"
_PLAIN_LEAVES: tuple[type[Any], ...] = (
    int,
    float,
    bool,
    type(None),
    Decimal,
    datetime.datetime,
    datetime.date,
    datetime.time,
    datetime.timedelta,
    uuid.UUID,
)
_SEQUENCES: tuple[Any, ...] = (list, tuple, Sequence, typing.Sequence)


class UndeclarableType(ValueError):
    """A return type the harness cannot see into, so cannot redact."""


def _join(prefix: str, segment: str) -> str:
    return f"{prefix}.{segment}" if prefix else segment


def _record_fields(tp: Any) -> dict[str, Any] | None:
    """``{field: annotation}`` for a dataclass, pydantic model or TypedDict; else None."""
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        hints = typing.get_type_hints(tp)
        fields = dataclasses.fields(tp)
        blocked = [f.name for f in fields if not f.init]
        if blocked:
            raise UndeclarableType(
                f"{tp.__name__}: field(s) {', '.join(blocked)} are init=False, so the "
                "result cannot be rebuilt from its fields"
            )
        init_vars = sorted(
            name
            for name, hint in hints.items()
            if isinstance(hint, dataclasses.InitVar) or hint is dataclasses.InitVar
        )
        if init_vars:
            raise UndeclarableType(
                f"{tp.__name__}: InitVar(s) {', '.join(init_vars)} are construction input no "
                "field holds, so the result cannot be rebuilt from its fields"
            )
        if hasattr(tp, "__post_init__"):
            raise UndeclarableType(
                f"{tp.__name__}: __post_init__ would run again on the masked copy; a "
                "capability result must be plain data with no construction hooks"
            )
        _refuse_hooks(tp, [f.name for f in fields], base=object)
        return {f.name: hints[f.name] for f in fields}
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        if tp.__private_attributes__:
            raise UndeclarableType(
                f"{tp.__name__}: private attributes ({', '.join(sorted(tp.__private_attributes__))})"
                " carry state no field declares, so they could not be redacted"
            )
        # (checked after private attributes: pydantic installs its own model_post_init to
        # initialise them, which is a construction hook too)
        if tp.model_post_init is not BaseModel.model_post_init:
            raise UndeclarableType(
                f"{tp.__name__}: model_post_init would run again on the masked copy; a "
                "capability result must be plain data with no construction hooks"
            )
        if tp.model_computed_fields:
            raise UndeclarableType(
                f"{tp.__name__}: computed field(s) {', '.join(sorted(tp.model_computed_fields))} "
                "are text no declared field holds, so they could not be redacted"
            )
        _refuse_hooks(tp, list(tp.model_fields), base=BaseModel)
        return {name: field.annotation for name, field in tp.model_fields.items()}
    if typing.is_typeddict(tp):
        return dict(typing.get_type_hints(tp))
    return None


_ACCESS_HOOKS = ("__getattribute__", "__getattr__", "__setattr__")


def _mentions_str(hint: Any) -> bool:
    return hint is str or any(_mentions_str(arg) for arg in typing.get_args(hint))


def _refuse_hooks(tp: type[Any], field_names: list[str], *, base: type[Any]) -> None:
    """Refuse a record whose attribute access runs code: it must be plain data by construction.

    - a declared field name that resolves (on the MRO) to a data descriptor -- one with
      ``__set__`` or ``__delete__`` -- other than a slot's member descriptor: reading or
      writing the field would run code;
    - ``__getattribute__`` / ``__getattr__`` / ``__setattr__`` overridden anywhere below
      ``base`` (``object`` for a dataclass, ``BaseModel`` for a model); a frozen
      dataclass's generated ``__setattr__`` is the one exception, being its guard;
    - a property whose value may be text (annotated to return ``str``, or not annotated):
      text computed at read time is text no declared field holds.
    """
    name_of = tp.__name__
    for field_name in field_names:
        attr = inspect.getattr_static(tp, field_name, None)
        is_data = hasattr(type(attr), "__set__") or hasattr(type(attr), "__delete__")
        if attr is not None and is_data and not isinstance(attr, types.MemberDescriptorType):
            raise UndeclarableType(
                f"{name_of}: field {field_name!r} is a data descriptor "
                f"({type(attr).__name__}); access to it runs code"
            )
    for klass in tp.__mro__:
        if klass is base or klass is object or issubclass(base, klass):
            continue
        for hook in _ACCESS_HOOKS:
            if hook not in vars(klass):
                continue
            params = getattr(klass, "__dataclass_params__", None)
            if hook == "__setattr__" and params is not None and params.frozen:
                continue  # the frozen dataclass's own guard
            raise UndeclarableType(
                f"{name_of}: {klass.__name__} overrides {hook}; attribute access must not run code"
            )
        for attr_name, attr in vars(klass).items():
            if not isinstance(attr, property) or attr.fget is None:
                continue
            returns = typing.get_type_hints(attr.fget).get("return")
            if returns is None or _mentions_str(returns):
                raise UndeclarableType(
                    f"{name_of}: property {attr_name!r} may return text no declared field holds"
                )


def text_paths(tp: Any, prefix: str = "", _seen: tuple[Any, ...] = ()) -> set[str]:
    """Every text-bearing path of ``tp``; raises :class:`UndeclarableType` when opaque."""
    if tp is str:
        return {prefix}
    if tp in _PLAIN_LEAVES or (isinstance(tp, type) and issubclass(tp, enum.Enum)):
        return set()
    origin = typing.get_origin(tp)
    if origin is typing.Literal:
        return set()
    if origin in (typing.Union, types.UnionType):
        return set().union(*(text_paths(arm, prefix, _seen) for arm in typing.get_args(tp)))
    if origin in _SEQUENCES:
        args = typing.get_args(tp)
        if origin is tuple and not (len(args) == 2 and args[1] is Ellipsis):
            raise UndeclarableType(f"{tp!r}: only tuple[X, ...] has declarable elements")
        if not args:
            raise UndeclarableType(f"{tp!r}: a sequence needs its element type")
        return text_paths(args[0], _join(prefix, ITEM), _seen)
    if tp in _seen:
        raise UndeclarableType(f"{tp!r}: a recursive type has no finite set of paths")
    fields = _record_fields(tp)
    if fields is None:
        raise UndeclarableType(
            f"{tp!r} is not plain data (str, numbers, dates, enums, sequences, dataclasses, "
            "pydantic models, TypedDicts): its text cannot be declared, so it cannot be redacted"
        )
    return set().union(
        *(text_paths(ann, _join(prefix, name), (*_seen, tp)) for name, ann in fields.items())
    )


# ------------------------------------------------------------------ values at runtime
class ResultMismatch(ValueError):
    """A runtime value that is not exactly its declared type, so it cannot be redacted."""


def _mismatch(path: str, expected: str, value: Any) -> ResultMismatch:
    where = path or "the result"
    return ResultMismatch(f"{where}: declared {expected}, got {type(value).__name__}")


def _walk(value: Any, tp: Any, path: str, updates: Mapping[str, str]) -> tuple[Any, dict[str, str]]:
    """Walk ``value`` along its DECLARED type ``tp``: its copy, and its text by concrete path.

    Strict, because redaction is only as good as the walk: the declared type's fields are
    walked, never the runtime object's, and anything the declaration does not describe is
    refused -- a subclass (with its extra fields), pydantic extras, ``TypedDict`` keys the
    type does not name, a ``str`` where the type says ``int``, an object of any other kind.
    """
    if tp is str:
        if type(value) is not str:
            raise _mismatch(path, "str", value)
        return updates.get(path, value), {path: value}
    if tp is type(None):
        if value is not None:
            raise _mismatch(path, "None", value)
        return None, {}
    origin = typing.get_origin(tp)
    if origin is typing.Literal:
        args = typing.get_args(tp)
        if type(value) not in {type(arg) for arg in args} or value not in args:
            raise _mismatch(path, repr(tp), value)
        return value, {}
    if origin in (typing.Union, types.UnionType):
        for arm in typing.get_args(tp):
            try:
                return _walk(value, arm, path, updates)
            except ResultMismatch:
                continue
        raise _mismatch(path, repr(tp), value)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        if type(value) is not tp:
            raise _mismatch(path, tp.__name__, value)
        return value, {}
    if tp in _PLAIN_LEAVES:
        # Exact, like every other node: a subclass of int or datetime can carry attributes
        # nobody declared (``Evil(int)`` with ``.secret``), and ``bool`` is not an ``int``
        # here, nor an ``int`` a ``float``. The consumer gets a fresh leaf, not the provider's.
        if type(value) is not tp:
            raise _mismatch(path, tp.__name__, value)
        return _construct(tp, path, lambda: _fresh(value)), {}
    if origin in _SEQUENCES:
        kinds: tuple[type[Any], ...] = (
            (list,) if origin is list else (tuple,) if origin is tuple else (list, tuple)
        )
        if type(value) not in kinds:
            raise _mismatch(path, repr(tp), value)
        item_type = typing.get_args(tp)[0]
        texts: dict[str, str] = {}
        items = []
        for i, item in enumerate(value):
            copy, found = _walk(item, item_type, _join(path, f"[{i}]"), updates)
            items.append(copy)
            texts.update(found)
        return (tuple(items) if type(value) is tuple else items), texts
    fields = _record_fields(tp)
    if fields is None:
        raise _mismatch(path, repr(tp), value)
    return _walk_record(value, tp, fields, path, updates)


def _construct(tp: Any, path: str, build: Callable[[], Any]) -> Any:
    """Build a copy; any failure is a :class:`ResultMismatch`, never a raw exception."""
    try:
        return build()
    except Exception as exc:  # the copy is the consumer's; its failure is a refusal
        name = getattr(tp, "__name__", repr(tp))
        raise ResultMismatch(
            f"{path or 'the result'}: {name} could not be rebuilt ({type(exc).__name__}: {exc})"
        ) from exc


def _fresh(value: Any) -> Any:
    """A new leaf equal to ``value`` (an exact plain leaf), built by its own type."""
    if isinstance(value, datetime.datetime | datetime.date | datetime.time):
        return value.replace()
    if isinstance(value, datetime.timedelta):
        return datetime.timedelta(
            days=value.days, seconds=value.seconds, microseconds=value.microseconds
        )
    if isinstance(value, uuid.UUID):
        return uuid.UUID(int=value.int)
    if isinstance(value, Decimal):
        return Decimal(str(value))  # ``Decimal(d)`` hands back ``d`` itself
    # int, float, bool: immutable exact values (small ints and bools are shared singletons)
    return type(value)(value)


def _walk_record(
    value: Any, tp: Any, fields: dict[str, Any], path: str, updates: Mapping[str, str]
) -> tuple[Any, dict[str, str]]:
    texts: dict[str, str] = {}
    copies: dict[str, Any] = {}

    def walk_field(name: str, field_value: Any) -> None:
        copy, found = _walk(field_value, fields[name], _join(path, name), updates)
        copies[name] = copy
        texts.update(found)

    if typing.is_typeddict(tp):
        if type(value) is not dict:
            raise _mismatch(path, tp.__name__, value)
        extra = sorted(set(value) - set(fields))
        if extra:
            raise ResultMismatch(f"{path or 'the result'}: keys {extra} are not in {tp.__name__}")
        for name, field_value in value.items():
            walk_field(name, field_value)
        return copies, texts
    if type(value) is not tp:  # a subclass may carry fields the declaration does not name
        raise _mismatch(path, tp.__name__, value)
    if isinstance(value, BaseModel):
        if value.model_extra:
            raise ResultMismatch(
                f"{path or 'the result'}: extra fields {sorted(value.model_extra)} are not in "
                f"{tp.__name__}"
            )
        for name in fields:
            walk_field(name, _read(value, name, path))
        # A fresh instance from the declared fields alone: no private attributes, no extras,
        # nothing of the provider's instance but what the declaration names.
        return _construct(tp, path, lambda: tp.model_construct(**copies)), texts
    extra = sorted(set(getattr(value, "__dict__", {})) - set(fields))
    if extra:
        raise ResultMismatch(
            f"{path or 'the result'}: attributes {extra} are not fields of {tp.__name__}"
        )
    for name in fields:
        walk_field(name, _read(value, name, path))
    return _construct(tp, path, lambda: _bare_dataclass(tp, copies)), texts


def _read(value: Any, name: str, path: str) -> Any:
    """A declared field of the provider's value; any failure reading it is a mismatch.

    A slots dataclass built without ``__init__``, or a ``default_factory`` field never set,
    has no value to read: that is the provider's broken value, refused, not a raw error.
    """
    try:
        return getattr(value, name)
    except Exception as exc:  # reading the provider's object is the provider's failure
        raise ResultMismatch(
            f"{_join(path, name)}: could not be read ({type(exc).__name__}: {exc})"
        ) from exc


def _bare_dataclass(tp: Any, values: dict[str, Any]) -> Any:
    """A ``tp`` instance holding ``values`` with no user code run in the consumer's path.

    ``object.__new__`` plus ``object.__setattr__`` per declared field: no ``__init__`` (a
    hand-written one included), no ``__post_init__``, no custom ``__setattr__``. It works
    for frozen dataclasses (their guard is a ``__setattr__`` this bypasses) and for slots
    ones (the slot descriptors are what ``object.__setattr__`` writes through).
    """
    copy = object.__new__(tp)
    for name, value in values.items():
        object.__setattr__(copy, name, value)
    return copy


def _matches(pattern: str, concrete: str) -> bool:
    p, c = pattern.split(".") if pattern else [], concrete.split(".") if concrete else []
    return len(p) == len(c) and all(
        a == b or (a == ITEM and b.startswith("[") and b.endswith("]"))
        for a, b in zip(p, c, strict=True)
    )


def extract_fields(value: Any, tp: Any, patterns: Sequence[str]) -> dict[str, str]:
    """``{concrete path: text}`` of ``value`` walked as its declared type ``tp``.

    Raises :class:`ResultMismatch` when ``value`` is not exactly ``tp``, or carries text at a
    path no declared pattern names.
    """
    _copy, texts = _walk(value, tp, "", {})
    stray = sorted(path for path in texts if not any(_matches(p, path) for p in patterns))
    if stray:
        raise ResultMismatch(f"text at undeclared path(s) {stray}")
    return texts


def rebuild(value: Any, tp: Any, updates: Mapping[str, str]) -> Any:
    """A copy of ``value`` (walked as ``tp``) with the text at each concrete path replaced."""
    copy, _texts = _walk(value, tp, "", updates)
    return copy


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def canonical_json(value: Any) -> str:
    """``value`` as one canonical JSON text: equal values give equal text.

    What the audit digest is taken over (``kernel/governance/audit/digest.py``, keyed). No
    fingerprint is computed here: an unkeyed hash of a short value (an email, a phone
    number) is reversed by guessing, so the only digest an audit row carries is the
    kernel's HMAC.
    """
    return json.dumps(_jsonable(value), sort_keys=True, default=str, ensure_ascii=False)


__all__ = [
    "ITEM",
    "ResultMismatch",
    "UndeclarableType",
    "canonical_json",
    "extract_fields",
    "rebuild",
    "text_paths",
]
