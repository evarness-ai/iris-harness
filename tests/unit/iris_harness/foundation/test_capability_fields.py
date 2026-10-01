"""Capability result fields: what can be declared, the concrete path map, the copy rule.

And the spec rules that rest on them (plugin-capabilities §4): every Protocol method
declares its effect and exactly the text fields of its return type; opaque types, odd
signatures and a catalogue that can be edited at runtime are refused.
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
from collections.abc import AsyncIterator, Awaitable, Iterator, Sequence
from typing import Any, Literal, Protocol, TypedDict

import pytest
from pydantic import BaseModel, ConfigDict, PrivateAttr, computed_field

from iris_harness.foundation import capabilities as catalogue
from iris_harness.foundation.capabilities import CapabilitySpec, MethodSpec
from iris_harness.foundation.capability_fields import (
    ResultMismatch,
    UndeclarableType,
    canonical_json,
    extract_fields,
    rebuild,
    text_paths,
)


class Kind(enum.Enum):
    A = "a"


@dataclasses.dataclass(frozen=True)
class Sender:
    name: str
    address: str | None


@dataclasses.dataclass(frozen=True)
class Message:
    id: int
    subject: str
    sender: Sender
    labels: tuple[str, ...]
    kind: Kind
    when: datetime.datetime | None = None


class Note(BaseModel):
    model_config = ConfigDict(frozen=True)
    title: str
    words: int


class Row(TypedDict):
    body: str
    count: int


@dataclasses.dataclass
class HasInitFalse:
    a: str
    b: str = dataclasses.field(init=False, default="x")


@dataclasses.dataclass
class Tree:
    label: str
    children: list[Tree]


class Handle:
    pass


# ------------------------------------------------------------------ what can be declared
@pytest.mark.parametrize(
    ("tp", "paths"),
    [
        (str, {""}),
        (int, set()),
        (Kind, set()),
        (Literal["x", "y"], set()),
        (str | None, {""}),
        (list[str], {"[]"}),
        (Sequence[Note], {"[].title"}),
        (tuple[Row, ...], {"[].body"}),
        (
            list[Message],
            {"[].subject", "[].sender.name", "[].sender.address", "[].labels.[]"},
        ),
        (None, set()),
    ],
)
def test_text_paths_names_every_str_leaf(tp: Any, paths: set[str]) -> None:
    assert text_paths(type(None) if tp is None else tp) == paths


@pytest.mark.parametrize(
    "tp", [bytes, dict[str, str], Any, object, Handle, tuple[str, int], list, HasInitFalse, Tree]
)
def test_opaque_types_are_refused(tp: Any) -> None:
    with pytest.raises(UndeclarableType):
        text_paths(tp)


# ------------------------------------------------------------------- values at runtime
MSGS = [
    Message(1, "hello", Sender("Ann", "ann@example.test"), ("x", "y"), Kind.A),
    Message(2, "again", Sender("Bob", None), (), Kind.A),
]
PATTERNS = ("[].subject", "[].sender.name", "[].sender.address", "[].labels.[]")


def test_extract_gives_concrete_indexed_paths() -> None:
    assert extract_fields(MSGS, list[Message], PATTERNS) == {
        "[0].subject": "hello",
        "[0].sender.name": "Ann",
        "[0].sender.address": "ann@example.test",
        "[0].labels.[0]": "x",
        "[0].labels.[1]": "y",
        "[1].subject": "again",
        "[1].sender.name": "Bob",
    }


def test_text_at_an_undeclared_path_is_refused() -> None:
    with pytest.raises(ResultMismatch, match="undeclared"):
        extract_fields(MSGS, list[Message], ("[].subject",))


def test_rebuild_copies_frozen_dataclasses_and_leaves_the_original_alone() -> None:
    out = rebuild(MSGS, list[Message], {"[0].sender.address": "MASKED", "[1].subject": "MASKED"})
    assert out[0].sender.address == "MASKED" and out[1].subject == "MASKED"
    assert out[0].subject == "hello" and out[0].labels == ("x", "y") and out[0].kind is Kind.A
    assert MSGS[0].sender.address == "ann@example.test"  # the provider's object is untouched
    assert out[0] is not MSGS[0] and out[0].sender is not MSGS[0].sender
    assert isinstance(out, list) and isinstance(out[0].labels, tuple)


def test_rebuild_copies_pydantic_models_and_typed_dicts() -> None:
    note = Note(title="t", words=3)
    assert rebuild([note], list[Note], {"[0].title": "M"}) == [Note(title="M", words=3)]
    assert note.title == "t"
    row: Row = {"body": "b", "count": 1}
    new = rebuild(row, Row, {"body": "M"})
    assert new == {"body": "M", "count": 1} and row["body"] == "b" and new is not row


def test_rebuild_of_a_bare_string() -> None:
    assert rebuild("secret", str, {"": "M"}) == "M"


# ------------------------------------- strict at runtime: the value must BE the declared type
@dataclasses.dataclass(frozen=True)
class SneakyMessage(Message):
    hidden: str = "extra text nobody declared"


@dataclasses.dataclass(slots=True)
class SlotsNote:
    title: str


@dataclasses.dataclass(slots=True)
class SneakySlotsNote(SlotsNote):
    hidden: str = "no __dict__ to catch this one by"


class SneakyNote(Note):
    hidden: str = "a model subclass with an undeclared field"


class LooseNote(BaseModel):
    model_config = ConfigDict(extra="allow")
    title: str


class Counted(BaseModel):
    words: int


@dataclasses.dataclass
class Plain:
    title: str


def _plain_with_stray_attribute() -> Plain:
    value = Plain("t")
    value.stray = "undeclared text"  # type: ignore[attr-defined]
    return value


class EvilInt(int):
    pass


class EvilDatetime(datetime.datetime):
    pass


def _evil_int() -> EvilInt:
    value = EvilInt(3)
    value.secret = "SECRET"  # type: ignore[attr-defined]
    return value


def _evil_datetime() -> EvilDatetime:
    value = EvilDatetime(2026, 1, 1)
    value.secret = "SECRET"  # type: ignore[attr-defined]
    return value


class Opaque:
    title = "the provider's own live object"


@pytest.mark.parametrize(
    ("value", "tp", "patterns"),
    [
        # a subclass with an extra str field (the verifier's first reproduction)
        ([SneakyMessage(1, "s", Sender("n", None), (), Kind.A)], list[Message], PATTERNS),
        # subclasses whose extra field no instance __dict__ reveals: only the exact-class
        # rule catches these
        (SneakySlotsNote("t"), SlotsNote, ("title",)),
        (SneakyNote(title="t", words=1), Note, ("title",)),
        # a declared type that is not walkable at all (the spec refuses it; the walk too)
        (b"raw", bytes, ()),
        # pydantic extra="allow"
        (LooseNote(title="t", secret="extra"), LooseNote, ("title",)),  # type: ignore[call-arg]
        # TypedDict extra keys
        ({"body": "b", "count": 1, "leak": "extra"}, Row, ("body",)),
        # a str in an int field
        (Counted.model_construct(words="many"), Counted, ()),
        # an opaque object returned as the provider's own instance
        (Opaque(), Note, ("title",)),
        # instance attributes that are not dataclass fields
        (_plain_with_stray_attribute(), Plain, ("title",)),
        # a list where the type says tuple, and a tuple where it says list
        (["a"], tuple[str, ...], ("[]",)),
        (("a",), list[str], ("[]",)),
        # an int where the type says str
        (7, str, ("",)),
        # leaf subclasses carrying undeclared attributes (the re-review's first leak)
        (_evil_int(), int, ()),
        (_evil_datetime(), datetime.datetime, ()),
        # numbers are exact: a bool is not an int, an int is not a float
        (True, int, ()),
        (1, float, ()),
    ],
)
def test_values_that_are_not_their_declared_type_are_refused(
    value: Any, tp: Any, patterns: tuple[str, ...]
) -> None:
    with pytest.raises(ResultMismatch):
        extract_fields(value, tp, patterns)
    with pytest.raises(ResultMismatch):
        rebuild(value, tp, {})


def test_leaves_come_back_fresh_and_exact() -> None:
    from decimal import Decimal

    amount = Decimal("12.50")
    out = rebuild([amount], list[Decimal], {})
    assert out == [amount] and out[0] is not amount and type(out[0]) is Decimal
    assert rebuild(1.5, float, {}) == 1.5 and rebuild(True, bool, {}) is True


class Hidden(BaseModel):
    title: str
    _secret: str = PrivateAttr(default="")


def test_a_model_declaring_private_attributes_is_refused_at_spec_time() -> None:
    with pytest.raises(UndeclarableType, match="private attributes"):
        text_paths(Hidden)


@dataclasses.dataclass
class RequiredInitVar:
    title: str
    seed: dataclasses.InitVar[str]


@dataclasses.dataclass
class OptionalInitVar:
    title: str
    seed: dataclasses.InitVar[str | None] = None


@dataclasses.dataclass
class PostInit:
    title: str

    def __post_init__(self) -> None:
        if "[redacted" in self.title:
            raise ValueError("rejects masked text")


class ModelPostInit(BaseModel):
    title: str

    def model_post_init(self, context: Any, /) -> None:
        pass


@pytest.mark.parametrize(
    ("tp", "match"),
    [
        (RequiredInitVar, "InitVar"),
        (OptionalInitVar, "InitVar"),
        (PostInit, "__post_init__"),
        (ModelPostInit, "model_post_init"),
    ],
)
def test_records_with_construction_hooks_are_refused_at_spec_time(tp: Any, match: str) -> None:
    with pytest.raises(UndeclarableType, match=match):
        text_paths(tp)


def test_a_failure_building_the_copy_is_a_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise ValueError("cannot build")

    value = Note(title="t", words=1)
    monkeypatch.setattr(Note, "model_construct", refuse)
    with pytest.raises(ResultMismatch, match="could not be rebuilt"):
        rebuild(value, Note, {"title": "M"})


# ------------------------------- no user code runs in the consumer's path: the copy is bare
INIT_CALLS: list[str] = []


@dataclasses.dataclass(init=False)
class HandInit:
    title: str

    def __init__(self, title: str) -> None:
        INIT_CALLS.append(title)  # a side effect: must run on the provider's side only
        self.title = title


@dataclasses.dataclass(init=False, frozen=True)
class FrozenHandInit:
    title: str

    def __init__(self, title: str) -> None:
        INIT_CALLS.append(title)
        object.__setattr__(self, "title", title)


@dataclasses.dataclass(init=False, slots=True)
class SlotsHandInit:
    title: str

    def __init__(self, title: str) -> None:
        INIT_CALLS.append(title)
        self.title = title


@pytest.mark.parametrize("tp", [HandInit, FrozenHandInit, SlotsHandInit])
def test_the_copy_runs_no_hand_written_init(tp: Any) -> None:
    INIT_CALLS.clear()
    provided = tp("my key")  # the provider builds its value: the side effect runs once
    assert text_paths(tp) == {"title"}
    out = rebuild(provided, tp, {"title": "[masked]"})
    assert INIT_CALLS == ["my key"]  # ... and never again in the consumer's path
    assert type(out) is tp and out is not provided and out.title == "[masked]"
    assert provided.title == "my key"


# ------------------------------------------------- reading the provider's fields is guarded
@dataclasses.dataclass
class WithFactory:
    title: str
    tags: list[str] = dataclasses.field(default_factory=list)


def test_a_slots_value_built_bare_is_a_mismatch_not_a_raw_error() -> None:
    bare = object.__new__(SlotsNote)  # never initialised: its slot is empty
    with pytest.raises(ResultMismatch, match="could not be read"):
        extract_fields(bare, SlotsNote, ("title",))


def test_a_default_factory_never_run_is_a_mismatch_not_a_raw_error() -> None:
    value = object.__new__(WithFactory)
    object.__setattr__(value, "title", "t")  # tags was never set
    with pytest.raises(ResultMismatch, match="could not be read"):
        rebuild(value, WithFactory, {})


# ------------------------------------------------ plain data by construction: no code on access
class _Desc:
    def __get__(self, obj: Any, owner: Any = None) -> str:
        return "computed"

    def __set__(self, obj: Any, value: Any) -> None:
        pass


@dataclasses.dataclass
class DescriptorField:
    title: str = _Desc()  # type: ignore[assignment]


@dataclasses.dataclass
class OverridesGetattr:
    title: str

    def __getattr__(self, name: str) -> Any:
        return "anything"


@dataclasses.dataclass
class OverridesGetattribute:
    title: str

    def __getattribute__(self, name: str) -> Any:
        return object.__getattribute__(self, name)


@dataclasses.dataclass
class OverridesSetattr:
    title: str

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)


class ModelOverridesSetattr(BaseModel):
    title: str

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)


class ModelComputed(BaseModel):
    title: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def shout(self) -> str:
        return self.title.upper()


@dataclasses.dataclass
class TextProperty:
    title: str

    @property
    def shout(self) -> str:
        return self.title.upper()


@dataclasses.dataclass
class UnannotatedProperty:
    title: str

    @property
    def shout(self):  # type: ignore[no-untyped-def]
        return self.title.upper()


@dataclasses.dataclass
class NumberProperty:
    title: str

    @property
    def length(self) -> int:
        return len(self.title)


@pytest.mark.parametrize(
    ("tp", "match"),
    [
        (DescriptorField, "data descriptor"),
        (OverridesGetattr, "__getattr__"),
        (OverridesGetattribute, "__getattribute__"),
        (OverridesSetattr, "__setattr__"),
        (ModelOverridesSetattr, "__setattr__"),
        (ModelComputed, "computed field"),
        (TextProperty, "property 'shout'"),
        (UnannotatedProperty, "property 'shout'"),
    ],
)
def test_records_whose_access_runs_code_are_refused_at_spec_time(tp: Any, match: str) -> None:
    with pytest.raises(UndeclarableType, match=match):
        text_paths(tp)


def test_what_stays_allowed() -> None:
    # a frozen dataclass's own __setattr__ guard, slot members, and a non-text property
    assert text_paths(Message) >= {"subject"}
    assert text_paths(SlotsNote) == {"title"}
    assert text_paths(NumberProperty) == {"title"}


def test_a_rebuilt_model_carries_no_state_of_the_providers_instance() -> None:
    original = Note(title="t", words=1)
    # state no field declares, planted where pydantic keeps private attributes
    object.__setattr__(original, "__pydantic_private__", {"_secret": "SECRET"})
    out = rebuild(original, Note, {"title": "M"})
    assert out == Note(title="M", words=1) and out is not original
    assert not getattr(out, "__pydantic_private__", None)


def test_an_optional_field_takes_either_arm() -> None:
    assert extract_fields(Sender("n", None), Sender, ("name", "address")) == {"name": "n"}


def test_canonical_json_is_stable_across_equal_values() -> None:
    # What the keyed audit digest is taken over (kernel/governance/audit/digest.py).
    assert canonical_json(MSGS) == canonical_json(list(MSGS))
    assert canonical_json(MSGS) != canonical_json(MSGS[:1])


# --------------------------------------------------------------------- the spec rules
class Mail(Protocol):
    def search(self, query: str, limit: int = 10) -> list[Message]: ...
    async def fetch(self, id: int) -> Message | None: ...
    def watch(self) -> AsyncIterator[Message]: ...
    def pages(self) -> Iterator[Note]: ...
    def mark(self, id: int) -> None: ...


MAIL_METHODS = {
    "search": MethodSpec(fields=PATTERNS),
    "fetch": MethodSpec(fields=("subject", "sender.name", "sender.address", "labels.[]")),
    "watch": MethodSpec(fields=("subject", "sender.name", "sender.address", "labels.[]")),
    "pages": MethodSpec(fields=("title",)),
    "mark": MethodSpec(effect="write", confirm="never"),
}


def test_a_full_spec_is_accepted_with_each_method_shape() -> None:
    spec = CapabilitySpec(name="test.mail", protocol=Mail, methods=MAIL_METHODS)
    assert dict(spec.shapes) == {
        "search": "value",
        "fetch": "async",
        "watch": "astream",
        "pages": "stream",
        "mark": "value",
    }
    assert spec.methods["mark"].confirm_mode == "never"
    assert MethodSpec(effect="write").confirm_mode == "once"


def _with(**changes: MethodSpec) -> dict[str, MethodSpec]:
    return {**MAIL_METHODS, **changes}


@pytest.mark.parametrize(
    ("methods", "match"),
    [
        (_with(search=MethodSpec(fields=("[].subject",))), "undeclared"),  # a text field missing
        (_with(pages=MethodSpec(fields=("title", "words"))), "not text fields"),
        ({k: v for k, v in MAIL_METHODS.items() if k != "mark"}, "needs one MethodSpec"),
    ],
)
def test_fields_must_be_exactly_the_text_fields(methods: dict[str, MethodSpec], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CapabilitySpec(name="test.mail", protocol=Mail, methods=methods)


class Opaque(Protocol):
    def raw(self) -> bytes: ...


class NoReturn(Protocol):
    def f(self):  # type: ignore[no-untyped-def]
        ...


class Varargs(Protocol):
    def f(self, *items: str) -> str: ...


class AsyncStream(Protocol):
    async def f(self) -> AsyncIterator[str]: ...


class SyncAwaitable(Protocol):
    def f(self) -> Awaitable[str]: ...


@pytest.mark.parametrize(
    ("protocol", "match"),
    [
        (Opaque, "not plain data"),
        (NoReturn, "return type"),
        (Varargs, "plain named"),
        (AsyncStream, "async method returns a value"),
        (SyncAwaitable, "must be declared async def"),
    ],
)
def test_undeclarable_methods_are_refused_when_the_spec_is_defined(
    protocol: type[Any], match: str
) -> None:
    member = "raw" if protocol is Opaque else "f"
    with pytest.raises(ValueError, match=match):
        CapabilitySpec(name="test.bad", protocol=protocol, methods={member: MethodSpec()})


def test_confirm_only_for_writes() -> None:
    with pytest.raises(ValueError):
        MethodSpec(effect="read", confirm="once")


def test_the_catalogue_is_read_only_at_runtime() -> None:
    with pytest.raises(TypeError):
        catalogue.CAPABILITIES["x.y"] = None  # type: ignore[index]


def test_a_provider_must_match_each_method_shape() -> None:
    spec = CapabilitySpec(name="test.mail", protocol=Mail, methods=MAIL_METHODS)

    class Provider:
        def search(self, query: str, limit: int = 10) -> list[Message]:
            return []

        def fetch(self, id: int) -> None:  # sync where the Protocol is async
            return None

        async def watch(self) -> AsyncIterator[Message]:
            yield MSGS[0]

        def pages(self) -> Iterator[Note]:
            return iter(())

        def mark(self, id: int) -> None:
            return None

    assert spec.missing_members(Provider()) == ("fetch (async in one, sync in the other)",)
