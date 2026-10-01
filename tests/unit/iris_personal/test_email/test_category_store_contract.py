"""Tests for the cross-cutting Category Pydantic contract (Track 1F)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from iris_personal.email.category_store import (
    ALLOWED_ROOTS,
    ALLOWED_TYPES,
    Category,
    make_path,
)


def _kw(**overrides):  # type: ignore[no-untyped-def]
    base = {
        "path": "email/shopping/apparel/outlet-brand",
        "type": "email",
        "root": "shopping",
        "branch": "apparel",
        "leaf": "outlet-brand",
    }
    base.update(overrides)
    return base


def test_minimal_valid() -> None:
    c = Category(**_kw())
    assert c.path == "email/shopping/apparel/outlet-brand"
    assert c.active is True
    assert c.sensitivity == "low"
    assert c.account_id is None
    assert c.metadata == {}


def test_make_path_composes_lowercase() -> None:
    p = make_path("Email", "Shopping", "Apparel", "Outlet-Brand")
    assert p == "email/shopping/apparel/outlet-brand"


def test_make_path_rejects_empty_parts() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        make_path("email", "shopping", "", "gap")


def test_type_must_be_allowed() -> None:
    with pytest.raises(ValidationError, match="type must be one of"):
        Category(**_kw(type="bogus"))
    # Every allowed type can be set.
    for t in ALLOWED_TYPES:
        path = f"{t}/shopping/apparel/outlet-brand"
        Category(**_kw(path=path, type=t))


def test_root_must_be_allowed() -> None:
    with pytest.raises(ValidationError, match="root must be one of"):
        Category(**_kw(root="retail"))
    # Every allowed root can be set.
    for r in ALLOWED_ROOTS:
        Category(**_kw(path=f"email/{r}/x/y", root=r))


def test_sensitivity_must_be_allowed() -> None:
    Category(**_kw(sensitivity="high"))
    with pytest.raises(ValidationError, match="sensitivity"):
        Category(**_kw(sensitivity="extreme"))


def test_path_must_be_lowercase() -> None:
    with pytest.raises(ValidationError, match="lowercase"):
        Category(**_kw(path="Email/Shopping/Apparel/Outlet-Brand"))


def test_cohesion_range_invariant() -> None:
    Category(**_kw(cohesion=0.8))
    with pytest.raises(ValidationError):
        Category(**_kw(cohesion=1.5))
    with pytest.raises(ValidationError):
        Category(**_kw(cohesion=-0.01))


def test_immutability() -> None:
    c = Category(**_kw())
    with pytest.raises(ValidationError):
        c.active = False  # type: ignore[misc]


def test_metadata_can_carry_arbitrary_payload() -> None:
    c = Category(
        **_kw(
            metadata={
                "top_domains": [("gap.com", 45)],
                "naming_rationale": "dominated by gap.com",
                "cluster_id": 7,
            }
        )
    )
    assert c.metadata["cluster_id"] == 7
