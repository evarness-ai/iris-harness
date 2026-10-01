"""Schema tests for the personal-assistant manifest extensions.

Covers the six new fields and two new nested models added in commit
5af5590 per canonical doc §4.4 and ADRs 0008, 0009, 0016.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from iris_harness.tools.skills.models import SkillMaintainer, SkillManifest, SkillSource


def _minimal_kwargs() -> dict[str, str]:
    """Bare-minimum required fields for a valid SkillManifest."""
    return {
        "name": "test-skill",
        "version": "0.1.0",
        "description": "x",
        "author": "y",
        "license": "Apache-2.0",
    }


def test_minimal_manifest_uses_safe_defaults() -> None:
    """All new fields fall back to first-party-in-tree defaults."""
    m = SkillManifest(**_minimal_kwargs())

    assert m.default_enabled is True
    assert m.trust_level == "first-party"
    assert m.iris_compatibility is None
    assert m.source is None
    assert m.maintainers == ()
    assert m.homepage is None


def test_community_without_source_rejected() -> None:
    """ADR-0009: a community skill MUST declare a source block."""
    with pytest.raises(ValidationError):
        SkillManifest(**_minimal_kwargs(), trust_level="community")


def test_community_with_source_accepted() -> None:
    """Full community shape with source + maintainers parses cleanly."""
    m = SkillManifest(
        **_minimal_kwargs(),
        trust_level="community",
        source=SkillSource(url="https://github.com/foo/bar", ref="v0.1.0"),
        maintainers=(SkillMaintainer(name="Alice", contact="a@example.com"),),
    )

    assert m.trust_level == "community"
    assert m.source is not None
    assert m.source.url == "https://github.com/foo/bar"
    assert m.maintainers[0].name == "Alice"


def test_bad_trust_level_rejected() -> None:
    """Literal type catches typos in trust_level."""
    with pytest.raises(ValidationError):
        SkillManifest(**_minimal_kwargs(), trust_level="rusted")  # type: ignore[arg-type]


def test_retired_calendar_visibility_key_still_loads() -> None:
    """``calendar_visibility`` was never read and is gone; a manifest that still
    carries it loads unchanged (unknown keys are ignored), so no installed skill breaks."""
    m = SkillManifest(**_minimal_kwargs(), calendar_visibility="always")  # type: ignore[call-arg]
    assert not hasattr(m, "calendar_visibility")


def test_source_requires_url() -> None:
    """SkillSource.url is required (min_length=1)."""
    with pytest.raises(ValidationError):
        SkillSource(url="")


def test_maintainer_requires_name() -> None:
    """SkillMaintainer.name is required (min_length=1)."""
    with pytest.raises(ValidationError):
        SkillMaintainer(name="")


def test_first_party_with_source_allowed() -> None:
    """First-party skills MAY declare source; the validator only requires
    it when trust_level='community'."""
    m = SkillManifest(
        **_minimal_kwargs(),
        # default trust_level='first-party'
        source=SkillSource(url="https://github.com/iris/iris", ref="main"),
    )

    assert m.trust_level == "first-party"
    assert m.source is not None
    assert m.source.url == "https://github.com/iris/iris"
