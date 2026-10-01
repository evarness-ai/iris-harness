"""The exfiltration guard's identity corpus, and the seam that supplies it.

M6.3 inverted `network_egress` -> `identity.loader`: the kernel asks for the user's
identity documents instead of reading them, which is what let `identity/` fold into
`memory/`. An inversion like that can be left unregistered, and an unregistered
exfiltration guard redacts nothing while reporting success -- strictly worse than the
import it replaced.

So these tests are the price of the inversion. The one that matters is
`test_a_built_runtime_registers_the_provider`: it fails if anyone drops the
registration, which is the only way the guard goes quiet.
"""

from __future__ import annotations

import pytest

from iris_harness.kernel.governance.identity_redaction import (
    clear_identity_text_provider,
    has_identity_text_provider,
    identity_texts,
    register_identity_text_provider,
)
from iris_harness.kernel.governance.plugins.network_egress import NetworkEgress


@pytest.fixture(autouse=True)
def _isolate_registry(owner_identity_seam):
    """Each test starts with no provider and leaves the real ones restored."""
    NetworkEgress._reset_identity_cache()
    yield
    NetworkEgress._reset_identity_cache()


def test_unregistered_is_none_not_empty() -> None:
    """`None` means "nobody told me"; `[]` means "nothing to redact". Different."""
    assert identity_texts() is None
    assert not has_identity_text_provider()

    register_identity_text_provider(list)
    assert identity_texts() == []
    assert has_identity_text_provider()


def test_the_guard_recognises_secret_shaped_literals_from_the_corpus() -> None:
    register_identity_text_provider(lambda: ["my key is sk-live-abc123def456ghi and more"])
    NetworkEgress._reset_identity_cache()

    literals = NetworkEgress._identity_secret_literals()
    assert "sk-live-abc123def456ghi" in literals
    # prose without digits is not secret-shaped, and must not become a deny-list entry
    assert not any(tok.isalpha() for tok in literals)


def test_an_unregistered_guard_warns_and_does_not_cache() -> None:
    """The quiet failure this whole seam is designed around.

    With no provider the guard yields no literals -- unavoidable, it has no corpus --
    but it must SAY so, and it must not memoise the emptiness, or a provider
    registered a moment later would never take effect.
    """
    assert NetworkEgress._identity_secret_literals() == frozenset()

    register_identity_text_provider(lambda: ["token abc123def456ghi789"])
    assert "abc123def456ghi789" in NetworkEgress._identity_secret_literals()


def test_a_provider_that_raises_degrades_without_breaking_the_check() -> None:
    def boom() -> list[str]:
        raise RuntimeError("unreadable identity document")

    register_identity_text_provider(boom)
    assert identity_texts() == []
    assert NetworkEgress._identity_secret_literals() == frozenset()


def test_a_built_runtime_registers_the_provider() -> None:
    """THE test. Importing the composition root must fill the seam.

    If this fails, the exfiltration guard has no corpus in production and denies
    nothing it should deny -- silently. Nothing else in the suite would notice.
    """
    import iris_harness.kernel.governance.identity_redaction as mod

    clear_identity_text_provider()
    assert not has_identity_text_provider()

    import importlib

    importlib.reload(importlib.import_module("iris_harness.runtime.identity_redaction"))
    assert has_identity_text_provider(), (
        "iris_harness.runtime.identity_redaction did not register a provider; "
        "the egress guard would redact nothing"
    )
    assert mod._provider is not None


def test_bootstrap_imports_the_registering_module() -> None:
    """And the composition root must actually import it.

    Registration in a module nobody imports is registration that never happens, which
    is exactly how this class of seam fails in practice.
    """
    import ast

    from iris_harness.foundation.paths import repo_root

    source = (repo_root() / "src/iris_harness/runtime/bootstrap.py").read_text(encoding="utf-8")
    imported = {
        node.names[0].name
        for node in ast.parse(source).body
        if isinstance(node, ast.Import) and node.names
    }
    assert "iris_harness.runtime.identity_redaction" in imported
