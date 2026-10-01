"""pytest plugin: make the optional ``ml`` extra unimportable, the way CI saw it.

CI installed from ``poetry.lock`` with no extras, so ``sentence-transformers`` and
its torch stack were simply absent; this dev box installs with ``-E ml``, so they
are present. That difference is the one real gap between a local run and the
hosted one: a module-level import of an ml-only package passes here and would have
failed there. All three import sites in the harness are deliberately lazy and
degrade with a clear message, and this plugin is how we keep them that way.

Used by ``scripts/ci_local.sh --no-ml``:

    PYTHONPATH=scripts pytest -p ci_no_ml ...
"""

from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from typing import Any

# The packages `poetry install -E ml` adds and a no-extras install does not.
BLOCKED = (
    "sentence_transformers",
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
)


class _AbsentFinder(MetaPathFinder):
    """Raise ModuleNotFoundError for the blocked roots and their submodules."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> ModuleSpec | None:
        root = fullname.split(".", 1)[0]
        if root in BLOCKED:
            raise ModuleNotFoundError(
                f"No module named {root!r} (hidden by scripts/ci_no_ml.py: CI installs "
                f"without the 'ml' extra, so this import must stay lazy and degrade)",
                name=fullname,
            )
        return None


def pytest_configure(config: Any) -> None:
    for name in list(sys.modules):
        if name.split(".", 1)[0] in BLOCKED:
            del sys.modules[name]
    sys.meta_path.insert(0, _AbsentFinder())
