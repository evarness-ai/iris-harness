"""CI guard: "now" and "today" come from the owner's clock, not the machine's.

``datetime.now().astimezone()`` and ``date.today()`` read the MACHINE's zone. On the VM
that happens to match ``IRIS_TZ`` only because its compose sets ``TZ`` from it; any other
install (a plain Docker run, a laptop in another zone) got "today" in the wrong zone
while the digest and reminders used ``IRIS_TZ``. Use
``iris_harness.foundation.clock.local_now()`` / ``local_today()`` (plugins:
``iris_harness.sdk.time``), or ``datetime.now(<zone>)`` when a zone is meant.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve()
while not (_ROOT / "src" / "iris_harness").is_dir():
    if _ROOT == _ROOT.parent:
        raise RuntimeError("could not locate repo root (src/iris_harness)")
    _ROOT = _ROOT.parent

# The one place allowed to read the machine's zone: the fallback when IRIS_TZ is unset.
ALLOWED = {"src/iris_harness/foundation/clock.py"}


def _violations(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = ast.unparse(node.func)
        naive_now = func in ("datetime.now", "datetime.datetime.now") and not (
            node.args or node.keywords
        )
        if naive_now or func in ("date.today", "datetime.date.today"):
            out.append(f"{path.relative_to(_ROOT)}:{node.lineno}  {func}()")
    return out


def test_nothing_reads_the_machine_clock_for_the_owner() -> None:
    files = [p for p in (_ROOT / "src").rglob("*.py") if str(p.relative_to(_ROOT)) not in ALLOWED]
    assert files, "no source found -- the guard checked nothing"
    found = [v for p in files for v in _violations(p)]
    assert not found, (
        "these read 'now'/'today' in the machine's zone, not the owner's (IRIS_TZ). Use "
        "foundation.clock.local_now()/local_today() (plugins: sdk.time), or "
        "datetime.now(<zone>):\n  " + "\n  ".join(found)
    )


def test_the_guard_catches_both_forms(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text(
        "from datetime import date, datetime\n"
        "a = datetime.now().astimezone()\n"
        "b = date.today()\n"
        "c = datetime.now(UTC)\n"
    )
    global _ROOT
    saved, _ROOT = _ROOT, tmp_path
    try:
        assert len(_violations(bad)) == 2
    finally:
        _ROOT = saved
