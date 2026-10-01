"""Reading a boolean from the environment, one definition.

There were six copies of this across the tree when M6.3 counted them, in **three
variants that disagree**. Two read a deny-list ("is this not off"):

- `{"", "0", "false", "no", "off"}` — `FLAG=` (set but empty) is OFF.
  Used by `server/iris_api`, `foundation/observability/{instruments,otlp_setup}`.
- `{"0", "false", "no", "off"}` — `FLAG=` is **ON**.
  Used by `runtime/bootstrap` and `server/channel_gateway`.

`export IRIS_SOMETHING=` turned a feature on in the runtime and off in the API,
which is not a distinction anyone designed. **Settled 2026-09-14 by the owner: an
empty value is OFF, everywhere.** `export FLAG=` reads as blank to any operator, so
the two readings that treated it as "on" were the outliers and are gone.

The third is `runtime/learning_controls._env_flag_on`, which reads an **allow-list**
(`{"1", "true", "yes", "on"}`) and so answers a different question: "is this
explicitly on", not "is this not off". `FLAG=banana` is false to it and **true** to
both others. It is :func:`env_flag_on` here.

Both functions still exist on purpose, because they answer different questions --
"is this not off" versus "is this explicitly on" -- and `FLAG=banana` still separates
them. What is gone is the third reading and the `empty_is_false` switch that carried
it: there is one answer for the empty string now, so there is nothing to pass.
"""

from __future__ import annotations

import os

_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})

# A wildcard bind address answers on every interface but none of them *is* the
# address — nothing reachable sits at 0.0.0.0 itself. `IRIS_API_HOST=0.0.0.0`
# (from `iris serve --host 0.0.0.0`) is a bind address; a health probe or a local
# HTTP call needs the loopback address instead.
_WILDCARD_BIND_HOSTS = {
    "0.0.0.0": "127.0.0.1",  # noqa: S104 - detecting the bind address, not binding to it
    "::": "[::1]",
}


def probe_host(host: str) -> str:
    """``host`` translated for a local probe: a wildcard bind address becomes
    its loopback equivalent, anything else is returned unchanged."""
    return _WILDCARD_BIND_HOSTS.get(host, host)


def env_flag(name: str, *, default: bool) -> bool:
    """Read ``name`` as a boolean.

    Unset returns ``default``. Otherwise the value is false when it is blank or
    matches ``0``/``false``/``no``/``off`` (case- and whitespace-insensitive), and
    true otherwise -- so ``FLAG=banana`` is true. Use :func:`env_flag_on` where a
    flag must be opted into deliberately.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE_VALUES


def env_flag_on(name: str) -> bool:
    """True only when ``name`` is explicitly one of ``1``/``true``/``yes``/``on``.

    The allow-list reading. Unset, blank, and anything unrecognised are all false --
    so ``FLAG=banana`` is false here and true to :func:`env_flag`. Use this where a
    flag must be opted into deliberately; use :func:`env_flag` where anything but an
    explicit "off" should count as on.
    """
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


__all__ = ["env_flag", "env_flag_on", "probe_host"]
