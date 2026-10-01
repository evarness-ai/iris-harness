"""Live stock-quote tool — fetches a current price via the code_exec sandbox.

Web-search snippets don't carry a live price, so this tool runs a FIXED httpx
fetch of Yahoo Finance's key-less chart endpoint *inside the Docker sandbox*
(governed network egress; no LLM code generation, no raw outbound HTTP from the
main process). It returns a clean one-line quote string and never raises — so
the ReAct loop can pass the result straight into a prompt.

Requires ``query1.finance.yahoo.com`` on the sandbox egress allowlist (added to
DEFAULT_ALLOWLIST in iris_harness.tools.sandbox.egress).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from iris_harness.foundation.process_state import track_globals

logger = logging.getLogger(__name__)

# Tickers: letters/digits plus the few symbols real tickers use (BRK-B, ^GSPC, EURUSD=X).
_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-^=]{1,15}$")

# Fixed fetch script run in the sandbox. argv[1] = ticker. Prints one JSON line.
_FETCH_SCRIPT = r"""
import sys, json
try:
    import httpx
    sym = sys.argv[1]
    r = httpx.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
        params={"interval": "1d", "range": "1d"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    r.raise_for_status()
    meta = r.json()["chart"]["result"][0]["meta"]
    print(json.dumps({
        "symbol": meta.get("symbol", sym),
        "price": meta.get("regularMarketPrice"),
        "currency": meta.get("currency", ""),
        "previous_close": meta.get("chartPreviousClose") or meta.get("previousClose"),
        "exchange": meta.get("exchangeName", ""),
    }))
except Exception as exc:  # noqa: BLE001
    print(json.dumps({"error": str(exc)}))
"""

_host: Any = None


def _default_run_shell(cmd: str) -> Any:
    """Lazy, reused sandbox host so repeated quotes don't re-create the workspace."""
    global _host
    from iris_harness.tools.sandbox_tools import SandboxToolHost

    if _host is None:
        _host = SandboxToolHost("iris-stock-quote")
    return _host.run_shell(cmd)


def _parse_last_json(text: str) -> dict[str, Any] | None:
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                return loaded
    return None


def stock_quote(symbol: str, *, run_shell: Callable[[str], Any] | None = None) -> str:
    """Return a live quote for ``symbol`` (e.g. "AAPL: 211.20 USD (+1.5% vs prev 208.00)").

    Always returns a string; never raises. ``run_shell`` is injectable for tests.
    """
    sym = (symbol or "").strip().upper()
    if not _TICKER_RE.match(sym):
        return "Error: stock_quote needs a ticker symbol, e.g. AAPL."

    runner = run_shell or _default_run_shell
    # Feed the script over stdin via heredoc; argv[1] carries the ticker.
    cmd = f"python3 - {sym} <<'IRIS_QUOTE_EOF'\n{_FETCH_SCRIPT}\nIRIS_QUOTE_EOF"
    try:
        res = runner(cmd)
    except Exception as exc:
        logger.exception("stock_quote sandbox run failed")
        return f"Stock quote unavailable: {exc}"

    stdout = str(getattr(res, "stdout", "") or "").strip()
    exit_code = int(getattr(res, "exit_code", 1) or 0)
    if not stdout:
        stderr = str(getattr(res, "stderr", "") or "").strip()
        return f"Stock quote unavailable: {stderr or 'sandbox returned no output'}"

    data = _parse_last_json(stdout)
    if data is None:
        return f"Stock quote unavailable: could not parse result ({stdout[:120]})"
    if data.get("error"):
        return f"Stock quote unavailable for {sym}: {data['error']}"

    price = data.get("price")
    if price is None:
        return f"No quote found for {sym}. Check the ticker symbol."
    if exit_code != 0:
        logger.warning("stock_quote nonzero exit (%s) but parsed a price", exit_code)

    currency = str(data.get("currency") or "").strip()
    prev = data.get("previous_close")
    change = ""
    if isinstance(price, int | float) and isinstance(prev, int | float) and prev:
        change = f" ({(price - prev) / prev * 100:+.2f}% vs prev close {prev})"
    out_sym = str(data.get("symbol") or sym)
    tail = f"{currency}{change}".strip()
    return f"{out_sym}: {price} {tail}".rstrip() + "  [source: Yahoo Finance, via sandbox]"


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_host")
