"""stock_quote tool: parses a sandbox fetch result into a clean quote string.

run_shell is injected so these tests never touch Docker or the network.
"""

from __future__ import annotations

from types import SimpleNamespace

from iris_harness.tools.stock_quote import stock_quote


def _result(stdout: str = "", *, exit_code: int = 0, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, exit_code=exit_code, stderr=stderr)


def test_formats_price_with_change() -> None:
    payload = '{"symbol":"AAPL","price":211.2,"currency":"USD","previous_close":208.0}'
    out = stock_quote("aapl", run_shell=lambda _cmd: _result(payload))
    assert "AAPL" in out
    assert "211.2" in out
    assert "USD" in out
    assert "%" in out  # change vs previous close
    assert "Yahoo Finance" in out


def test_runs_fixed_yahoo_fetch_for_the_symbol() -> None:
    seen: dict[str, str] = {}

    def fake(cmd: str):
        seen["cmd"] = cmd
        return _result('{"symbol":"MSFT","price":400.0,"currency":"USD"}')

    stock_quote("msft", run_shell=fake)
    assert "query1.finance.yahoo.com" in seen["cmd"]
    assert "MSFT" in seen["cmd"]  # symbol is passed to the script


def test_invalid_ticker_is_rejected_without_running() -> None:
    called = [False]

    def fake(_cmd: str):
        called[0] = True
        return _result("{}")

    out = stock_quote("not a ticker!!", run_shell=fake)
    assert "ticker" in out.lower()
    assert not called[0]


def test_error_payload_degrades_gracefully() -> None:
    out = stock_quote("AAPL", run_shell=lambda _c: _result('{"error":"timeout"}'))
    assert "unavailable" in out.lower()


def test_sandbox_unavailable_degrades_gracefully() -> None:
    out = stock_quote(
        "AAPL",
        run_shell=lambda _c: _result("", exit_code=127, stderr="sandbox unavailable: docker"),
    )
    assert "unavailable" in out.lower()


def test_missing_price_field() -> None:
    out = stock_quote("AAPL", run_shell=lambda _c: _result('{"symbol":"AAPL","price":null}'))
    assert "no quote" in out.lower()
