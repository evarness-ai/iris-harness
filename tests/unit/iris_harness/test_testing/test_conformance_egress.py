"""The conformance suite fails a plugin whose example call contacted an undeclared host (#103)."""

from __future__ import annotations

import re
from typing import Any

import pytest

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.http import EgressDenied
from iris_harness.testing import (
    ConformanceError,
    assert_conformant,
    check_conformance,
    fake_http,
    plugin,
)

_URLS: dict[str, Any] = {
    "https://api.open-meteo.com/v1/forecast": {"json": {"ok": True}},
    "https://evil.example/collect": {"text": "should not be reached"},
}


def _weather(egress: dict[str, Any] | None) -> Any:
    def setup(api: PluginAPI) -> None:
        http = api.http

        def forecast(args: dict[str, Any]) -> str:
            try:
                return http.get(args["url"]).text
            except EgressDenied as exc:
                return f"denied: {exc.host}"

        api.register_tool("forecast", 'Forecast. Args: {"url": str}.', forecast)

    manifest: dict[str, Any] = {
        "name": "weather",
        "provides": ["tool"],
        "tools": {"forecast": {"effect": "read"}},
    }
    if egress is not None:
        manifest["egress"] = egress
    return plugin(setup, manifest=manifest)


_DECLARED = {"hosts": ["api.open-meteo.com"]}


def test_a_plugin_that_contacts_only_its_declared_host_conforms() -> None:
    with fake_http(_URLS):
        assert_conformant(
            _weather(_DECLARED),
            tools={"forecast": {"url": "https://api.open-meteo.com/v1/forecast"}},
        )


def test_an_undeclared_host_in_an_example_call_is_an_egress_violation() -> None:
    with fake_http(_URLS):
        violations = check_conformance(
            _weather(_DECLARED), tools={"forecast": {"url": "https://evil.example/collect"}}
        )
    assert [(v.check, v.subject) for v in violations] == [("egress", "forecast")]
    tokens = re.findall(r"[A-Za-z0-9.-]+", violations[0].detail)
    assert any(token == "evil.example" for token in tokens)


def test_a_plugin_with_no_egress_declaration_fails_on_every_host() -> None:
    with fake_http(_URLS), pytest.raises(ConformanceError) as raised:
        assert_conformant(
            _weather(None), tools={"forecast": {"url": "https://api.open-meteo.com/v1/forecast"}}
        )
    assert [v.check for v in raised.value.violations] == ["egress"]
    assert "declares no egress" in raised.value.violations[0].detail


# -- a capability provider's own requests are scoped to the capability call --------------
def _weather_provider(egress: dict[str, Any]) -> Any:
    from datetime import UTC, datetime

    from iris_harness.sdk.capabilities import Forecast, ForecastPeriod

    class Provider:
        def __init__(self, api: PluginAPI) -> None:
            self._http = api.http

        async def forecast(self, location: str, days: int = 3) -> Forecast:
            url = "https://api.open-meteo.com/v1/forecast"
            try:
                await self._http.arequest("GET", url, params={"q": location})
            except EgressDenied:
                pass
            now = datetime.now(UTC)
            period = ForecastPeriod(now, now, 12.0, None, None, "Clear")
            return Forecast(location=location, issued_at=now, periods=(period,))

    def setup(api: PluginAPI) -> None:
        api.provide("weather.forecast", Provider(api))

    return plugin(
        setup,
        manifest={
            "name": "weather",
            "capabilities": {"provides": ["weather.forecast"]},
            "egress": egress,
        },
    )


_FORECAST = {"weather.forecast": {"forecast": {"location": "Berlin"}}}


def test_a_capability_providers_request_is_recorded_under_the_capability_call() -> None:
    with fake_http(_URLS) as sent:
        assert_conformant(_weather_provider(_DECLARED), capabilities=_FORECAST)
    assert len(sent) == 1


def test_a_capability_provider_with_no_declared_host_fails_conformance() -> None:
    with fake_http(_URLS) as sent:
        violations = check_conformance(_weather_provider({}), capabilities=_FORECAST)
    assert sent == []
    assert [(v.check, v.subject) for v in violations] == [
        ("egress", "capability:weather.forecast.forecast")
    ]


def test_the_conformance_check_names_are_frozen() -> None:
    """``ConformanceCheck`` is a stable name: adding a check is additive, removing is not."""
    from typing import get_args

    from iris_harness.testing.conformance import ConformanceCheck

    assert set(get_args(ConformanceCheck)) == {
        "mount",
        "coverage",
        "audit",
        "caller",
        "approval",
        "example",
        "egress",
        "content",
    }
