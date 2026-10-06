"""``weather.forecast``: the catalogue's first published capability (L4.5 hardening)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

from iris_harness.foundation.capabilities import (
    CAPABILITIES,
    Forecast,
    ForecastPeriod,
    published_capability,
)
from iris_harness.sdk import capabilities as sdk_capabilities


def test_the_catalogue_publishes_weather_forecast() -> None:
    spec = published_capability("weather.forecast")
    assert spec is not None
    assert "weather.forecast" in CAPABILITIES
    assert sdk_capabilities.CAPABILITIES is CAPABILITIES


def test_forecast_is_a_governed_external_read() -> None:
    spec = CAPABILITIES["weather.forecast"]
    method = spec.methods["forecast"]
    assert (method.effect, method.content, method.confirm_mode) == ("read", "external", "never")
    assert set(method.fields) == {"location", "periods.[].summary"}
    assert spec.shapes["forecast"] == "async"
    assert spec.fan_out is None  # one provider


def test_a_conforming_provider_is_accepted_and_a_partial_one_is_not() -> None:
    spec = CAPABILITIES["weather.forecast"]

    class Good:
        async def forecast(self, location: str, days: int = 3) -> Forecast:
            raise NotImplementedError

    class Sync:
        def forecast(self, location: str, days: int = 3) -> Forecast:
            raise NotImplementedError

    assert spec.missing_members(Good()) == ()
    assert spec.missing_members(Sync()) != ()
    assert spec.missing_members(object()) == ("forecast",)


def test_the_records_are_frozen_plain_data() -> None:
    now = datetime(2026, 10, 5, tzinfo=UTC)
    period = ForecastPeriod(now, now, 12.5, None, None, "Light rain")
    forecast = Forecast("Lisbon, Portugal", now, (period,))
    assert dataclasses.is_dataclass(forecast) and dataclasses.is_dataclass(period)
    assert forecast.periods[0].summary == "Light rain"
