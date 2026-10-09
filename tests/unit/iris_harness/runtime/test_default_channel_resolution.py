"""Default-channel fallback severity: an unset-up surface is not a typo (issue #110)."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from iris_harness.runtime.channel_wiring import _resolve_default_channel

_YAML = """\
default: telegram
channels:
  - name: console
    type: console
  - name: telegram
    type: telegram
    bot_token: ${TELEGRAM_BOT_TOKEN}
"""


def _runtime(tmp_path: Path) -> Any:
    (tmp_path / "channels.yaml").write_text(_YAML)
    channels = SimpleNamespace(channels=lambda: ["console", "web"])
    return SimpleNamespace(channels=channels, config_dir=tmp_path, default_channel="")


def test_declared_but_unconfigured_default_falls_back_without_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rt = _runtime(tmp_path)
    with caplog.at_level(logging.INFO, logger="iris_harness.runtime.channel_wiring"):
        _resolve_default_channel(rt, "telegram")
    assert rt.default_channel == "console"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_undeclared_default_still_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rt = _runtime(tmp_path)
    with caplog.at_level(logging.INFO, logger="iris_harness.runtime.channel_wiring"):
        _resolve_default_channel(rt, "telegramm")
    assert rt.default_channel == "console"
    assert [r for r in caplog.records if r.levelno == logging.WARNING]
