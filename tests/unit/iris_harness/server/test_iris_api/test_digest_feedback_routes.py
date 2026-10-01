"""``POST /api/digest/not-useful`` — the 👎 on a digest Focus line (loop-proof D17, V31).

Pinned: it writes the existing surface-suppression ledger (the Focus slot's own
consult then hides the sender), the next day's footer names it, it needs a credential
but is not a governed write, and it refuses a body that is not a sender address.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.server.iris_api.main import create_app
from iris_harness.services.digest.learned import learned_yesterday_line
from iris_harness.services.learning.suppression import (
    EMAIL_FOCUS_SURFACE,
    EMAIL_SEARCH_SUBSYSTEM,
    SurfaceFeedbackStore,
    email_focus_dims,
)

PATH = "/api/digest/not-useful"


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path))
    return tmp_path


def _client(tmp_path: Path, **kwargs: object) -> TestClient:
    return TestClient(
        create_app(runtime=SimpleNamespace(data_dir=tmp_path), auto_start_runtime=False),
        **kwargs,  # type: ignore[arg-type]
    )


def test_not_useful_suppresses_the_sender_in_focus(data_dir: Path) -> None:
    with _client(data_dir, headers=auth_headers()) as client:
        resp = client.post(PATH, json={"sender": "GoldenPi <News@goldenpi.example>"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True, "sender": "news@goldenpi.example", "hidden_from": "focus"}

    store = SurfaceFeedbackStore()
    store.ensure_schema()
    assert store.should_suppress(
        EMAIL_SEARCH_SUBSYSTEM, EMAIL_FOCUS_SURFACE, email_focus_dims("news@goldenpi.example")
    )


def test_the_next_days_footer_names_it(data_dir: Path) -> None:
    with _client(data_dir, headers=auth_headers()) as client:
        client.post(PATH, json={"sender": "news@goldenpi.example"})
    tomorrow = datetime.now(UTC) + timedelta(days=1)
    assert (
        learned_yesterday_line(now=tomorrow, tz=ZoneInfo("UTC"))
        == "learned yesterday: news@goldenpi.example hidden from Focus"
    )


def test_the_link_encoded_sender_round_trips(data_dir: Path) -> None:
    """The web button posts the sender it unquoted from ``iris:not-useful/<quoted>``."""
    from urllib.parse import quote, unquote

    sender = unquote(quote("news+digest@goldenpi.example", safe=""))
    with _client(data_dir, headers=auth_headers()) as client:
        assert client.post(PATH, json={"sender": sender}).json()["sender"] == sender


@pytest.mark.parametrize("sender", ["", "no-at-sign", "@goldenpi.example", "news@"])
def test_a_body_that_is_not_an_address_is_refused(data_dir: Path, sender: str) -> None:
    with _client(data_dir, headers=auth_headers()) as client:
        assert client.post(PATH, json={"sender": sender}).status_code == 422


def test_it_needs_a_credential(data_dir: Path) -> None:
    with _client(data_dir) as client:
        assert client.post(PATH, json={"sender": "news@goldenpi.example"}).status_code == 401


def test_it_is_not_a_governed_write(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Like /surface-feedback: a read-only phone can still hide noise from its digest."""
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    with _client(data_dir, headers=auth_headers()) as client:
        assert client.post(PATH, json={"sender": "news@goldenpi.example"}).status_code == 200
