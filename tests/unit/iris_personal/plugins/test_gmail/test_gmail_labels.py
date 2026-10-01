"""Gmail labels for the email judge (loop-proof PR 5): ``ensure_labels`` and
``modify_labels`` against a fake Gmail service. No network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httplib2
import pytest
from googleapiclient.errors import HttpError

from iris_personal.email.write_approvals import approve_mailbox_writes
from iris_personal.plugins.gmail import gmail_fetch
from iris_personal.plugins.gmail.provider import GmailProvider

ACCT = "gmail:owner@example.com"


@pytest.fixture(autouse=True)
def _writes_approved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests are about what Gmail does once the owner approved mailbox writes
    (R4); the gate itself is ``test_gmail_write_gate.py``'s."""
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path / "data"))
    approve_mailbox_writes(ACCT, "test")


def _error(status: int, reason: str = "") -> HttpError:
    body = json.dumps(
        {
            "error": {
                "code": status,
                "message": "e",
                "errors": [{"reason": reason}] if reason else [],
            }
        }
    ).encode()
    return HttpError(httplib2.Response({"status": str(status)}), body)


class _Req:
    def __init__(self, run: Any) -> None:
        self._run = run

    def execute(self) -> Any:
        return self._run()


class _Labels:
    def __init__(self, existing: dict[str, str]) -> None:
        self.by_name = dict(existing)
        self.list_calls = 0
        self.created: list[dict[str, Any]] = []

    def list(self, *, userId: str) -> _Req:
        def run() -> dict[str, Any]:
            self.list_calls += 1
            return {"labels": [{"id": i, "name": n} for n, i in self.by_name.items()]}

        return _Req(run)

    def create(self, *, userId: str, body: dict[str, Any]) -> _Req:
        def run() -> dict[str, Any]:
            self.created.append(body)
            new_id = f"Label_{len(self.by_name) + 1}"
            self.by_name[body["name"]] = new_id
            return {"id": new_id, "name": body["name"]}

        return _Req(run)


class _Messages:
    def __init__(self, errors: list[HttpError] | None = None) -> None:
        self.bodies: list[dict[str, Any]] = []
        self._errors = list(errors or [])

    def batchModify(self, *, userId: str, body: dict[str, Any]) -> _Req:
        def run() -> None:
            self.bodies.append(body)
            if self._errors:
                raise self._errors.pop(0)

        return _Req(run)


class _Service:
    def __init__(self, labels: dict[str, str] | None = None, errors: list[HttpError] | None = None):
        self.labels_api = _Labels(labels or {})
        self.messages_api = _Messages(errors)

    def users(self) -> Any:
        svc = self

        class _Users:
            def labels(self) -> _Labels:
                return svc.labels_api

            def messages(self) -> _Messages:
                return svc.messages_api

        return _Users()


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Any:
    gmail_fetch.forget_labels()
    sleeps: list[float] = []
    monkeypatch.setattr(gmail_fetch.time, "sleep", sleeps.append)
    yield sleeps
    gmail_fetch.forget_labels()


def _use(monkeypatch: pytest.MonkeyPatch, service: _Service) -> None:
    monkeypatch.setattr(gmail_fetch, "_service_for", lambda account_id: service)


# ─── ensure_labels ───────────────────────────────────────────────────────


def test_ensure_labels_creates_the_missing_ones_and_their_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc = _Service({"Receipts": "Label_1", "IRIS/Bill": "Label_2"})
    _use(monkeypatch, svc)

    got = gmail_fetch.ensure_labels(ACCT, ["IRIS/Bill", "IRIS/Event"])

    assert got["IRIS/Bill"] == "Label_2"  # reused, not re-created
    assert got["IRIS/Event"] == svc.labels_api.by_name["IRIS/Event"]
    assert set(got) == {"IRIS/Bill", "IRIS/Event"}
    created = [b["name"] for b in svc.labels_api.created]
    assert created == ["IRIS", "IRIS/Event"]  # the parent first, so Gmail nests it
    assert all(
        b["labelListVisibility"] == "labelShow" and b["messageListVisibility"] == "show"
        for b in svc.labels_api.created
    )


def test_ensure_labels_caches_per_account(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _Service({"IRIS": "Label_1", "IRIS/Bill": "Label_2"})
    _use(monkeypatch, svc)
    gmail_fetch.ensure_labels(ACCT, ["IRIS/Bill"])
    gmail_fetch.ensure_labels(ACCT, ["IRIS/Bill"])
    assert svc.labels_api.list_calls == 1
    assert svc.labels_api.created == []
    gmail_fetch.forget_labels(ACCT)
    gmail_fetch.ensure_labels(ACCT, ["IRIS/Bill"])
    assert svc.labels_api.list_calls == 2


def test_ensure_labels_read_only_grant_is_a_scope_error(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _Service()

    def refuse(*, userId: str, body: dict[str, Any]) -> _Req:
        def run() -> None:
            raise _error(403, "insufficientPermissions")

        return _Req(run)

    svc.labels_api.create = refuse  # type: ignore[method-assign]
    _use(monkeypatch, svc)
    with pytest.raises(gmail_fetch.GmailScopeError, match="iris auth gmail login"):
        gmail_fetch.ensure_labels(ACCT, ["IRIS/Bill"])


# ─── modify_labels ───────────────────────────────────────────────────────


def test_modify_labels_batches_at_1000(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _Service()
    _use(monkeypatch, svc)
    ids = [f"m{i}" for i in range(2500)]

    sent = gmail_fetch.modify_labels(ACCT, ids, ["Label_2"], ["Label_3", "Label_4"])

    assert sent == 2500
    assert [len(b["ids"]) for b in svc.messages_api.bodies] == [1000, 1000, 500]
    assert all(b["addLabelIds"] == ["Label_2"] for b in svc.messages_api.bodies)
    assert all(b["removeLabelIds"] == ["Label_3", "Label_4"] for b in svc.messages_api.bodies)


def test_modify_labels_retries_a_rate_limit(monkeypatch: pytest.MonkeyPatch, _clean: Any) -> None:
    svc = _Service(errors=[_error(429), _error(403, "userRateLimitExceeded")])
    _use(monkeypatch, svc)

    assert gmail_fetch.modify_labels(ACCT, ["m1"], ["Label_2"], []) == 1
    assert len(svc.messages_api.bodies) == 3  # two refused, the third went through
    assert _clean == [1.0, 2.0]
    assert "removeLabelIds" not in svc.messages_api.bodies[-1]


def test_modify_labels_gives_up_after_the_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _Service(errors=[_error(429)] * 5)
    _use(monkeypatch, svc)
    with pytest.raises(HttpError):
        gmail_fetch.modify_labels(ACCT, ["m1"], ["Label_2"], [])
    assert len(svc.messages_api.bodies) == 5


def test_modify_labels_read_only_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _Service(errors=[_error(403, "insufficientPermissions")])
    _use(monkeypatch, svc)
    with pytest.raises(gmail_fetch.GmailScopeError, match="owner@example.com"):
        gmail_fetch.modify_labels(ACCT, ["m1"], ["Label_2"], [])
    assert len(svc.messages_api.bodies) == 1  # not retried


def test_modify_labels_invalid_label_forgets_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _Service({"IRIS/Bill": "Label_2"}, errors=[_error(400)])
    _use(monkeypatch, svc)
    gmail_fetch.ensure_labels(ACCT, ["IRIS/Bill"])
    with pytest.raises(HttpError):
        gmail_fetch.modify_labels(ACCT, ["m1"], ["Label_2"], [])
    gmail_fetch.ensure_labels(ACCT, ["IRIS/Bill"])
    assert svc.labels_api.list_calls == 2


def test_the_provider_offers_the_labelling_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_personal.email.providers import LabellingProvider

    svc = _Service({"IRIS": "Label_1", "IRIS/Bill": "Label_2"})
    _use(monkeypatch, svc)
    provider = GmailProvider()
    assert isinstance(provider, LabellingProvider)
    assert provider.ensure_labels(ACCT, ("IRIS/Bill",)) == {"IRIS/Bill": "Label_2"}
    assert provider.modify_labels(ACCT, ("m1",), ("Label_2",), ()) == 1
