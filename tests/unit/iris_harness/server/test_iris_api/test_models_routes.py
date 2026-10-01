"""The Models tab's API: tier fields and intent routing, with the route guard (ADR-0120)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.settings import SETTINGS_DB_NAME, SettingsStore
from iris_harness.kernel.governance.devices import DeviceService
from iris_harness.llm.tier_router import TierRouter
from iris_harness.server.iris_api.main import create_app

YAML = """
tiers:
  tier1: {name: Fast, provider: lmstudio, model: granite4:latest, max_tokens: 1024,
          temperature: 0.3, timeout_seconds: 90, use_for: [general]}
  tier2: {name: Advanced, provider: lmstudio, model: qwen2.5:7b-instruct, max_tokens: 4096,
          temperature: 0.7, timeout_seconds: 150, use_for: [calendar]}
  private: {name: Private, provider: ollama, model: qwen2.5:7b-instruct, max_tokens: 4096,
            temperature: 0.7, timeout_seconds: 90, use_for: [finance]}
"""


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    (tmp_path / "llm_tiers.yaml").write_text(YAML, encoding="utf-8")
    store = SettingsStore(db_path=tmp_path / SETTINGS_DB_NAME)
    router = TierRouter.load_from_yaml(tmp_path / "llm_tiers.yaml", settings=store)
    runtime = SimpleNamespace(tier_router=router, data_dir=tmp_path, plugin_registry=None)
    devices = DeviceService()
    app = create_app(runtime=runtime, auto_start_runtime=False)  # type: ignore[arg-type]
    with TestClient(app, base_url="http://iris.test") as client:
        yield SimpleNamespace(
            client=client,
            router=router,
            store=store,
            owner=_pair(devices, "control"),
            reader=_pair(devices, "read"),
        )


def _pair(devices: DeviceService, scope: str) -> dict[str, str]:
    code = devices.start_pairing(scope=scope, actor="service")
    return {
        "Authorization": "Bearer "
        + devices.claim(code=code.code, name=f"{scope} phone", kind="browser").token
    }


def _tier(body: dict, name: str) -> dict:
    return next(t for t in body["tiers"] if t["name"] == name)


def test_models_lists_tiers_with_their_file_values(world: SimpleNamespace) -> None:
    body = world.client.get("/models", headers=world.reader).json()
    assert _tier(body, "tier1")["model"] == "granite4:latest"
    assert _tier(body, "tier1")["changed"] == []
    assert body["intents"]["finance"] == "private"
    assert "ollama" in body["providers"]


def test_a_control_device_edits_a_tier_for_the_next_turn(world: SimpleNamespace) -> None:
    r = world.client.patch(
        "/models/tiers/tier1", json={"model": "qwen2.5:7b-instruct"}, headers=world.owner
    )
    assert r.status_code == 200, r.text
    tier = _tier(r.json(), "tier1")
    assert tier["model"] == "qwen2.5:7b-instruct"
    assert tier["file"]["model"] == "granite4:latest"
    assert tier["changed"] == ["model"]
    assert world.router.get_llm_config("general").model == "qwen2.5:7b-instruct"
    status = world.client.get("/system/restart", headers=world.owner).json()
    assert status["waiting_for_restart"] == ["llm_tiers:tier:tier1"]

    reset = world.client.delete("/models/tiers/tier1/override", headers=world.owner)
    assert _tier(reset.json(), "tier1")["model"] == "granite4:latest"


def test_a_route_change_needs_the_confirm(world: SimpleNamespace) -> None:
    refused = world.client.patch(
        "/models/tiers/private", json={"provider": "lmstudio"}, headers=world.owner
    )
    assert refused.status_code == 409
    assert world.router.get_tier_by_name("private").provider == "ollama"

    ok = world.client.patch(
        "/models/tiers/private",
        json={"provider": "lmstudio", "confirm": True},
        headers=world.owner,
    )
    assert ok.status_code == 200
    assert world.router.get_tier_by_name("private").provider == "lmstudio"


def test_moving_an_intent_off_its_route_needs_the_confirm(world: SimpleNamespace) -> None:
    refused = world.client.patch(
        "/models/intents/finance", json={"tier": "tier2"}, headers=world.owner
    )
    assert refused.status_code == 409
    assert world.router.intent_tier_map()["finance"] == "private"

    same_route = world.client.patch(
        "/models/intents/calendar", json={"tier": "tier1"}, headers=world.owner
    )
    assert same_route.status_code == 200
    assert same_route.json()["moved_intents"] == ["calendar"]

    back = world.client.delete("/models/intents/calendar/override", headers=world.owner)
    assert back.json()["moved_intents"] == []


@pytest.mark.parametrize(
    ("path", "body", "status"),
    [
        ("/models/tiers/tier9", {"model": "x"}, 404),
        ("/models/tiers/tier1", {"temperature": 9}, 422),
        ("/models/tiers/tier1", {}, 422),
        ("/models/tiers/tier1", {"num_ctx": 1}, 422),
        ("/models/intents/nope", {"tier": "tier1"}, 404),
    ],
)
def test_bad_requests(world: SimpleNamespace, path: str, body: dict, status: int) -> None:
    assert world.client.patch(path, json=body, headers=world.owner).status_code == status


def test_a_read_device_edits_nothing(world: SimpleNamespace) -> None:
    r = world.client.patch("/models/tiers/tier1", json={"model": "x"}, headers=world.reader)
    assert r.status_code == 403
    assert world.router.get_tier_by_name("tier1").model == "granite4:latest"
