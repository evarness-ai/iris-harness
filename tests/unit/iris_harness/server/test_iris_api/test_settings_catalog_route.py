"""``GET /settings/catalog``: every setting described, and no secret's value (ADR-0120)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.runtime.plugin_host.manifest import PluginManifest
from iris_harness.server.iris_api.main import create_app


def _registry_with(manifest: PluginManifest) -> SimpleNamespace:
    return SimpleNamespace(plugins=lambda: [SimpleNamespace(manifest=manifest)])


# A process the deployment runs beside the harness, with its own settings catalog.
SIDECAR_CATALOG = """
owner: demo_sidecar
settings:
  IRIS_DEMO_SIDECAR_BUDGET:
    kind: float
    default: 5
    applies: restart
    label: Demo sidecar budget
    description: What the demo sidecar may spend.
    tab: guards
"""


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("IRIS_HOME", str(tmp_path))
    sidecar = tmp_path / "sidecar.yaml"
    sidecar.write_text(SIDECAR_CATALOG, encoding="utf-8")
    monkeypatch.setenv("IRIS_SETTINGS_SIDECAR_CATALOGS", str(sidecar))
    plugin = PluginManifest.model_validate(
        {
            "name": "demo_plugin",
            "settings": {
                "IRIS_DEMO_SWITCH": {
                    "kind": "bool",
                    "default": False,
                    "applies": "now",
                    "label": "Demo switch",
                    "description": "Turns the demo on.",
                    "tab": "agents",
                }
            },
        }
    )
    runtime = SimpleNamespace(plugin_registry=_registry_with(plugin), data_dir=tmp_path)
    app = create_app(runtime=runtime, auto_start_runtime=False)  # type: ignore[arg-type]
    with TestClient(app, headers=auth_headers()) as c:
        yield c


def _rows(client: TestClient, query: str = "") -> dict[str, dict[str, object]]:
    body = client.get(f"/settings/catalog{query}").json()
    assert body["count"] == len(body["settings"])
    return {row["name"]: row for row in body["settings"]}


def test_the_catalog_lists_core_and_plugin_settings(client: TestClient) -> None:
    rows = _rows(client)
    assert rows["IRIS_LESSON_CAPTURE_ENABLED"]["owner"] == "core"
    assert rows["IRIS_DEMO_SWITCH"]["owner"] == "plugin:demo_plugin"
    assert rows["IRIS_DEMO_SIDECAR_BUDGET"]["owner"] == "demo_sidecar"
    assert rows["IRIS_WEBUI_ALLOW_WRITES"]["guarded"] is True


def test_a_secret_says_whether_it_is_set_and_never_what_it_is(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_VAULT_MASTER_KEY", "do-not-leak-me")
    monkeypatch.setenv("IRIS_LOG_LEVEL", "DEBUG")

    body = client.get("/settings/catalog").text
    rows = _rows(client)

    assert "do-not-leak-me" not in body
    assert rows["IRIS_VAULT_MASTER_KEY"]["is_set"] is True
    assert rows["IRIS_VAULT_MASTER_KEY"]["value"] is None
    assert rows["IRIS_LOG_LEVEL"]["value"] == "DEBUG"


def test_paths_and_urls_are_not_shown_either(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_API_URL", "http://secret-host.internal:8003")
    rows = _rows(client)
    assert rows["IRIS_API_URL"]["value"] is None
    assert rows["IRIS_API_URL"]["is_set"] is True


def test_the_tab_filter(client: TestClient) -> None:
    rows = _rows(client, "?tab=guards")
    assert rows
    assert {row["tab"] for row in rows.values()} == {"guards"}
