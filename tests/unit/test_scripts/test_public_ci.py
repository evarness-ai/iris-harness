"""The public repository's GitHub workflows (OSS plan R11), pinned.

They live in ``.github/public/`` in the private repository (GitHub reads only fixed paths,
so they are inert there) and become ``.github/`` in the export. This test reads whichever
the tree has, so it ships and holds in the public repository too.

What it pins: every R11 check exists under its required name; no workflow runs on
``pull_request_target``; every action is pinned to a full commit SHA; the token is
read-only unless a job asks for more; checkout never persists credentials; no ``run:``
script interpolates event data (a PR title is attacker-controlled); the test matrix is
the supported Python range; PyPI publishing is trusted publishing, never a token; the
docs deploy writes Pages from ``main`` only, and only from its deploy job; the web console
lints, type-checks, builds and passes its Playwright specs on every PR with release.yml's
Node, installing exactly the browsers those specs use; Dependabot holds the
web console to minor and patch updates and moves React with its types.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import tomllib
import yaml

ROOT = Path(__file__).resolve().parents[3]
_PRIVATE_LAYOUT = ROOT / ".github" / "public"
GITHUB = _PRIVATE_LAYOUT if _PRIVATE_LAYOUT.is_dir() else ROOT / ".github"
WORKFLOWS = GITHUB / "workflows"
FILES = sorted(WORKFLOWS.glob("*.yml"))

# The required pull-request checks of R11, as GitHub names them (job `name:`, matrix
# expanded). Branch protection lists exactly these.
REQUIRED_CHECKS = {
    "lint",
    "test (3.12)",
    "test (3.13)",
    "playground",
    "quickstart",
    "security",
    "codeql (python)",
    "codeql (javascript-typescript)",
    "docs",
    "dco",
    "licenses",
    "pr-title",
    "webui",
}
_SHA_PIN = re.compile(r"^[\w.-]+/[\w.-]+(/[\w./-]+)?@[0-9a-f]{40}$")


def _load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), path
    return data


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    # YAML 1.1 reads a bare `on:` key as the boolean True.
    on = workflow.get("on", workflow.get(True))
    if isinstance(on, str):
        return {on: None}
    if isinstance(on, list):
        return dict.fromkeys(on)
    assert isinstance(on, dict)
    return on


def _steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for job in workflow["jobs"].values() for step in job.get("steps", [])]


def _expand(name: str, job: dict[str, Any]) -> list[str]:
    matrix = job.get("strategy", {}).get("matrix", {})
    names = [name]
    for key, values in matrix.items():
        token = "${{ matrix." + key + " }}"
        if token in name:
            names = [n.replace(token, str(v)) for n in names for v in values]
    return names


def test_the_public_workflows_exist() -> None:
    assert {p.name for p in FILES} >= {
        "ci.yml",
        "security.yml",
        "pr.yml",
        "release.yml",
        "release-drafter.yml",
        "docs-deploy.yml",
    }
    assert (GITHUB / "dependabot.yml").is_file()
    assert (GITHUB / "release-drafter.yml").is_file()


def test_every_required_check_is_a_job() -> None:
    names: set[str] = set()
    for path in FILES:
        for job_id, job in _load(path)["jobs"].items():
            names.update(_expand(job.get("name", job_id), job))
    missing = REQUIRED_CHECKS - names
    assert not missing, f"R11 checks with no job: {sorted(missing)}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_never_pull_request_target(path: Path) -> None:
    triggers = _triggers(_load(path))
    assert "pull_request_target" not in triggers
    assert "workflow_run" not in triggers  # the same privilege, one hop removed


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_commit_sha(path: Path) -> None:
    for step in _steps(_load(path)):
        uses = step.get("uses")
        if uses is None or uses.startswith("./"):
            continue
        assert _SHA_PIN.match(uses), f"{path.name}: {uses!r} is not pinned to a full SHA"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_the_token_is_read_only_unless_a_job_asks(path: Path) -> None:
    workflow = _load(path)
    top = workflow.get("permissions")
    assert top is not None, f"{path.name}: no top-level permissions"
    assert top in ({}, {"contents": "read"}), f"{path.name}: top-level permissions {top}"
    for job_id, job in workflow["jobs"].items():
        for scope, level in (job.get("permissions") or {}).items():
            assert level in {"read", "write", "none"}, (job_id, scope, level)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_checkout_never_persists_credentials(path: Path) -> None:
    for step in _steps(_load(path)):
        if str(step.get("uses", "")).startswith("actions/checkout@"):
            assert step.get("with", {}).get("persist-credentials") is False, path.name


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_run_script_interpolates_event_data(path: Path) -> None:
    # `${{ github.event.* }}` / `${{ github.head_ref }}` inside a `run:` is a script
    # injection: the value goes through `env:` and is read as a shell variable instead.
    for step in _steps(_load(path)):
        script = step.get("run", "")
        assert not re.search(r"\$\{\{\s*github\.(event|head_ref)", script), (
            path.name,
            step.get("name"),
        )


def test_the_test_matrix_is_the_supported_python_range() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["tool"]["poetry"]["dependencies"]["python"] == ">=3.12,<3.14"
    test = _load(WORKFLOWS / "ci.yml")["jobs"]["test"]
    assert test["strategy"]["matrix"]["python-version"] == ["3.12", "3.13"]


def test_pypi_publishing_is_trusted_publishing() -> None:
    jobs = _load(WORKFLOWS / "release.yml")["jobs"]
    publish = jobs["pypi"]
    assert publish["permissions"] == {"id-token": "write"}
    assert publish["environment"]["name"] == "pypi"
    (step,) = [s for s in publish["steps"] if "pypi-publish" in str(s.get("uses", ""))]
    assert "password" not in (step.get("with") or {}), "trusted publishing needs no token"
    assert set(_triggers(_load(WORKFLOWS / "release.yml"))) == {"push"}


def test_dependabot_covers_python_actions_and_the_web_console() -> None:
    config = _load(GITHUB / "dependabot.yml")
    ecosystems = {(u["package-ecosystem"], u["directory"]) for u in config["updates"]}
    assert ecosystems == {("pip", "/"), ("github-actions", "/"), ("npm", "/webui")}


def test_dependabot_moves_react_with_its_types_and_holds_majors_until_launch() -> None:
    updates = _load(GITHUB / "dependabot.yml")["updates"]
    (npm,) = [u for u in updates if u["package-ecosystem"] == "npm"]
    # A dependency lands in the FIRST group it matches, so react must precede the catch-all.
    assert list(npm["groups"])[0] == "react"
    assert set(npm["groups"]["react"]["patterns"]) == {
        "react",
        "react-dom",
        "@types/react",
        "@types/react-dom",
    }
    assert npm["groups"]["webui-minor-patch"]["update-types"] == ["minor", "patch"]
    # Until the release-1 launch: no semver-major web console update at all.
    held = {"dependency-name": "*", "update-types": ["version-update:semver-major"]}
    assert held in npm["ignore"]
    # The Python dependencies hold majors until launch the same way.
    pip = next(u for u in updates if u["package-ecosystem"] == "pip")
    assert held in pip["ignore"]


def test_the_webui_job_lints_typechecks_and_builds_on_every_pr() -> None:
    workflow = _load(WORKFLOWS / "ci.yml")
    # A `paths:` filter would leave the required `webui` check pending on a PR that does
    # not touch webui/, so the workflow carries none and the job has no `if`.
    triggers = _triggers(workflow)
    for event in ("pull_request", "push"):
        assert not {"paths", "paths-ignore"} & set(triggers[event] or {})
    job = workflow["jobs"]["webui"]
    assert "if" not in job
    assert job["defaults"]["run"]["working-directory"] == "webui"
    (node,) = [s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/setup-node@")]
    (release_node,) = [
        s
        for s in _steps(_load(WORKFLOWS / "release.yml"))
        if str(s.get("uses", "")).startswith("actions/setup-node@")
    ]
    assert node["uses"] == release_node["uses"]
    assert node["with"]["node-version"] == release_node["with"]["node-version"] == "22"
    assert [s["run"] for s in job["steps"] if "run" in s] == [
        "npm ci --no-audit --no-fund",
        "npm run lint",
        "npx --no-install tsc --noEmit",
        "npm run build",
        # The Playwright specs (viewport, nav gating, Setup, Governance, Call trace), after
        # the build. The browsers come with the job's Playwright image: nothing is
        # apt-installed from the runner's mirror.
        "npx --no-install playwright test",
    ]


def test_the_webui_job_installs_exactly_the_browsers_the_specs_use() -> None:
    config = (ROOT / "webui" / "playwright.config.ts").read_text(encoding="utf-8")
    # Each project's device fixes its engine: iPhone 13 is WebKit, Pixel 7 Chromium.
    devices = set(re.findall(r'devices\["([^"]+)"\]', config))
    assert devices == {"iPhone 13", "Pixel 7"}, "a new project may need another browser"
    job = _load(WORKFLOWS / "ci.yml")["jobs"]["webui"]
    # The image ships WebKit and Chromium (and Firefox); nothing installs browsers.
    assert not [s for s in job["steps"] if "playwright install" in s.get("run", "")]
    image = job["container"]["image"]
    assert image.startswith("mcr.microsoft.com/playwright:v")
    tag, _, digest = image.partition("@")
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest), "pin the image by digest"
    # The image's browsers are built for one Playwright release: the lockfile's.
    lock = json.loads((ROOT / "webui" / "package-lock.json").read_text(encoding="utf-8"))
    version = lock["packages"]["node_modules/@playwright/test"]["version"]
    assert tag.split(":")[1] == f"v{version}-noble"


def test_the_docs_deploy_writes_pages_from_main_only() -> None:
    workflow = _load(WORKFLOWS / "docs-deploy.yml")
    triggers = _triggers(workflow)
    assert set(triggers) == {"push", "workflow_dispatch"}
    assert triggers["push"] == {"branches": ["main"]}
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"] == {"group": "pages", "cancel-in-progress": False}
    jobs = workflow["jobs"]
    # Only the deploy job may write Pages or mint the OIDC token the deploy needs.
    deploy = jobs["deploy"]
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}
    assert deploy["environment"]["name"] == "github-pages"
    assert "steps.deployment.outputs.page_url" in deploy["environment"]["url"]
    for job_id, job in jobs.items():
        if job_id != "deploy":
            assert "permissions" not in job, f"{job_id} widens the token"
    # A private repository has no free Pages: the build (and so the deploy) skips there.
    assert jobs["build"]["if"] == "${{ !github.event.repository.private }}"
    assert jobs["deploy"]["needs"] == "build"
    script = "\n".join(step.get("run", "") for step in jobs["build"]["steps"])
    assert "-r docs/requirements.txt" in script
    assert "mkdocs" in script and "build --strict" in script


def test_codeql_runs_only_on_a_public_repository_and_the_secret_scan_always() -> None:
    jobs = _load(WORKFLOWS / "security.yml")["jobs"]
    # Code scanning on a private repository needs paid Advanced Security: CodeQL skips there.
    assert jobs["codeql"]["if"] == "${{ !github.event.repository.private }}"
    # gitleaks and the identity scan are free and guard every push: never conditional.
    assert "if" not in jobs["security"]
    script = "\n".join(step.get("run", "") for step in jobs["security"]["steps"])
    assert "oss_pii_scan.sh --strict" in script and "gitleaks git" in script
