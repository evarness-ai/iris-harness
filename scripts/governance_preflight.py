#!/usr/bin/env python3
# ruff: noqa: S607 - fixed `docker info` probe, no user input on the argv
"""Governance enforce-flip preflight: report which gates are ready on this host.

Read-only. Safe to run anywhere (macOS dev box included) — it never flips a switch,
only inspects current posture + prerequisites for the four operator enforce flips:

  1. threat-detection  mode: shadow -> enforce   (needs guard models + clean battery)
  2. mcp-signing        mode: shadow -> enforce   (needs signed servers, verify clean)
  3. sandbox            runtime: docker -> gvisor (needs Linux host + runsc)
  4. grounding judge    IRIS_CURATOR_GROUNDING_LLM (needs provenance through ReAct)

See docs/usage-guides/governance-enforce-flips.md for the full runbook.

Usage:
    python scripts/governance_preflight.py          # human-readable readiness matrix
    python scripts/governance_preflight.py --json    # machine-readable (CI / dashboards)

Exit code is always 0 — this is an advisory report, not a gate.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import urllib.error
import urllib.request
from pathlib import Path
from shutil import which

REPO_ROOT = Path(__file__).resolve().parents[1]
GOV = REPO_ROOT / "config" / "governance"

OLLAMA_ENDPOINT = os.getenv("IRIS_OLLAMA_ENDPOINT", "http://localhost:11434")
GUARD_MODELS = ("llama-guard3:1b", "llama-guard3")  # 1b default; bare name = any tag
TRUTHY = {"1", "true", "yes", "on"}

# Status glyphs (ASCII only — repo bans emojis).
OK = "[ ok ]"  # ready / satisfied
NO = "[ -- ]"  # not satisfied / not applicable here
WARN = "[ !! ]"  # needs attention before flipping


def _read_yaml_scalar(path: Path, key: str) -> str | None:
    """Tiny top-level `key: value` reader — avoids a yaml dependency for a 2-field probe."""
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#") or ":" not in line:
            continue
        k, _, v = line.partition(":")
        if k.strip() == key:
            return v.split("#", 1)[0].strip().strip("\"'") or None
    return None


def _env_on(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY


def _ollama_models() -> list[str] | None:
    """Return installed Ollama model names, or None if the daemon is unreachable."""
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed http://localhost endpoint
            f"{OLLAMA_ENDPOINT}/api/tags", timeout=2
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def _docker_runtimes() -> list[str] | None:
    """Return Docker's registered runtime names, or None if docker is unavailable."""
    if which("docker") is None:
        return None
    import subprocess

    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{json .Runtimes}}"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return list(json.loads(proc.stdout).keys())
    except ValueError:
        return None


def _trust_store_key_count() -> int:
    """Count public keys in the MCP trust store (rough readiness signal)."""
    path = GOV / "mcp-trust.yaml"
    if not path.exists():
        return 0
    count = 0
    in_keys = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("keys:"):
            in_keys = "[]" not in stripped
            continue
        if in_keys:
            if raw[:1] not in (" ", "\t", "-") and stripped:
                in_keys = False
            elif stripped.startswith("- ") or stripped.startswith("key_id:"):
                count += stripped.startswith("- ")
    return count


def gather() -> dict:
    is_linux = platform.system() == "Linux"
    models = _ollama_models()
    runtimes = _docker_runtimes()
    guard_present = (
        any(m == g or m.split(":", 1)[0] == "llama-guard3" for m in models for g in GUARD_MODELS)
        if models is not None
        else None
    )
    runsc = ("runsc" in runtimes) if runtimes is not None else None

    return {
        "host": {"platform": platform.system(), "is_linux": is_linux},
        "threat": {
            "mode": _read_yaml_scalar(GOV / "threat-detection.yaml", "mode"),
            "prompt_guard_env": _env_on("IRIS_GOVERNANCE_PROMPT_GUARD"),
            "input_safety_env": _env_on("IRIS_GOVERNANCE_INPUT_SAFETY"),
            "output_safety_env": _env_on("IRIS_CURATOR_OUTPUT_SAFETY"),
            "guard_model_present": guard_present,  # None = ollama unreachable
            "ollama_reachable": models is not None,
        },
        "mcp": {
            "mode": _read_yaml_scalar(GOV / "mcp-signing.yaml", "mode"),
            "unsigned_policy": _read_yaml_scalar(GOV / "mcp-signing.yaml", "unsigned_policy"),
            "trust_keys": _trust_store_key_count(),
            "verify_cli": which("iris") is not None,
        },
        "sandbox": {
            "runtime": _read_yaml_scalar(GOV / "sandbox.yaml", "runtime"),
            "runtime_env": os.getenv("IRIS_SANDBOX_RUNTIME") or None,
            "runsc_available": runsc,  # None = docker unreachable
            "docker_reachable": runtimes is not None,
        },
        "grounding": {
            "enabled_env": _env_on("IRIS_CURATOR_GROUNDING_LLM"),
            "faithfulness_env": _env_on("IRIS_CURATOR_FAITHFULNESS_LLM"),
            "leak_env": _env_on("IRIS_CURATOR_LEAK_JUDGE"),
        },
    }


def _line(status: str, label: str, detail: str = "") -> str:
    return f"  {status} {label}" + (f" - {detail}" if detail else "")


def render(s: dict) -> int:
    out: list[str] = []
    out.append("Governance enforce-flip preflight")
    out.append(f"host: {s['host']['platform']} (linux={s['host']['is_linux']})")
    out.append("")

    t = s["threat"]
    out.append(f"1. threat-detection   (current mode: {t['mode']})")
    if t["ollama_reachable"]:
        out.append(
            _line(
                OK if t["guard_model_present"] else WARN,
                "Llama Guard 3 (ollama)",
                "present" if t["guard_model_present"] else "pull llama-guard3:1b",
            )
        )
    else:
        out.append(_line(NO, "Llama Guard 3 (ollama)", "ollama unreachable on " + OLLAMA_ENDPOINT))
    out.append(
        _line(
            OK if t["prompt_guard_env"] else NO,
            "IRIS_GOVERNANCE_PROMPT_GUARD",
            "on" if t["prompt_guard_env"] else "off (shadow-enable to test G1/G2)",
        )
    )
    out.append(
        _line(
            OK if t["input_safety_env"] else NO,
            "IRIS_GOVERNANCE_INPUT_SAFETY",
            "on" if t["input_safety_env"] else "off (shadow-enable to screen user turns)",
        )
    )
    out.append(
        _line(
            OK if t["output_safety_env"] else NO,
            "IRIS_CURATOR_OUTPUT_SAFETY",
            "on" if t["output_safety_env"] else "off (shadow-enable to test G3)",
        )
    )
    out.append(
        _line(
            NO if t["mode"] != "enforce" else OK,
            "live battery + shadow review",
            "run IRIS_THREAT_BATTERY_LIVE=1 pytest ...test_live_battery... before flipping",
        )
    )
    out.append("")

    m = s["mcp"]
    out.append(
        f"2. mcp-signing        (current mode: {m['mode']}, unsigned_policy: {m['unsigned_policy']})"
    )
    out.append(
        _line(
            OK if m["trust_keys"] else WARN,
            "trust-store keys",
            f"{m['trust_keys']} key(s)" if m["trust_keys"] else "none (iris mcp keygen)",
        )
    )
    out.append(
        _line(
            OK if m["verify_cli"] else NO,
            "iris mcp verify",
            "cli present (run it; must exit 0)" if m["verify_cli"] else "iris CLI not on PATH",
        )
    )
    out.append("")

    sb = s["sandbox"]
    eff = sb["runtime_env"] or sb["runtime"]
    out.append(f"3. sandbox            (current runtime: {eff})")
    if not s["host"]["is_linux"]:
        out.append(_line(NO, "gVisor (runsc)", "Linux-only; macOS stays on docker by design"))
    elif sb["docker_reachable"]:
        out.append(
            _line(
                OK if sb["runsc_available"] else WARN,
                "runsc docker runtime",
                "registered" if sb["runsc_available"] else "install gVisor + register runsc",
            )
        )
    else:
        out.append(_line(NO, "runsc docker runtime", "docker unreachable"))
    out.append("")

    g = s["grounding"]
    out.append("4. grounding judge")
    out.append(
        _line(
            OK if g["enabled_env"] else NO,
            "IRIS_CURATOR_GROUNDING_LLM",
            "enabled" if g["enabled_env"] else "off (set =1 to enable)",
        )
    )
    out.append("")
    out.append("Read-only report. Runbook: docs/usage-guides/governance-enforce-flips.md")
    print("\n".join(out))
    return 0


def main() -> int:
    s = gather()
    if "--json" in sys.argv[1:]:
        print(json.dumps(s, indent=2))
        return 0
    return render(s)


if __name__ == "__main__":
    raise SystemExit(main())
