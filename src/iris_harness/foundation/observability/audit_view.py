"""What a screen may read of an audit row's payload: a closed list of fields, never the rest.

An audit row's ``payload_json`` holds what the kernel stamped (``_AUDITED_PAYLOAD_KEYS``)
plus whatever a hook put in ``audit_metadata`` -- an exception string, a matched span, an
account. Every read surface (``GET /governance/audit``, ``iris governance audit``, the
Call-trace governance timeline) shows only the fields named here, each only when it has
the scalar type it is documented with; anything else in the payload stays in the ledger.
The keyed digests themselves are not shown: ``digest_alg`` says the call was fingerprinted
and with which key, which is what a reader can use.

It lives in ``foundation`` because the Call-trace builder (``trace_builder``) reads the
ledger without importing the kernel that writes it.
"""

from __future__ import annotations

import json
from typing import Any, Literal

#: Field -> the scalar types it may have. Who called (``caller``: a plugin's code, an
#: approved-call executor, ``mcp:<client>``), whether a model wrote the answer
#: (``deterministic`` + ``handler``, OSS plan R15), what was called, the digest key, the
#: session the row belongs to, who reads the answer (``audience``, ADR-0125), which plugin
#: owns the tool (``tool_plugin``), which call the row is about (``call_id``, a ULID; ``held_call_id``
#: on the approved re-execution of a held call, #134; ``parent_call_id`` on an egress
#: request's row: the tool call it was made inside, #103) and which model (``model``, ``provider``) a model call
#: was bound for -- names and identifiers, never what was said to them.
PUBLIC_PAYLOAD_FIELDS: dict[str, tuple[type, ...]] = {
    "caller": (str,),
    "deterministic": (bool,),
    "handler": (str,),
    "tool_name": (str,),
    "tool_plugin": (str,),
    "model": (str,),
    "provider": (str,),
    "capability": (str,),
    "method": (str,),
    "capability_provider": (str,),
    "digest_alg": (str,),
    "session_id": (str,),
    "audience": (str,),
    "call_id": (str,),
    "held_call_id": (str,),
    "parent_call_id": (str,),
    # Where the call sits (#134 stage 2): the turn the row was written in, which attempt of
    # the call it is (``attempt``, a count: the one int field), the held attempt an approved
    # re-execution replays, and the run id on the rows of a halted run that was re-entered.
    "turn_id": (str,),
    "attempt": (int,),
    "replay_of": (str,),
    "resumed_from_run": (str,),
}

#: The tier a governed call leaves the owner's machines on. The egress gate reads
#: ``tier_3`` as cloud (``kernel/governance/plugins/egress_gate.py``), and the tier a
#: model call is audited under is decided by where its provider runs (``llm/locality.py``).
CLOUD_TIER = "tier_3"

Locality = Literal["local", "cloud"]


def public_payload(payload_json: str | None) -> dict[str, Any]:
    """The documented fields of a row's payload; an unreadable payload gives ``{}``."""
    try:
        payload = json.loads(payload_json or "{}")
    except ValueError:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    for key, types in PUBLIC_PAYLOAD_FIELDS.items():
        value = payload.get(key)
        # bool is an int subclass, so a bool only passes where the table names bool itself.
        if isinstance(value, bool) and bool not in types:
            continue
        if isinstance(value, types) and value != "":
            out[key] = value
    return out


def tier_locality(tier: str | None) -> Locality | None:
    """Where a row's tier runs: ``cloud`` for the cloud tier, ``local`` for any other."""
    if not tier:
        return None
    return "cloud" if tier == CLOUD_TIER else "local"


__all__ = ["CLOUD_TIER", "PUBLIC_PAYLOAD_FIELDS", "Locality", "public_payload", "tier_locality"]
