"""Proposal-quality measurement for the digital-twin learning loop.

The behavior miner (L1) and intention rollup (L3) only PROPOSE; the user accepts or
rejects each one (HITL). The acceptance ratio is the cheapest honest signal of whether a
miner earns its keep: a miner whose proposals are mostly rejected is noise the user has to
wade through, while one whose proposals are mostly accepted is learning the real person.

This is a pure, deterministic aggregation over the proposal tables — no LLM, no thresholds,
no hand-tuned heuristics — so it is safe to surface read-only (CLI / API / web) and to
watch over time after the loop is enabled. Read it before flipping the gate (is the miner
proposing anything?) and after (is the user keeping what it proposes?).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


class _ProposalStore(Protocol):
    def list_behavior_proposals(self, *, status: str = ...) -> list[Any]: ...
    def list_intentions(self, *, status: str = ...) -> list[Any]: ...


@dataclass(frozen=True)
class SubsystemQuality:
    """Accept/reject tallies for one miner's proposals."""

    subsystem: str  # "behaviors" | "intentions"
    awaiting: int  # currently pending/proposed (in the review queue)
    accepted: int  # approved (behaviors) / active (intentions)
    rejected: int  # rejected (behaviors) / dismissed (intentions)

    @property
    def reviewed(self) -> int:
        return self.accepted + self.rejected

    @property
    def acceptance_rate(self) -> float:
        """Accepted / reviewed — the precision proxy. 0.0 when nothing's been reviewed."""
        return self.accepted / self.reviewed if self.reviewed else 0.0

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reviewed"] = self.reviewed
        d["acceptance_rate"] = round(self.acceptance_rate, 3)
        return d


@dataclass(frozen=True)
class ProposalQuality:
    behaviors: SubsystemQuality
    intentions: SubsystemQuality

    def as_dict(self) -> dict[str, Any]:
        return {
            "behaviors": self.behaviors.as_dict(),
            "intentions": self.intentions.as_dict(),
        }


def build_proposal_quality(store: _ProposalStore) -> ProposalQuality:
    """Tally accept/reject across both miners' proposal queues (pure, deterministic)."""
    behaviors = SubsystemQuality(
        subsystem="behaviors",
        awaiting=len(store.list_behavior_proposals(status="pending")),
        accepted=len(store.list_behavior_proposals(status="approved")),
        rejected=len(store.list_behavior_proposals(status="rejected")),
    )
    intentions = SubsystemQuality(
        subsystem="intentions",
        awaiting=len(store.list_intentions(status="proposed")),
        accepted=len(store.list_intentions(status="active")),
        rejected=len(store.list_intentions(status="dismissed")),
    )
    return ProposalQuality(behaviors=behaviors, intentions=intentions)


__all__ = ["ProposalQuality", "SubsystemQuality", "build_proposal_quality"]
