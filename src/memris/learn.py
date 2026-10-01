"""Learning vocabulary — the policy of ADR-0115 decision 7.

A property the ontology lacks is **observed**, never written as a statement on sight:

- **alias**: its name is close enough to an existing term (``similarity`` at or above
  ``alias_similarity``) whose domain admits the subject → recorded as an alias of that
  term. From then on, what is said with it is said with that term.
- **active**: it turned up ``min_observations`` times across ``min_episodes`` distinct
  episodes → it becomes ``<prefix>:<name>`` (the prefix is the caller's), a term of its own, with the domain it was
  seen with and the range it was given. Its statements follow the caller's usual
  confirmation rules; nothing about activation confirms anything.
- **candidate**: anything else — counted, with a few examples kept, and forgotten
  (:meth:`Learner.expire`) once unseen for ``expire_days``.

A term the owner rejected is never learned again. memris calls no model: the similarity
is a callable the caller supplies, like the resolver's.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from memris.graph import MemoryGraph
from memris.model import LearnedTerm, TermKind, utc

Outcome = Literal["alias", "active", "activated", "counted", "rejected"]


@dataclass(frozen=True)
class Observation:
    term: LearnedTerm
    outcome: Outcome

    @property
    def usable_as(self) -> str | None:
        """The term a statement may be written with now — or None while it is only counted."""
        if self.outcome == "alias":
            return self.term.alias_of
        if self.outcome in ("active", "activated"):
            return self.term.name
        return None


_NAME_RE = re.compile(r"[^a-z0-9]+")


def term_local_name(raw: str) -> str:
    """``"Mentors "`` / ``"mentors-juniors"`` → ``mentors`` / ``mentors_juniors``."""
    return _NAME_RE.sub("_", (raw or "").strip().lower()).strip("_")


def _local(name: str) -> str:
    return name.partition(":")[2] or name


class Learner:
    def __init__(
        self,
        graph: MemoryGraph,
        *,
        prefix: str,
        similarity: Callable[[str, str], float] | None = None,
        names_for: Callable[[str], list[str]] | None = None,
        alias_similarity: float = 0.85,
        min_observations: int = 5,
        min_episodes: int = 3,
        expire_days: int = 90,
        max_examples: int = 3,
    ) -> None:
        self.graph = graph
        self.prefix = prefix
        self.similarity = similarity
        self.names_for = names_for
        self.alias_similarity = alias_similarity
        self.min_observations = max(1, min_observations)
        self.min_episodes = max(1, min_episodes)
        self.expire_days = expire_days
        self.max_examples = max(0, max_examples)

    def _now(self) -> datetime:
        return self.graph._now()

    def name_for(self, raw: str) -> str:
        return f"{self.prefix}:{term_local_name(raw)}"

    # -- alias ---------------------------------------------------------------------

    def _names(self, term_name: str) -> list[str]:
        declared = self.graph.ontology.relations.get(
            term_name
        ) or self.graph.ontology.attributes.get(term_name)
        names = [_local(term_name).replace("_", " ")]
        if declared is not None:
            names.append(declared.label)
        if self.names_for is not None:
            names.extend(self.names_for(term_name))
        return [n for n in names if n]

    def _alias(self, raw: str, domain: str) -> tuple[str, float] | None:
        """The declared term ``raw`` is a near-synonym of, in a domain that admits it."""
        if self.similarity is None:
            return None
        wanted = term_local_name(raw).replace("_", " ")
        onto = self.graph.declared
        best: tuple[str, float] | None = None
        for table in (onto.relations, onto.attributes):
            for name, term in table.items():
                if term.deprecated or getattr(term, "synthesized", False):
                    continue
                if not self.graph.ontology.is_subclass(domain, term.domain):
                    continue
                score = max(self.similarity(wanted, n) for n in self._names(name))
                if score >= self.alias_similarity and (best is None or score > best[1]):
                    best = (name, score)
        return best

    # -- observing -----------------------------------------------------------------

    def observe(
        self,
        raw: str,
        *,
        domain: str,
        range_: str,
        kind: TermKind = "attribute",
        example: str | None = None,
        episode: str | None = None,
    ) -> Observation | None:
        """Count one sighting of a property the ontology lacks. None for an empty name."""
        local = term_local_name(raw)
        if not local:
            return None
        name = self.name_for(local)
        domain = self.graph.ontology.qualify(domain)
        now = self._now()
        term = self.graph.store.get_term(name)
        if term is None:
            alias = self._alias(local, domain)
            if alias is not None:
                term = LearnedTerm(
                    name=name,
                    kind=kind,
                    label=local.replace("_", " "),
                    domain=domain,
                    range=range_,
                    status="alias",
                    first_seen=now,
                    last_seen=now,
                    alias_of=alias[0],
                    episodes=(episode,) if episode else (),
                    examples=(example,) if example and self.max_examples else (),
                    decided_by="similarity",
                )
                self.graph.store.save_term(term)
                return Observation(term, "alias")
            term = LearnedTerm(
                name=name,
                kind=kind,
                label=local.replace("_", " "),
                domain=domain,
                range=range_,
                status="candidate",
                first_seen=now,
                last_seen=now,
                observations=0,
            )
        seen = term.evolve(
            observations=term.observations + 1,
            last_seen=now,
            episodes=(
                term.episodes + (episode,)
                if episode and episode not in term.episodes
                else term.episodes
            ),
            examples=(
                (term.examples + (example,))[-self.max_examples :]
                if example and example not in term.examples and self.max_examples
                else term.examples
            ),
        )
        if term.status in ("rejected", "alias", "active"):
            self.graph.store.save_term(seen)
            outcome: Outcome = "rejected" if term.status == "rejected" else term.status
            return Observation(seen, outcome)
        if seen.observations >= self.min_observations and len(seen.episodes) >= self.min_episodes:
            seen = seen.evolve(status="active", activated_at=now, decided_by="evidence")
            self.graph.store.save_term(seen)
            self.graph.refresh_vocabulary()
            return Observation(seen, "activated")
        self.graph.store.save_term(seen)
        return Observation(seen, "counted")

    # -- decisions -------------------------------------------------------------------

    def reject(self, name: str, *, decided_by: str) -> LearnedTerm | None:
        """The owner's no: never learned again. An active term's statements still read."""
        term = self.graph.store.get_term(name)
        if term is None:
            return None
        rejected = term.evolve(status="rejected", decided_by=decided_by)
        self.graph.store.save_term(rejected)
        self.graph.refresh_vocabulary()
        return rejected

    def activate(self, name: str, *, decided_by: str) -> LearnedTerm | None:
        """The owner's yes, before the evidence would have got there."""
        term = self.graph.store.get_term(name)
        if term is None or term.status == "alias":
            return None
        active = term.evolve(
            status="active", activated_at=term.activated_at or self._now(), decided_by=decided_by
        )
        self.graph.store.save_term(active)
        self.graph.refresh_vocabulary()
        return active

    def expire(self) -> list[str]:
        """Forget candidates unseen for ``expire_days``. Returns their names."""
        cutoff = self._now() - timedelta(days=self.expire_days)
        gone = [t.name for t in self.graph.store.terms("candidate") if utc(t.last_seen) < cutoff]
        for name in gone:
            self.graph.store.delete_term(name)
        return gone


__all__ = ["Learner", "Observation", "Outcome", "term_local_name"]
