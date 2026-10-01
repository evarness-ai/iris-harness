"""Provider-agnostic email contract.

Returned by ``gmail-inbox`` (Track 1C) and any future ``*-inbox``
skills (outlook-inbox, imap-inbox, icloud-inbox). Consumed by
``email-triage`` and downstream classifier / followup skills.

Body fields (``body_text``, ``body_html``) are populated during the
fetch and used immediately; they are **not** persisted to
``data/email.db`` per the canonical doc §3.1 — only the snippet
survives. See ``iris_personal.email.store.EmailStore``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# The provider set is open: a message's ``provider`` is the ``name`` of the mail provider
# that fetched it (``email.providers`` registry, keyed the same way as
# ``email_accounts.provider``), so a third-party provider plugin -- written against
# ``email.provider_api`` -- records its own mail without a change here. A closed Literal
# used to list the in-tree ones (gmail, imap, demo, ...), which no plugin could extend.
EmailProvider = str


class EmailAttachment(BaseModel):
    """Descriptor for one attached file on an email.

    Carries metadata only — never the bytes. Downloaders fetch via
    ``attachment_id`` against the provider's API when needed.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    filename: str = Field(..., min_length=1)
    mime_type: str = Field(..., min_length=1)
    size_bytes: int = Field(..., ge=0)
    attachment_id: str = Field(..., min_length=1)


class EmailMessage(BaseModel):
    """Provider-agnostic email envelope.

    Identity invariant: ``(provider, id)`` is globally unique. The
    ``id`` is the provider-native message identifier (Gmail message
    id, IMAP UID, etc.) — different providers may reuse string-shaped
    ids, hence the pair-level uniqueness.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    # ─── Identity ────────────────────────────────────────────────────
    id: str = Field(..., min_length=1)
    provider: EmailProvider = Field(..., min_length=1)
    account_id: str = Field(..., min_length=1)  # soft FK to email_accounts.id
    thread_id: str | None = Field(default=None, min_length=1)

    # ─── Envelope ────────────────────────────────────────────────────
    from_address: str = Field(..., min_length=3)
    from_domain: str | None = Field(default=None, min_length=1)
    to: tuple[str, ...] = Field(default_factory=tuple)
    cc: tuple[str, ...] = Field(default_factory=tuple)
    subject: str = ""
    received_at: datetime

    # ─── Content ─────────────────────────────────────────────────────
    snippet: str = ""  # first ~400 chars; persisted
    body_text: str | None = None  # transit only — NOT persisted
    body_html: str | None = None  # transit only — NOT persisted

    # ─── Provider metadata ───────────────────────────────────────────
    labels: tuple[str, ...] = Field(default_factory=tuple)
    attachments: tuple[EmailAttachment, ...] = Field(default_factory=tuple)
    headers_subset: dict[str, str] = Field(default_factory=dict)

    # ─── IRIS enrichment ─────────────────────────────────────────────
    # The triage/classifier verdict (ADR-0017 path: email/root/branch/leaf), or
    # the provider's own bucket until triage runs (see ``classified_source``).
    # Populated on read from data/email.db when the message has been
    # classified; None on a freshly-fetched envelope before triage. Not a
    # provider field — IRIS-internal — but travels with the envelope so
    # downstream skills (followup detection) can gate on it.
    classified_category: str | None = Field(default=None, min_length=1)
    # Who wrote ``classified_category``: ``"iris"`` (triage, a user correction) or
    # ``"vendor"`` (the mailbox's own bucket, e.g. a Gmail tab, mapped to a reserved
    # path by the provider plugin). None while unclassified. Readers that learn from
    # or act on a classification (wiki backfill, finance, followups) trust only
    # ``"iris"``; triage still classifies ``"vendor"`` rows and overwrites them.
    classified_source: Literal["iris", "vendor"] | None = None

    # The provider's own category for this message, already mapped to an IRIS topic
    # path by the provider plugin (the gmail plugin maps CATEGORY_PROMOTIONS to
    # ``email/promotions`` from its YAML table). Set on a freshly-fetched envelope;
    # ``EmailStore.upsert`` stores it as a ``"vendor"`` classification unless IRIS has
    # already classified the row. None when the provider has no bucket for it.
    vendor_category: str | None = Field(default=None, min_length=1)


class CategoryRepresentative(BaseModel):
    """One representative email surfaced from a discovered cluster.

    Lightweight projection of ``EmailMessage`` carrying just the fields
    needed for human review and LLM naming. Persisted to the proposals
    JSONL.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: str = Field(..., min_length=1)
    subject: str = ""
    from_address: str = Field(..., min_length=1)
    snippet: str = ""


class CategoryProposal(BaseModel):
    """One discovered cluster proposed as a category candidate.

    Produced by ``iris_personal.plugins.email_workflows.discovery.bootstrap_categories`` per
    ADR-0017 (dynamic discovery) + ADR-0018 (CLI shape). A proposal is
    a draft — Track 1F's ``iris email accept-categories`` is what
    converts accepted proposals into rows in ``data/iris.db.categories``.

    The ``cohesion`` field (mean intra-cluster centroid cosine
    similarity) is the load-bearing confidence signal per ADR-0018 §3
    — LLM self-reported confidence was inflated in the spike and is
    not stored here.

    The ``member_ids`` field lists every email belonging to the
    cluster, not just the representatives, so Track 1G can seed kNN
    from the full membership.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    # ─── Identity ────────────────────────────────────────────────────
    cluster_id: int = Field(..., ge=0)
    size: int = Field(..., ge=1)

    # ─── Composition ─────────────────────────────────────────────────
    top_domains: tuple[tuple[str, int], ...] = Field(default_factory=tuple)
    representatives: tuple[CategoryRepresentative, ...] = Field(default_factory=tuple)
    member_ids: tuple[str, ...] = Field(default_factory=tuple)

    # ─── Quality signals ─────────────────────────────────────────────
    cohesion: float = Field(..., ge=0.0, le=1.0)  # mean intra-cluster centroid cos sim

    # ─── Naming ──────────────────────────────────────────────────────
    proposed_root: str | None = None
    proposed_branch: str | None = None
    proposed_leaf: str | None = None
    naming_rationale: str | None = None
