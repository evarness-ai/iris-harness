"""Save a sandbox-generated script as a quarantined draft skill proposal.

Exposed to the chat LLM as ``propose_skill_from_sandbox``. The script is
written under ``config/skills/auto/<slug>/`` along with proposal metadata
(``proposal.yaml``), an empty run log (``runs.jsonl``), and a narrative
(``NARRATIVE.md``). Promotion to a first-class tool is gated by ``/queue``
and the coding-agent PR pipeline — this function never lands code itself.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from uuid import uuid4

from iris_harness.tools.skills.loader import (
    DEFAULT_SANDBOX_SCRIPT_NAME,
    load_skill_proposal,
    record_sandbox_run,
    resolve_auto_skills_root,
    save_skill_proposal,
    scaffold_skill_proposal,
)
from iris_harness.tools.skills.models import SkillProposal

logger = logging.getLogger(__name__)

NARRATIVE_FILE = "NARRATIVE.md"
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    """Lowercase + kebab-case a free-text intent into a stable slug."""
    cleaned = _SLUG_PATTERN.sub("-", text.lower()).strip("-")
    return cleaned or "unnamed-skill"


def propose_skill_from_sandbox(
    repo_root: Path,
    *,
    script: str,
    intent: str,
    narrative: str,
    slug: str | None = None,
) -> dict[str, object]:
    """Persist a sandbox script as a quarantined draft skill proposal.

    Returns ``{"ok": bool, ...}`` so callers can surface the result directly
    to the LLM. Existing slugs are not overwritten — repeat invocations
    return the existing proposal_dir with ``"ok": False`` and a reason.
    """
    if not script.strip():
        return {"ok": False, "error": "script is empty"}
    if not intent.strip():
        return {"ok": False, "error": "intent is required"}
    if not narrative.strip():
        return {"ok": False, "error": "narrative is required"}

    final_slug = _slugify(slug) if slug else _slugify(intent)
    auto_root_rel = "config/skills/auto"
    proposal_dir_rel = f"{auto_root_rel}/{final_slug}"
    manifest_path_rel = f"{proposal_dir_rel}/manifest.yaml"

    existing_manifest = resolve_auto_skills_root(repo_root) / final_slug / "manifest.yaml"
    updated_existing = False
    if existing_manifest.exists():
        try:
            proposal = load_skill_proposal(repo_root, final_slug)
        except (FileNotFoundError, ValueError) as exc:
            return {
                "ok": False,
                "error": f"existing slug is not a valid sandbox proposal: {exc}",
                "slug": final_slug,
                "proposal_dir": proposal_dir_rel,
            }
        if proposal.source_kind != "sandbox":
            return {
                "ok": False,
                "error": f"slug already exists with source_kind={proposal.source_kind!r}",
                "slug": final_slug,
                "proposal_dir": proposal.proposal_dir,
            }
        proposal = proposal.model_copy(
            update={
                "skill_name": intent.strip(),
                "source_description": intent.strip(),
                "sandbox_script_path": proposal.sandbox_script_path or DEFAULT_SANDBOX_SCRIPT_NAME,
            }
        )
        updated_existing = True
    else:
        proposal = SkillProposal(
            proposal_id=f"prop-{uuid4().hex[:12]}",
            task_id=f"sandbox:{final_slug}",
            skill_name=intent.strip(),
            skill_slug=final_slug,
            scope="sandbox",
            source_description=intent.strip(),
            proposal_dir=proposal_dir_rel,
            manifest_path=manifest_path_rel,
            source_kind="sandbox",
            sandbox_script_path=DEFAULT_SANDBOX_SCRIPT_NAME,
        )

    try:
        proposal_dir, _created = scaffold_skill_proposal(
            repo_root, proposal, sandbox_script=script, overwrite=updated_existing
        )
    except (OSError, ValueError) as exc:
        logger.exception("failed to scaffold sandbox skill proposal: %s", final_slug)
        return {"ok": False, "error": f"failed to scaffold proposal: {exc}"}

    (proposal_dir / NARRATIVE_FILE).write_text(narrative.strip() + "\n", encoding="utf-8")
    save_skill_proposal(repo_root, proposal)
    proposal = record_sandbox_run(repo_root, final_slug, exit_code=0)

    return {
        "ok": True,
        "proposal_id": proposal.proposal_id,
        "slug": final_slug,
        "proposal_dir": proposal_dir_rel,
        "status": proposal.status,
        "promotion_threshold": proposal.promotion_threshold,
        "run_count": proposal.run_count,
        "updated_existing": updated_existing,
    }
