"""Code-first skills subsystem for IRIS."""

from .loader import (
    discover_skill_dirs,
    load_skill_manifest,
    load_skill_package,
    load_skill_proposal,
    record_sandbox_run,
    resolve_auto_skills_root,
    resolve_skills_root,
    save_skill_proposal,
    scaffold_skill_proposal,
)
from .models import (
    SkillManifest,
    SkillPackage,
    SkillProposal,
    SkillRequirements,
    SkillToolManifest,
    ToolArg,
)
from .registry import SkillRegistry

__all__ = [
    "SkillManifest",
    "SkillPackage",
    "SkillProposal",
    "SkillRegistry",
    "SkillRequirements",
    "SkillToolManifest",
    "ToolArg",
    "discover_skill_dirs",
    "load_skill_manifest",
    "load_skill_package",
    "load_skill_proposal",
    "record_sandbox_run",
    "resolve_auto_skills_root",
    "resolve_skills_root",
    "save_skill_proposal",
    "scaffold_skill_proposal",
]
