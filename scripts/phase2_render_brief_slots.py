"""Render just the Phase 2 brief slot tools — no SkillRegistry, no web-fetch.

Useful when the full brief render aborts because a slot tool that's
unrelated to Phase 2 (e.g. web-fetch news) can't reach the network.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, Path(relative_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    iris_tasks = _load("iris_tasks_tools", "config/skills/builtin/iris-tasks/tools.py")
    print("== open tasks ==")
    for row in iris_tasks.ListOpenTasksTool()._run():
        print(f"  - [{row['task_id']}] {row['title']} (p{row['priority']})")

    print("\n== due today ==")
    for row in iris_tasks.ListDueTodayTool()._run():
        print(f"  - [{row['task_id']}] {row['title']} due {row['due']}")

    print("\n== resolved followups ==")
    for row in iris_tasks.ListResolvedFollowupsTool()._run():
        print(f"  - [{row['task_id']}] {row['title']} reply from {row.get('from', '')}")

    email_followup = _load("eft_tools", "config/skills/email/email-followup/tools.py")
    print("\n== open followups ==")
    for row in email_followup.ListOpenFollowupsTool()._run():
        print(f"  - [{row['task_id'][:8]}] {row['title']} awaiting {row['from']}")

    email_triage = _load("ett_tools", "config/skills/email/email-triage/tools.py")
    print("\n== inbox summary ==")
    for row in email_triage.EmailInboxSummaryTool()._run():
        print(
            f"  - {row['scope']}: {row['classified']} classified, "
            f"{row['pending']} pending, {row['unclassified']} unclassified "
            f"({row['total']} total)"
        )


if __name__ == "__main__":
    main()
