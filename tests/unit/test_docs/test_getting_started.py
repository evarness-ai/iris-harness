"""The getting-started pages say only what the CLI does (OSS plan R7).

``scripts/ci_quickstart.sh`` RUNS the pages' shell blocks on a fresh install
(``scripts/getting_started.py run``). Blocks it cannot run (they need your mailbox, a
browser, a model download) are annotated ``<!-- ci: skip ... -->``; this test is what
still holds them to the code. Every ``iris ...`` line on every page, run or not, must
name a real command with options and arguments that parse, so a renamed command or a
dropped flag fails here, in the plain suite, before a reader hits it.

It also pins the runner's bookkeeping: the page order matches the site's nav, every
annotation is one the runner knows, every page has a block that runs, and the install
blocks name this distribution with the email extra (R6's hero path).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import click
import pytest
import typer
import yaml

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs" / "getting-started"


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "getting_started", ROOT / "scripts" / "getting_started.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


gs = _runner()
BLOCKS = gs.all_blocks(DOCS)


def _iris_lines() -> list[tuple[str, list[str]]]:
    return [(b.where, argv) for b in BLOCKS for argv in gs.iris_commands(b.body)]


def test_the_page_order_is_the_sites_nav_order() -> None:
    nav = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))["nav"]
    [section] = [entry["Getting started"] for entry in nav if "Getting started" in entry]
    listed = [Path(next(iter(item.values()))).name for item in section]
    assert tuple(listed) == gs.PAGE_ORDER


def test_every_block_has_a_known_annotation_and_every_page_runs_one() -> None:
    by_page: dict[str, set[str]] = {}
    for block in BLOCKS:
        kind, _allowed, _reason = block.action()  # raises on an unknown annotation
        by_page.setdefault(block.page, set()).add(kind)
    assert set(by_page) == set(gs.PAGE_ORDER)
    assert all("run" in kinds for kinds in by_page.values()), by_page


def test_the_annotation_parser_rejects_what_it_does_not_know() -> None:
    page = "<!-- ci: maybe -->\n```bash\niris doctor\n```\n"
    [block] = gs.parse_blocks(page, "x.md")
    with pytest.raises(ValueError, match="unknown annotation"):
        block.action()
    page = "<!-- ci: exit 0,1 -->\n\n```bash\niris doctor\n```\n```text\nnot shell\n```\n"
    [block] = gs.parse_blocks(page, "x.md")
    assert block.action() == ("run", (0, 1), "")


def test_the_install_blocks_name_this_distribution_with_the_email_extra() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    name = re.search(r'^name = "([^"]+)"', pyproject, re.MULTILINE)
    assert name is not None
    # The public tree declares the extra; this tree carries it as an export line.
    assert re.search(r"^(# oss-export: public-line )?email = \[", pyproject, re.MULTILINE)
    installs = [b for b in BLOCKS if b.action()[0] == "install"]
    assert installs, "the install page shows the install command"
    for block in installs:
        assert f'install "{name.group(1)}[email]"' in block.body, block.where


def _resolve(root: click.Group, argv: list[str]) -> tuple[click.Command, click.Context, list[str]]:
    ctx = click.Context(root, info_name="iris")
    cmd: Any = root
    rest = list(argv)
    while isinstance(cmd, click.Group) and rest and not rest[0].startswith("-"):
        sub = cmd.get_command(ctx, rest[0])
        if sub is None:
            raise AssertionError(f"no command `{rest[0]}` under `{ctx.command_path}`")
        ctx = click.Context(sub, info_name=rest.pop(0), parent=ctx)
        cmd = sub
    return cmd, ctx, rest


@pytest.mark.parametrize(("where", "argv"), _iris_lines(), ids=lambda v: str(v))
def test_every_iris_command_on_the_pages_parses(where: str, argv: list[str]) -> None:
    from iris_harness.main import app

    root = typer.main.get_command(app)
    assert isinstance(root, click.Group)
    cmd, ctx, rest = _resolve(root, argv)
    if isinstance(cmd, click.Group):  # a bare `iris` or a group: its own options only
        cmd.parse_args(click.Context(cmd, info_name=ctx.info_name, parent=ctx.parent), rest)
        return
    try:
        cmd.make_context(ctx.info_name, rest, parent=ctx.parent)
    except click.ClickException as exc:
        message = f"{where}: `iris {' '.join(argv)}`: {exc.format_message()}"
        raise AssertionError(message) from exc


def test_the_pages_use_the_stable_commands() -> None:
    """The hero path (R6) is the stable quickstart CLI, and the pages show all of it."""
    from iris_harness.testing import stable_tier

    shown = {tuple(argv) for _where, argv in _iris_lines()}
    for command in stable_tier().quickstart_cli:
        assert any(line[: len(command)] == tuple(command) for line in shown), command
