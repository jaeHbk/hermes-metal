"""`hermes wiki` subcommand — init, status, schema-edit shortcuts.

Stays stdlib-only (uses src.wiki). The ingest/lint/digest features
that USE the wiki live in their own modules; this one is just the
bootstrap surface.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import wiki


def _cmd_init(args: argparse.Namespace) -> int:
    paths = wiki.get_paths()
    existed = wiki.is_initialized(paths)
    paths = wiki.init_wiki(paths, force=args.force)
    if existed and not args.force:
        print(f"hermes wiki: already initialized at {paths.root}")
        print("            (use --force to reset index/log/schema to defaults).")
        return 0
    print(f"hermes wiki: initialized at {paths.root}")
    print(f"  schema:  {paths.schema.relative_to(paths.root.parent)}")
    print(f"  index:   {paths.index.relative_to(paths.root.parent)}")
    print(f"  log:     {paths.log.relative_to(paths.root.parent)}")
    print(f"  sources/ topics/ digests/  ready")
    print()
    print("Next steps:")
    print(f"  - Edit {paths.schema.name} to describe how your vault is organized.")
    print(f"  - Ingest a source:  hermes ingest <path>")
    print(f"  - In the REPL, use  /file <name>  to file an answer as a wiki page.")
    return 0


def _cmd_status(_args: argparse.Namespace) -> int:
    paths = wiki.get_paths()
    if not wiki.is_initialized(paths):
        print(f"hermes wiki: not initialized at {paths.root}")
        print(f"  run: hermes wiki init")
        return 2
    sources = list(paths.sources_dir.glob("*.md")) if paths.sources_dir.is_dir() else []
    topics = list(paths.topics_dir.glob("*.md")) if paths.topics_dir.is_dir() else []
    digests = list(paths.digests_dir.glob("*.md")) if paths.digests_dir.is_dir() else []
    convos = list(paths.conversations_dir.glob("*.md")) if paths.conversations_dir.is_dir() else []
    print(f"hermes wiki: {paths.root}")
    print(f"  sources:        {len(sources)}")
    print(f"  topics:         {len(topics)}")
    print(f"  digests:        {len(digests)}")
    print(f"  conversations:  {len(convos)}")
    if paths.log.is_file():
        # Last log line that starts with "## ["
        log_text = paths.log.read_text(encoding="utf-8")
        last = ""
        for line in reversed(log_text.splitlines()):
            if line.startswith("## ["):
                last = line[3:]  # strip the "## "
                break
        if last:
            print(f"  last op:  {last}")
    return 0


def run(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hermes wiki",
                                description="Initialize and inspect the wiki layer.")
    sub = p.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init", help="Create wiki/ structure (idempotent).")
    init.add_argument("--force", action="store_true",
                      help="Reset index.md, log.md, .hermes-agents.md to defaults.")
    init.set_defaults(func=_cmd_init)

    status = sub.add_parser("status", help="Show page counts and last log entry.")
    status.set_defaults(func=_cmd_status)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(run())
