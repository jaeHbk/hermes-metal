"""`hermes lint` — wiki health-check.

Read-only. Reports without auto-fixing because every "fix" here is a
judgment call:

* **orphans** — pages no other page links to. Sometimes correct (a
  source might be standalone), sometimes a sign the user forgot to
  follow up on it.
* **stubs** — pages referenced by ``[[Foo]]`` links but with no
  matching file. Either typo or a planned page that hasn't been
  written yet.
* **stale** — pages whose ``hermes-updated`` frontmatter is older
  than ``--stale-days`` (default 30).
* **unused sources** — pages in ``sources/`` not linked from any
  ``topics/`` or ``digests/`` page. Usually means the source was
  ingested but never woven into a synthesis.

Exit code is 0 unless ``--strict`` is passed; ``--strict`` returns 1
if any issue is found, suitable for CI.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import wiki


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_UPDATED_RE = re.compile(r'^hermes-updated:\s*"?([^"\n]+)"?\s*$', re.MULTILINE)


def _read_updated(path: Path) -> datetime | None:
    """Parse ``hermes-updated`` from frontmatter; None if missing/unparseable.

    Uses ``datetime.fromisoformat`` (Python 3.11+) which handles fractional
    seconds and `Z` suffix natively. Falls back to ``%Y-%m-%d`` for the
    date-only form. Without this, an ISO timestamp like
    ``2026-06-04T19:00:00.123Z`` would silently be skipped from stale
    detection, hiding stale pages forever.
    """
    try:
        head = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    fm = _FRONTMATTER_RE.match(head)
    if not fm:
        return None
    m = _UPDATED_RE.search(fm.group(1))
    if not m:
        return None
    raw = m.group(1).strip()
    # Python 3.11's fromisoformat accepts the trailing 'Z' (UTC) and
    # fractional seconds. Try it first.
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00") if raw.endswith("Z") else raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass
    # Date-only fallback.
    try:
        return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _index_outbound_links(paths: wiki.WikiPaths) -> dict[Path, set[str]]:
    """Map every wiki page (by full path) → set of stems it links to.

    Keyed on the full ``Path`` (not stem) so two pages with the same
    stem in different subdirs (e.g. ``topics/foo.md`` and
    ``sources/foo.md``) are kept distinct. The returned values are
    target stems — link resolution remains stem-based, which is fine
    because that ambiguity is symmetric: an incoming `[[foo]]` matches
    any page whose stem is ``foo``.
    """
    out: dict[Path, set[str]] = {}
    for p in wiki.all_pages(paths):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        targets = wiki.parse_links(text)
        # Strip self-loops: a page linking to its own stem isn't a
        # real inbound for orphan-detection purposes — without this,
        # `foo.md` containing `[[foo]]` masks itself from orphan flags.
        targets.discard(wiki.page_stem(p))
        out[p] = targets
    return out


def _inbound_count(out_links: dict[Path, set[str]], target_stem: str) -> int:
    """How many pages link to ``target_stem``."""
    return sum(1 for targets in out_links.values() if target_stem in targets)


# ------------------------------------------------------------------- run


def run(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="hermes lint",
        description="Wiki health-check: orphans, stubs, stale, unused sources.",
    )
    p.add_argument("--stale-days", type=int, default=30,
                   help="Pages not updated in this many days are reported as stale (default 30).")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 if any issue is found (suitable for CI).")
    args = p.parse_args(argv)

    paths = wiki.get_paths()
    if not wiki.is_initialized(paths):
        print(f"hermes lint: wiki not initialized at {paths.root}", file=sys.stderr)
        print(f"             run: hermes wiki init", file=sys.stderr)
        return 2

    pages = wiki.all_pages(paths)
    if not pages:
        print(f"hermes lint: wiki is empty (no pages under sources/topics/digests).")
        return 0

    out_links = _index_outbound_links(paths)
    all_stems = {wiki.page_stem(p) for p in pages}

    # --- orphans: TOPIC pages with no inbound link.
    # Sources without inbound links go in "unused sources" instead.
    # Digests are chronologically terminal — they're not expected to be
    # linked from elsewhere (a daily digest doesn't need a backref to
    # be useful), so they're exempt from the orphan check.
    # Iterate on full paths (not stems) so two pages with the same stem
    # in different subdirs are both evaluated.
    orphans: list[Path] = []
    unused_sources: list[Path] = []
    for path in sorted(pages):
        stem = wiki.page_stem(path)
        inbound = _inbound_count(out_links, stem)
        if path.parent == paths.sources_dir:
            if inbound == 0:
                unused_sources.append(path)
        elif path.parent == paths.topics_dir:
            if inbound == 0:
                orphans.append(path)
        # digests/ — never flagged as orphan

    # --- stubs: link targets that don't have a matching page.
    referenced = set().union(*out_links.values()) if out_links else set()
    stubs = sorted(referenced - all_stems)

    # --- stale: hermes-updated older than threshold.
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.stale_days)
    stale: list[tuple[Path, datetime]] = []
    for path in sorted(pages):
        updated = _read_updated(path)
        if updated and updated < cutoff:
            stale.append((path, updated))

    # ----- render
    issue_count = 0

    def _section(title: str, items: list[str]) -> None:
        nonlocal issue_count
        if not items:
            return
        issue_count += len(items)
        print(f"\n## {title} ({len(items)})")
        for line in items:
            print(f"  - {line}")

    print(f"hermes lint: {paths.root}")
    print(f"  pages: {len(pages)}  links: {sum(len(v) for v in out_links.values())}")

    _section(
        "Orphan topics/digests (no inbound links)",
        [str(p.relative_to(paths.root)) for p in orphans],
    )
    _section(
        "Unused sources (in sources/ but never cited)",
        [str(p.relative_to(paths.root)) for p in unused_sources],
    )
    _section(
        f"Stubs (referenced but no page exists)",
        stubs,
    )
    _section(
        f"Stale (not updated in {args.stale_days} days)",
        [
            f"{p.relative_to(paths.root)}  (last update {ts.date()})"
            for p, ts in stale
        ],
    )

    if issue_count == 0:
        print("\nclean — no issues.")
    else:
        print(f"\nfound {issue_count} issue(s).")

    if args.strict and issue_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
