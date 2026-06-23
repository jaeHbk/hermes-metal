"""`hermes ingest-links <file>` — fetch and summarize every URL in a list.

Reads a plain text file of URLs (one per line; blank lines and lines starting
with `#` are ignored), fetches and extracts each via :mod:`src.web`, and runs
each through the shared :func:`src.ingest_cmd.ingest_text` write-path — so the
batch reuses exactly the same page/index/log/rollback logic as single ingest.

Failure policy is skip-and-continue: a fetch or chat error logs a warning,
skips that URL, and the loop keeps going. At the end we print a summary and
write any failed URLs to ``<input>.failed.txt`` (itself a valid links file, so
retry is just re-running on that file). New pages are auto-indexed unless
``--no-index``.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from src import web, ingest_cmd, wiki
from src.server.client import HermesError


# Match http(s) URLs embedded anywhere in free-form text (e.g. a markdown note
# body). Trailing punctuation a sentence/markdown-link wrapper leaves on the
# tail (``.,;:!?)]>"'`` and a closing paren) is trimmed in ``extract_urls``;
# the character class here is intentionally permissive so we capture the whole
# token first and clean it up afterward. This is the single URL-recognition
# regex shared by the batch links file, the single-URL CLI path, and the
# watcher's pasted-link auto-ingest.
_URL_RE = re.compile(r"https?://[^\s<>)\]}\"'`]+", re.IGNORECASE)
# Punctuation that commonly abuts a URL in prose but is not part of it.
_URL_TRAILING_TRIM = ".,;:!?\"'`)]}>"


def extract_urls(text: str) -> list[str]:
    """Return de-duplicated http(s) URLs found anywhere in ``text``.

    Order-preserving. Each URL has trailing sentence/markup punctuation
    stripped (``https://x.example/a).`` → ``https://x.example/a``) so a link
    pasted mid-sentence or inside a markdown ``[label](url)`` doesn't carry a
    stray ``)`` or ``.`` into the fetch. The scheme is validated via
    :func:`urllib.parse.urlparse` so only ``http``/``https`` survive.

    This is the shared primitive behind the links file, the single-URL CLI,
    and the watcher's auto-ingest of pasted links.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in _URL_RE.findall(text):
        url = raw.rstrip(_URL_TRAILING_TRIM)
        if not url:
            continue
        if urlparse(url).scheme not in ("http", "https"):
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def _parse_links_file(path: Path) -> list[str]:
    """Return de-duplicated http(s) URLs from ``path``, preserving order.
    Blank lines and `#` comments are skipped; non-http(s) lines are dropped."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        scheme = urlparse(line).scheme
        if scheme not in ("http", "https"):
            print(f"  skip (not http/https): {line}", file=sys.stderr)
            continue
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def _name_from_url(url: str) -> str:
    parsed = urlparse(url)
    segs = [s for s in parsed.path.split("/") if s]
    if segs:
        return segs[-1].rsplit(".", 1)[0]
    return parsed.netloc or url


def run(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="hermes ingest-links",
        description="Fetch + summarize every URL in a text file into wiki pages.",
    )
    p.add_argument("file", help="Text file of URLs (one per line; # comments ignored).")
    p.add_argument("--force", action="store_true",
                   help="Re-summarize URLs whose wiki page already exists.")
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="Cap on chat server response per page.")
    p.add_argument("--no-index", action="store_true",
                   help="Skip auto-indexing the new pages at the end.")
    args = p.parse_args(argv)

    links_path = Path(args.file).expanduser().resolve()
    if not links_path.is_file():
        print(f"hermes ingest-links: file not found: {links_path}", file=sys.stderr)
        return 2

    paths = wiki.get_paths()
    if not wiki.is_initialized(paths):
        print(f"hermes ingest-links: wiki not initialized at {paths.root}", file=sys.stderr)
        print(f"                     run: hermes wiki init", file=sys.stderr)
        return 2

    urls = _parse_links_file(links_path)
    if not urls:
        print("hermes ingest-links: no valid URLs in file.", file=sys.stderr)
        return 2

    total = len(urls)
    width = len(str(total))
    ok = skipped = 0
    failures: list[tuple[str, str]] = []
    wrote_any = False

    for i, url in enumerate(urls, 1):
        prefix = f"  [{i:>{width}}/{total}]"
        try:
            art = web.fetch_article(url)
        except web.WebError as exc:
            print(f"{prefix} FAIL {url}  ({exc})", file=sys.stderr)
            failures.append((url, str(exc)))
            continue

        page_name = art.title or _name_from_url(url)
        extra = {"source-url": url, "ingested-via": "url"}
        if art.title:
            extra["source-title"] = art.title
        if art.date:
            extra["source-date"] = art.date

        try:
            res = ingest_cmd.ingest_text(
                art.text, page_name=page_name, source_label=url,
                extra_frontmatter=extra, max_tokens=args.max_tokens, force=args.force,
            )
        except HermesError as exc:
            print(f"{prefix} FAIL {url}  (chat: {exc})", file=sys.stderr)
            failures.append((url, f"chat: {exc}"))
            continue
        except (OSError, ValueError) as exc:
            # ingest_text rolls back the page then re-raises write/index errors.
            # Treat them as per-URL failures so one bad write doesn't abort the
            # whole batch (skip-and-continue is the module's contract).
            print(f"{prefix} FAIL {url}  (write: {exc})", file=sys.stderr)
            failures.append((url, f"write: {exc}"))
            continue

        if res.status == ingest_cmd.REFUSED_HANDWRITTEN:
            print(f"{prefix} FAIL {url}  (hand-written file at {res.page_path.name})",
                  file=sys.stderr)
            failures.append((url, "hand-written target"))
        elif res.status == ingest_cmd.ALREADY_EXISTS:
            print(f"{prefix} skip {url}  (already ingested)", file=sys.stderr)
            skipped += 1
        else:  # WROTE
            print(f"{prefix} ok   {res.page_path.name}", file=sys.stderr)
            ok += 1
            wrote_any = True

    # Auto-index new pages unless suppressed.
    if wrote_any and not args.no_index:
        print(f"hermes ingest-links: indexing {ok} new page(s)...", file=sys.stderr)
        from src import index_cmd  # lazy: pulls in the LanceDB/embedding stack
        index_cmd.run(["--backfill"])

    print(f"Done: {ok} ingested, {skipped} already present, {len(failures)} failed.",
          file=sys.stderr)
    if failures:
        print("Failed (retry these):", file=sys.stderr)
        for url, reason in failures:
            print(f"  {url}  ({reason})", file=sys.stderr)
        failed_path = links_path.with_suffix(".failed.txt")
        failed_path.write_text("\n".join(u for u, _ in failures) + "\n", encoding="utf-8")
        print(f"  wrote failed URLs to {failed_path}", file=sys.stderr)
        return 1
    return 0
