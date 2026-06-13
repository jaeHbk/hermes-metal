# URL Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user hand hermes a list of web links and get per-source wiki pages, via `hermes ingest --url <url>` and `hermes ingest-links <file>`.

**Architecture:** Three isolated units. `src/web.py` turns a URL into clean article text (httpx + trafilatura). `src/ingest_cmd.py` is refactored to expose a shared `ingest_text()` core that both file and URL ingestion call (one tested write/rollback path). `src/ingest_links_cmd.py` is a skip-and-continue batch driver over a links file. CLI wiring adds `--url` to `ingest` and a new `ingest-links` subcommand; new pages are auto-indexed by reusing `index_cmd.run(["--backfill"])`.

**Tech Stack:** Python 3.14, httpx (existing dep), trafilatura (new dep), pytest with `httpx.MockTransport` and monkeypatch (existing test patterns).

**Spec:** `docs/superpowers/specs/2026-06-12-url-ingestion-design.md`

**Branch:** `feat/phases-b-e` (already checked out).

---

## File Structure

| File | Responsibility | New/Modify |
|---|---|---|
| `requirements.txt` | add `trafilatura` dep | Modify |
| `src/web.py` | `fetch_article(url) -> Article`; `WebError`; `Article` dataclass | Create |
| `src/ingest_cmd.py` | extract `ingest_text()` core + `IngestResult`; `run()` calls it; add `--url` | Modify |
| `src/ingest_links_cmd.py` | batch driver: parse file, loop, skip-and-continue, auto-index, `.failed.txt` | Create |
| `src/cli.py` | `--url` on `ingest` subparser; new `ingest-links` subparser + dispatch | Modify |
| `src/doctor.py` | trafilatura import probe in a new `check_web()` section | Modify |
| `tests/test_web.py` | `fetch_article` in isolation (mocked httpx) | Create |
| `tests/test_ingest_links_cmd.py` | batch driver (mocked fetch + chat + index) | Create |
| `tests/test_ingest.py` | extend: `--url` path + `ingest_text` direct; existing tests must still pass | Modify |
| `tests/test_doctor_plist.py` | extend: trafilatura probe | Modify |

**Key interfaces locked here (used across tasks):**

```python
# src/web.py
@dataclass
class Article:
    url: str
    title: str        # "" if none found
    text: str         # clean markdown body
    author: str = ""  # "" if none found
    date: str = ""    # ISO date "" if none found

class WebError(Exception): ...

def fetch_article(url: str, *, timeout: float = 20.0,
                  _client: "httpx.Client | None" = None) -> Article: ...
```

```python
# src/ingest_cmd.py
WROTE = "WROTE"
ALREADY_EXISTS = "ALREADY_EXISTS"
REFUSED_HANDWRITTEN = "REFUSED_HANDWRITTEN"

@dataclass
class IngestResult:
    status: str            # one of the three above
    page_path: Path | None
    summary: str

def ingest_text(body: str, *, page_name: str, source_label: str,
                extra_frontmatter: dict[str, str] | None = None,
                max_tokens: int = 1024, force: bool = False) -> IngestResult: ...
```

`source_label` is what the LLM sees ("Source path:" line in `_build_user_prompt`). For files it's the path; for URLs it's the URL.

---

## Task 1: Add trafilatura dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the dependency line**

Edit `requirements.txt`, after the `PyYAML>=6.0.2` line (before `numpy`), add:

```
# Web article extraction for `hermes ingest --url` / `ingest-links`. Pure-Python
# with arm64 wheels; strips boilerplate (nav/ads) and emits clean markdown.
trafilatura>=1.12.0
```

- [ ] **Step 2: Install it into the venv**

Run: `.venv/bin/pip install 'trafilatura>=1.12.0'`
Expected: ends with `Successfully installed trafilatura-...` (plus its deps: lxml, charset-normalizer, etc.)

- [ ] **Step 3: Verify import**

Run: `.venv/bin/python -c "import trafilatura; print(trafilatura.__version__)"`
Expected: a version string like `1.12.2` (no traceback)

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "deps: add trafilatura for web article extraction"
```

---

## Task 2: `src/web.py` — fetch + extract (test first)

**Files:**
- Create: `tests/test_web.py`
- Create: `src/web.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web.py`:

```python
"""Tests for src/web.py — URL fetch + article extraction.

Uses httpx.MockTransport so the suite needs no network (same pattern as
tests/test_streaming.py). trafilatura runs for real against canned HTML —
it is deterministic on fixed input.
"""
from __future__ import annotations

import httpx
import pytest

from src import web


_ARTICLE_HTML = """
<!DOCTYPE html>
<html><head><title>Attention Explained</title>
<meta property="article:published_time" content="2024-11-02"></head>
<body>
<nav>home about contact</nav>
<article>
<h1>Attention Explained</h1>
<p>The attention mechanism lets a model weigh the relevance of each input
token when producing each output token. This paragraph is intentionally long
enough that trafilatura treats it as the main article body rather than
boilerplate, which requires a few sentences of real prose to trip its
content-density heuristic reliably across versions.</p>
<p>A second substantial paragraph reinforces that the article body is the
dominant text block on the page, well clear of the 200-character floor that
fetch_article enforces before it accepts an extraction as real content.</p>
</article>
<footer>copyright 2024</footer>
</body></html>
"""


def _client(responder) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(responder))


def test_fetch_article_extracts_body_and_title():
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=_ARTICLE_HTML)

    art = web.fetch_article("https://blog.example/attn", _client=_client(responder))
    assert art.url == "https://blog.example/attn"
    assert art.title == "Attention Explained"
    assert "attention mechanism" in art.text.lower()
    # Boilerplate stripped.
    assert "home about contact" not in art.text.lower()
    assert "copyright" not in art.text.lower()


def test_fetch_article_http_error_raises_weberror():
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    with pytest.raises(web.WebError) as exc:
        web.fetch_article("https://paywall.example/x", _client=_client(responder))
    assert "403" in str(exc.value)


def test_fetch_article_empty_content_raises_weberror():
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html="<html><body><nav>menu</nav></body></html>")

    with pytest.raises(web.WebError) as exc:
        web.fetch_article("https://empty.example/x", _client=_client(responder))
    assert "no extractable content" in str(exc.value).lower()


def test_fetch_article_timeout_raises_weberror():
    def responder(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    with pytest.raises(web.WebError) as exc:
        web.fetch_article("https://slow.example/x", _client=_client(responder))
    assert "timeout" in str(exc.value).lower()


def test_fetch_article_non_html_raises_weberror():
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"%PDF-1.4 binary garbage",
                              headers={"Content-Type": "application/pdf"})

    with pytest.raises(web.WebError):
        web.fetch_article("https://files.example/x.pdf", _client=_client(responder))


def test_fetch_article_titleless_keeps_empty_title():
    body = ("<html><body><article><p>" + ("Plain prose without any heading. " * 20)
            + "</p></article></body></html>")

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=body)

    art = web.fetch_article("https://notitle.example/page", _client=_client(responder))
    assert art.title == ""
    assert "plain prose" in art.text.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_web.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.web'` (collection error).

- [ ] **Step 3: Write the implementation**

Create `src/web.py`:

```python
"""Fetch a URL and extract its readable article text.

This module is the one piece the wiki-ingest pipeline was missing for URLs:
turn a web page into clean markdown the LLM can summarize. It knows nothing
about the wiki or the chat server — it only does HTTP + boilerplate removal,
so it is testable with a mocked httpx transport and has no daemon dependency.

We use httpx (already a project dependency) for the fetch and trafilatura for
extraction. trafilatura strips nav/ads/footer chrome and emits the main
article body as markdown, and surfaces title/author/date metadata when the
page provides them.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx
import trafilatura


# A real browser User-Agent: many sites 403 a bare python-httpx client.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Below this, an "extraction" is almost certainly nav/boilerplate residue
# rather than a real article. Skip rather than ingest noise.
_MIN_CONTENT_CHARS = 200


class WebError(Exception):
    """Any failure fetching or extracting a URL (HTTP error, timeout,
    non-HTML, or no extractable article content)."""


@dataclass
class Article:
    url: str
    title: str = ""
    text: str = ""
    author: str = ""
    date: str = ""


def fetch_article(
    url: str,
    *,
    timeout: float = 20.0,
    _client: httpx.Client | None = None,
) -> Article:
    """Fetch ``url`` and return its extracted article as an :class:`Article`.

    Raises :class:`WebError` on HTTP error, timeout, non-HTML response, or when
    no article-like content can be extracted. ``_client`` is for tests only
    (inject an ``httpx.Client`` backed by a MockTransport).
    """
    client = _client or httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": _USER_AGENT},
    )
    try:
        try:
            resp = client.get(url)
        except httpx.TimeoutException as exc:
            raise WebError(f"timeout after {timeout:g}s fetching {url}") from exc
        except httpx.HTTPError as exc:
            raise WebError(f"fetch error for {url}: {exc}") from exc

        if resp.status_code >= 400:
            raise WebError(f"HTTP {resp.status_code} fetching {url}")

        ctype = resp.headers.get("Content-Type", "")
        if ctype and "html" not in ctype.lower():
            raise WebError(f"not HTML ({ctype}) at {url}")

        html = resp.text
    finally:
        if _client is None:
            client.close()

    # with_metadata=False: in trafilatura 2.x, with_metadata=True prepends a
    # YAML frontmatter block (title/date) to the markdown body. We get those
    # fields separately via extract_metadata() below, so keep the body clean.
    extracted = trafilatura.extract(
        html,
        output_format="markdown",
        with_metadata=False,
        include_comments=False,
        include_tables=True,
    )
    if not extracted or len(extracted.strip()) < _MIN_CONTENT_CHARS:
        raise WebError(f"no extractable content at {url}")

    meta = trafilatura.extract_metadata(html)
    title = (getattr(meta, "title", None) or "") if meta else ""
    author = (getattr(meta, "author", None) or "") if meta else ""
    date = (getattr(meta, "date", None) or "") if meta else ""

    return Article(
        url=url,
        title=title.strip(),
        text=extracted.strip(),
        author=author.strip(),
        date=date.strip(),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_web.py -v`
Expected: all 6 tests PASS. If `test_fetch_article_titleless_keeps_empty_title` fails because trafilatura infers a title from the first line, change that test's assertion to `assert art.title in ("", "Plain prose without any heading.")` — title inference varies by version and is not load-bearing.

- [ ] **Step 5: Commit**

```bash
git add src/web.py tests/test_web.py
git commit -m "feat(web): add fetch_article — httpx fetch + trafilatura extraction"
```

---

## Task 3: Refactor `ingest_cmd.py` — extract `ingest_text()` core

This is a pure refactor: behavior is unchanged, pinned by the existing `tests/test_ingest.py`. We move the body of `run()` (everything after reading the file) into `ingest_text()`.

**Files:**
- Modify: `src/ingest_cmd.py`

- [ ] **Step 1: Run the existing ingest tests to confirm green baseline**

Run: `.venv/bin/python -m pytest tests/test_ingest.py -v`
Expected: all PASS (this is the baseline the refactor must preserve).

- [ ] **Step 2: Add the result types and `ingest_text()` above `run()`**

In `src/ingest_cmd.py`, add near the top after the imports (the `from src import wiki` / `from src.server.client import ...` block):

```python
from dataclasses import dataclass
from pathlib import Path


WROTE = "WROTE"
ALREADY_EXISTS = "ALREADY_EXISTS"
REFUSED_HANDWRITTEN = "REFUSED_HANDWRITTEN"


@dataclass
class IngestResult:
    status: str               # WROTE | ALREADY_EXISTS | REFUSED_HANDWRITTEN
    page_path: Path | None
    summary: str
```

Then add the core function just above `def run(`:

```python
def ingest_text(
    body: str,
    *,
    page_name: str,
    source_label: str,
    extra_frontmatter: dict[str, str] | None = None,
    max_tokens: int = 1024,
    force: bool = False,
) -> IngestResult:
    """Summarize ``body`` via the chat server and write a wiki source page.

    This is the shared write-path for every ingest entry point (file, URL,
    batch). ``page_name`` is the (already-derived) wiki page stem; it is
    slugified here so callers don't have to. ``source_label`` is the
    provenance string the LLM sees (a file path or a URL).
    ``extra_frontmatter`` is merged into the page frontmatter (e.g.
    ``{"source-url": ...}``).

    Returns an :class:`IngestResult`. Raises ``HermesError`` if the chat
    server fails, and re-raises any exception from the index/log write after
    rolling back the page. Caller is responsible for the wiki-initialized
    check being meaningful (we re-check and raise via the status/return).
    """
    stem = _slugify(page_name)
    paths = wiki.get_paths()
    page_path = paths.sources_dir / f"{stem}.md"

    # Hand-written-file guard MUST run regardless of --force.
    if page_path.exists() and not wiki.is_managed(page_path):
        return IngestResult(REFUSED_HANDWRITTEN, page_path, "")
    if page_path.exists() and not force:
        return IngestResult(ALREADY_EXISTS, page_path, "")

    if len(body) > 32_000:
        body = body[:32_000] + "\n\n[truncated for length]"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(source_label, body)},
    ]
    client = HermesClient()
    body_out = client.chat_sync(messages, max_tokens=max_tokens, temperature=0.3)

    meta = {"source-path": source_label}
    if extra_frontmatter:
        meta.update(extra_frontmatter)
    page = wiki.Page(title=stem, body=body_out, frontmatter=meta)
    summary = _extract_summary(body_out)

    wiki.write_page(page_path, page)
    try:
        wiki.update_index_row(paths, "Sources", stem, summary)
        wiki.append_log(
            paths, "ingest", stem,
            detail=f"Source: {source_label}\nWiki page: {page_path.relative_to(paths.root.parent)}",
        )
    except Exception:
        try:
            page_path.unlink()
        except OSError:
            pass
        raise
    return IngestResult(WROTE, page_path, summary)
```

Note: `_build_user_prompt` currently takes `(source_path: Path, body)`. It only interpolates `{source_path}` into a string, so passing a `str` URL works unchanged. Update its annotation for honesty:

In `_build_user_prompt`, change the signature line:

```python
def _build_user_prompt(source_path, body: str) -> str:
```

(drop the `Path` type hint so a URL string is clearly acceptable; the body is unchanged.)

- [ ] **Step 3: Rewrite `run()` to call the core**

Replace the body of `run()` from the line `src = Path(args.path)...` through the final `return 0` with this. Keep the argparse block above it (the `path`, `--force`, `--name`, `--max-tokens` arguments) — we add `--url` in Task 5's CLI wiring, but `ingest_cmd.run` itself stays file-only; the URL path lives in cli.py. The new body:

```python
    src = Path(args.path).expanduser().resolve()
    if not src.is_file():
        print(f"hermes ingest: source not found: {src}", file=sys.stderr)
        return 2

    paths = wiki.get_paths()
    if not wiki.is_initialized(paths):
        print(f"hermes ingest: wiki not initialized at {paths.root}", file=sys.stderr)
        print(f"               run: hermes wiki init", file=sys.stderr)
        return 2

    print(f"hermes ingest: reading {src}", file=sys.stderr)
    body_in = _read_source(src)

    print(f"hermes ingest: calling chat server (this may take 30-60s)...", file=sys.stderr)
    try:
        res = ingest_text(
            body_in,
            page_name=args.name or src.stem,
            source_label=str(src),
            max_tokens=args.max_tokens,
            force=args.force,
        )
    except HermesError as exc:
        print(f"hermes ingest: chat server error: {exc}", file=sys.stderr)
        print(f"               (try: hermes doctor)", file=sys.stderr)
        return 1

    if res.status == REFUSED_HANDWRITTEN:
        print(f"hermes ingest: refusing to overwrite hand-written file: {res.page_path}",
              file=sys.stderr)
        return 1
    if res.status == ALREADY_EXISTS:
        print(f"hermes ingest: {res.page_path.name} already exists "
              f"(use --force to re-summarize).", file=sys.stderr)
        return 0

    print(f"hermes ingest: wrote {res.page_path.relative_to(paths.root.parent)}")
    print(f"               summary: {res.summary}")
    return 0
```

- [ ] **Step 4: Run the existing ingest tests — must still pass unchanged**

Run: `.venv/bin/python -m pytest tests/test_ingest.py -v`
Expected: all PASS, same as the Step 1 baseline. The refactor preserved behavior. If `test_ingest_rolls_back_page_on_index_failure` fails, verify the `try/except` around `update_index_row`/`append_log` is intact in `ingest_text`.

- [ ] **Step 5: Run the full suite to catch any import fallout**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all PASS (210 baseline + 7 new web tests = 217). NOTE: always scope to `tests/` — bare `pytest` recurses into `third_party/llama.cpp` and errors at collection on unrelated vendored tests.

- [ ] **Step 6: Commit**

```bash
git add src/ingest_cmd.py
git commit -m "refactor(ingest): extract shared ingest_text() core; run() delegates"
```

---

## Task 4: Add `ingest_text()` direct tests + URL-frontmatter coverage

**Files:**
- Modify: `tests/test_ingest.py`

- [ ] **Step 1: Write new tests at the end of `tests/test_ingest.py`**

Append:

```python
# ----------------------------------------------------------- ingest_text core


def test_ingest_text_writes_with_extra_frontmatter(vault, fake_chat):
    """The shared core writes a page and threads extra frontmatter (the
    URL-provenance path uses this for source-url)."""
    wiki.init_wiki()
    paths = wiki.get_paths()

    res = ingest_cmd.ingest_text(
        "Raw article body text.",
        page_name="My Article",
        source_label="https://example.com/my-article",
        extra_frontmatter={"source-url": "https://example.com/my-article",
                           "ingested-via": "url"},
    )
    assert res.status == ingest_cmd.WROTE
    page = paths.sources_dir / "My_Article.md"
    assert page.is_file()
    body = page.read_text()
    assert 'source-url: "https://example.com/my-article"' in body
    assert 'ingested-via: "url"' in body


def test_ingest_text_already_exists_is_idempotent(vault, fake_chat):
    wiki.init_wiki()
    ingest_cmd.ingest_text("body", page_name="dup", source_label="x")
    res = ingest_cmd.ingest_text("body", page_name="dup", source_label="x")
    assert res.status == ingest_cmd.ALREADY_EXISTS


def test_ingest_text_refuses_handwritten(vault, fake_chat):
    wiki.init_wiki()
    paths = wiki.get_paths()
    (paths.sources_dir / "hand.md").write_text("# mine\n\nhand-written\n")
    res = ingest_cmd.ingest_text("body", page_name="hand", source_label="x", force=True)
    assert res.status == ingest_cmd.REFUSED_HANDWRITTEN
    assert (paths.sources_dir / "hand.md").read_text() == "# mine\n\nhand-written\n"
```

- [ ] **Step 2: Run the new tests**

Run: `.venv/bin/python -m pytest tests/test_ingest.py -k ingest_text -v`
Expected: 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ingest.py
git commit -m "test(ingest): cover ingest_text core + extra-frontmatter path"
```

---

## Task 5: CLI wiring — `--url` on `ingest`, new `ingest-links` subcommand

**Files:**
- Modify: `src/cli.py`

- [ ] **Step 1: Add `--url` to the `ingest` subparser and a URL branch in `_cmd_ingest`**

In `src/cli.py`, in the `ingest` subparser block (where `ingest.add_argument("path", ...)` is), change `path` to be optional and add `--url`:

```python
    ingest = sub.add_parser(
        "ingest",
        help="Summarize a raw source into a wiki page (writes wiki/sources/<stem>.md).",
    )
    ingest.add_argument("path", nargs="?", default=None,
                        help="Path to the source file.")
    ingest.add_argument("--url", default=None,
                        help="Fetch and summarize a web page instead of a local file.")
    ingest.add_argument("--force", action="store_true",
                        help="Overwrite an existing wiki page.")
    ingest.add_argument("--name", default=None,
                        help="Override the wiki page name.")
    ingest.add_argument("--max-tokens", type=int, default=1024,
                        help="Cap on chat server response.")
    ingest.set_defaults(func=_cmd_ingest)
```

Then replace `_cmd_ingest` with a version that branches on `--url`:

```python
def _cmd_ingest(args: argparse.Namespace) -> int:
    from src import ingest_cmd

    if args.url and args.path:
        print("hermes ingest: pass either a path or --url, not both.", file=sys.stderr)
        return 2
    if args.url:
        return _ingest_one_url(
            args.url, name=args.name, force=args.force, max_tokens=args.max_tokens,
        )
    if not args.path:
        print("hermes ingest: provide a file path or --url <url>.", file=sys.stderr)
        return 2

    flags: list[str] = [args.path]
    if args.force:
        flags.append("--force")
    if args.name:
        flags.extend(["--name", args.name])
    flags.extend(["--max-tokens", str(args.max_tokens)])
    return ingest_cmd.run(flags)
```

- [ ] **Step 2: Add the shared single-URL helper used by both `--url` and the batch driver**

Still in `src/cli.py`, add this helper (it returns a process exit code and prints; the batch driver in Task 6 has its own loop and does NOT use this — it calls the lower-level pieces directly. This helper is only for the single `--url` path):

```python
def _ingest_one_url(url: str, *, name: str | None, force: bool, max_tokens: int) -> int:
    from src import web, ingest_cmd, wiki

    paths = wiki.get_paths()
    if not wiki.is_initialized(paths):
        print(f"hermes ingest: wiki not initialized at {paths.root}", file=sys.stderr)
        print(f"               run: hermes wiki init", file=sys.stderr)
        return 2

    try:
        art = web.fetch_article(url)
    except web.WebError as exc:
        print(f"hermes ingest: {exc}", file=sys.stderr)
        return 1

    page_name = name or art.title or _name_from_url(url)
    extra = {"source-url": url, "ingested-via": "url"}
    if art.title:
        extra["source-title"] = art.title
    if art.date:
        extra["source-date"] = art.date

    print(f"hermes ingest: fetched {url} — summarizing (30-60s)...", file=sys.stderr)
    try:
        res = ingest_cmd.ingest_text(
            art.text, page_name=page_name, source_label=url,
            extra_frontmatter=extra, max_tokens=max_tokens, force=force,
        )
    except HermesError as exc:
        print(f"hermes ingest: chat server error: {exc}", file=sys.stderr)
        return 1

    if res.status == ingest_cmd.REFUSED_HANDWRITTEN:
        print(f"hermes ingest: refusing to overwrite hand-written file: {res.page_path}",
              file=sys.stderr)
        return 1
    if res.status == ingest_cmd.ALREADY_EXISTS:
        print(f"hermes ingest: {res.page_path.name} already exists "
              f"(use --force to re-summarize).", file=sys.stderr)
        return 0
    print(f"hermes ingest: wrote {res.page_path.name} — {res.summary}")
    return 0


def _name_from_url(url: str) -> str:
    """Derive a page name from a URL when there's no title: last non-empty
    path segment, else the host."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    segs = [s for s in parsed.path.split("/") if s]
    if segs:
        return segs[-1].rsplit(".", 1)[0]  # drop a trailing .html etc.
    return parsed.netloc or url
```

- [ ] **Step 3: Add the `ingest-links` subparser + dispatch**

In `_build_parser`, after the `ingest` subparser block, add:

```python
    ingest_links = sub.add_parser(
        "ingest-links",
        help="Fetch + summarize every URL in a text file (one URL per line).",
    )
    ingest_links.add_argument("file", help="Path to a text file of URLs (one per line; "
                                           "blank lines and # comments ignored).")
    ingest_links.add_argument("--force", action="store_true",
                              help="Re-summarize URLs whose wiki page already exists.")
    ingest_links.add_argument("--max-tokens", type=int, default=1024,
                              help="Cap on chat server response per page.")
    ingest_links.add_argument("--no-index", action="store_true",
                              help="Skip auto-indexing the new pages at the end.")
    ingest_links.set_defaults(func=_cmd_ingest_links)
```

And add the dispatch function:

```python
def _cmd_ingest_links(args: argparse.Namespace) -> int:
    from src import ingest_links_cmd
    flags = [args.file, "--max-tokens", str(args.max_tokens)]
    if args.force:
        flags.append("--force")
    if args.no_index:
        flags.append("--no-index")
    return ingest_links_cmd.run(flags)
```

- [ ] **Step 4: Smoke-check the CLI parses (no server needed for --help)**

Run: `.venv/bin/python -m src.cli ingest --help`
Expected: usage text showing `[--url URL] [--force] [--name NAME] [--max-tokens MAX_TOKENS] [path]`.

Run: `.venv/bin/python -m src.cli ingest-links --help`
Expected: usage text showing `file`, `--force`, `--max-tokens`, `--no-index`.

- [ ] **Step 5: Run the full suite (nothing should break)**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cli.py
git commit -m "feat(cli): wire ingest --url and ingest-links subcommands"
```

---

## Task 6: `src/ingest_links_cmd.py` — batch driver (test first)

**Files:**
- Create: `tests/test_ingest_links_cmd.py`
- Create: `src/ingest_links_cmd.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ingest_links_cmd.py`:

```python
"""Tests for src/ingest_links_cmd.py — the batch URL driver.

web.fetch_article and the chat client are monkeypatched; index_cmd.run is
monkeypatched so the batch test needs no embed server. All-mocked, no network.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src import ingest_links_cmd, ingest_cmd, web, wiki


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_WIKI_PATH", str(tmp_path / "wiki"))
    monkeypatch.delenv("HERMES_VAULT_PATH", raising=False)
    return tmp_path


@pytest.fixture
def fake_chat(monkeypatch):
    def _stub(self, messages, **_kw):
        return ("## Summary\nS.\n\n## Key claims\n- c\n\n"
                "## Entities and concepts\n- [[E]]: x.\n\n## Open questions\n- q\n")
    monkeypatch.setattr("src.server.client.HermesClient.chat_sync", _stub)


@pytest.fixture
def no_index(monkeypatch):
    """Stub index_cmd.run so auto-index doesn't hit the embed server.
    Returns a list that records calls."""
    calls = []
    def _stub(argv=None):
        calls.append(argv)
        return 0
    monkeypatch.setattr("src.index_cmd.run", _stub)
    return calls


def _fake_fetch(mapping):
    """Build a fetch_article stand-in from {url: Article-or-WebError}."""
    def _f(url, **_kw):
        v = mapping[url]
        if isinstance(v, Exception):
            raise v
        return v
    return _f


def _links_file(tmp_path: Path, urls: list[str]) -> Path:
    f = tmp_path / "links.txt"
    f.write_text("\n".join(urls) + "\n")
    return f


def test_batch_ingests_all_good(vault, fake_chat, no_index, monkeypatch):
    wiki.init_wiki()
    paths = wiki.get_paths()
    urls = ["https://a.example/one", "https://b.example/two", "https://c.example/three"]
    mapping = {u: web.Article(url=u, title=u.rsplit("/", 1)[1], text="body " * 50)
               for u in urls}
    monkeypatch.setattr("src.web.fetch_article", _fake_fetch(mapping))

    f = _links_file(vault, urls)
    rc = ingest_links_cmd.run([str(f)])
    assert rc == 0
    idx = paths.index.read_text()
    assert "[one](sources/one.md)" in idx
    assert "[two](sources/two.md)" in idx
    assert "[three](sources/three.md)" in idx
    # Auto-index ran once with --backfill.
    assert no_index and "--backfill" in no_index[0]


def test_batch_skips_failures_and_writes_failed_file(vault, fake_chat, no_index, monkeypatch):
    wiki.init_wiki()
    good = "https://a.example/ok"
    bad = "https://paywall.example/x"
    mapping = {
        good: web.Article(url=good, title="ok", text="body " * 50),
        bad: web.WebError("HTTP 403 fetching " + bad),
    }
    monkeypatch.setattr("src.web.fetch_article", _fake_fetch(mapping))

    f = _links_file(vault, [good, bad])
    rc = ingest_links_cmd.run([str(f)])
    # Non-zero because at least one failed.
    assert rc == 1
    failed = f.with_suffix(".failed.txt")
    assert failed.is_file()
    assert bad in failed.read_text()
    assert good not in failed.read_text()


def test_batch_rerun_is_idempotent(vault, fake_chat, no_index, monkeypatch):
    wiki.init_wiki()
    u = "https://a.example/one"
    mapping = {u: web.Article(url=u, title="one", text="body " * 50)}
    monkeypatch.setattr("src.web.fetch_article", _fake_fetch(mapping))
    f = _links_file(vault, [u])

    assert ingest_links_cmd.run([str(f)]) == 0
    # Second run: page already exists → ALREADY_EXISTS, no duplicate row.
    assert ingest_links_cmd.run([str(f)]) == 0
    idx = wiki.get_paths().index.read_text()
    assert idx.count("[one](sources/one.md)") == 1


def test_batch_skips_comments_blanks_and_nonhttp(vault, fake_chat, no_index, monkeypatch):
    wiki.init_wiki()
    u = "https://a.example/one"
    mapping = {u: web.Article(url=u, title="one", text="body " * 50)}
    fetched = []
    def _f(url, **_kw):
        fetched.append(url)
        return mapping[url]
    monkeypatch.setattr("src.web.fetch_article", _f)

    f = vault / "links.txt"
    f.write_text("\n".join([
        "# a comment", "", u, "  ", "ftp://nope.example/x", u,  # dup at end
    ]) + "\n")
    rc = ingest_links_cmd.run([str(f)])
    assert rc == 0
    # Only the one valid http URL fetched, and only once (de-duped).
    assert fetched == [u]


def test_batch_no_index_flag_skips_indexing(vault, fake_chat, no_index, monkeypatch):
    wiki.init_wiki()
    u = "https://a.example/one"
    mapping = {u: web.Article(url=u, title="one", text="body " * 50)}
    monkeypatch.setattr("src.web.fetch_article", _fake_fetch(mapping))
    f = _links_file(vault, [u])
    rc = ingest_links_cmd.run([str(f), "--no-index"])
    assert rc == 0
    assert no_index == []  # index_cmd.run never called


def test_batch_uninitialized_wiki_fails_before_fetch(vault, monkeypatch):
    # Wiki NOT initialized.
    called = []
    monkeypatch.setattr("src.web.fetch_article",
                        lambda *a, **k: called.append(1))
    f = _links_file(vault, ["https://a.example/one"])
    rc = ingest_links_cmd.run([str(f)])
    assert rc == 2
    assert called == []  # never fetched


def test_batch_missing_file_returns_2(vault):
    rc = ingest_links_cmd.run([str(vault / "nope.txt")])
    assert rc == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ingest_links_cmd.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingest_links_cmd'`.

- [ ] **Step 3: Write the implementation**

Create `src/ingest_links_cmd.py`:

```python
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
import sys
from pathlib import Path
from urllib.parse import urlparse

from src import web, ingest_cmd, wiki
from src.server.client import HermesError


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
        from src import index_cmd
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ingest_links_cmd.py -v`
Expected: all 7 PASS.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all PASS (220 + 7 = 227).

- [ ] **Step 6: Commit**

```bash
git add src/ingest_links_cmd.py tests/test_ingest_links_cmd.py
git commit -m "feat(ingest-links): batch URL driver with skip-and-continue + auto-index"
```

---

## Task 7: Doctor probe for trafilatura

**Files:**
- Modify: `src/doctor.py`
- Modify: `tests/test_doctor_plist.py` (or wherever doctor tests live — confirm with `ls tests/ | grep doctor`)

- [ ] **Step 1: Add a `check_web()` section to `src/doctor.py`**

Add this function near `check_python()`:

```python
def check_web() -> Section:
    """Probe the optional web-ingest extractor (trafilatura). Web ingest is
    additive; if trafilatura is missing, `ingest --url` / `ingest-links` are
    the only things affected, so this is a WARN, not a FAIL."""
    s = Section("Web ingest")
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    py = str(venv_py) if venv_py.is_file() else sys.executable
    rc, _out, err = _run([py, "-c", "import trafilatura"], timeout=10.0)
    if rc == 0:
        s.add(Result("trafilatura", OK, "import OK (web ingest available)"))
    else:
        s.add(Result(
            "trafilatura", WARN,
            "not installed — `hermes ingest --url` / `ingest-links` unavailable",
            fix=".venv/bin/pip install trafilatura",
        ))
    return s
```

Note: confirm `sys` is imported at the top of `doctor.py` (it is — it's used elsewhere). Confirm `_run` exists (it's used by `check_python`).

- [ ] **Step 2: Register the section in doctor's `run_all()`**

Sections are assembled in `run_all()` (around line 861) as a returned list `[check_host(), check_repo(), ..., check_digest()]`. Add `check_web()` immediately after `check_python(),`:

```python
        check_python(),
        check_web(),
        check_models(),
```

(The `check_models()` line is shown only as an anchor — it already exists; insert `check_web(),` between `check_python(),` and `check_models(),`.)

- [ ] **Step 3: Manually verify doctor shows the section**

Run: `.venv/bin/python -m src.cli doctor 2>&1 | grep -A2 "Web ingest"`
Expected (trafilatura installed in Task 1):
```
Web ingest
  [OK  ] trafilatura  import OK (web ingest available)
```

- [ ] **Step 4: Add a doctor test**

First check the file: `ls tests/ | grep doctor` → `test_doctor_plist.py`. Append to it:

```python
def test_check_web_reports_trafilatura():
    from src import doctor
    section = doctor.check_web()
    assert section.title == "Web ingest"
    assert len(section.results) == 1
    # trafilatura is a project dep (installed), so it should be OK; if a CI
    # env lacks it the WARN path is also acceptable — assert it's one of them.
    assert section.results[0].level in (doctor.OK, doctor.WARN)
```

- [ ] **Step 5: Run the doctor test**

Run: `.venv/bin/python -m pytest tests/test_doctor_plist.py -k check_web -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/doctor.py tests/test_doctor_plist.py
git commit -m "feat(doctor): probe trafilatura availability for web ingest"
```

---

## Task 8: README documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the new commands**

In `README.md`, in the Commands table, update the `hermes ingest` row and add an `ingest-links` row:

```
| `hermes ingest <path>` / `--url <url>` | Summarize a local file or a fetched web page into `wiki/sources/`. |
| `hermes ingest-links <file>` | Fetch + summarize every URL in a text file; skips failures, auto-indexes. |
```

And in the "The wiki" section, after the existing `hermes ingest` example, add:

```sh
# Ingest from the web instead of a local file:
hermes ingest --url https://example.com/article

# Or a whole reading list (one URL per line; # comments and blanks ignored):
hermes ingest-links ~/reading.txt
# Failures are skipped and written to ~/reading.failed.txt for a retry run.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document ingest --url and ingest-links"
```

---

## Task 9: IMPROVEMENTS.md changelog entry

**Files:**
- Modify: `IMPROVEMENTS.md`

- [ ] **Step 1: Add a dated entry at the top (after the intro, before the most recent existing entry)**

```markdown
## 2026-06-12 — Web ingestion (`ingest --url` + `ingest-links`)

**Problem:** Ingest only accepted local files. The natural "second brain" inflow
— a browser full of tabs/links — had no path in; the user had to manually save
each page to disk first.

**Change:** New `src/web.py` (`fetch_article`: httpx fetch + trafilatura
boilerplate-stripping → clean markdown, with title/date metadata). Refactored
`src/ingest_cmd.py` to expose a shared `ingest_text()` core (page → index → log
write-with-rollback) that file, URL, and batch ingestion all call — one tested
write-path, no duplication. New `src/ingest_links_cmd.py`: a skip-and-continue
batch driver over a text file of URLs, idempotent on re-run, writing failed URLs
to `<input>.failed.txt` (a valid links file for retry) and auto-indexing new
pages via the existing `--backfill`. CLI gains `hermes ingest --url <url>` and
`hermes ingest-links <file> [--force] [--max-tokens] [--no-index]`. `doctor`
gains a Web-ingest section probing trafilatura. Added `trafilatura` to
requirements.

**Files:** `src/web.py` (new), `src/ingest_cmd.py` (refactor), `src/ingest_links_cmd.py`
(new), `src/cli.py`, `src/doctor.py`, `requirements.txt`, `README.md`,
`tests/test_web.py` (new), `tests/test_ingest_links_cmd.py` (new),
`tests/test_ingest.py` (extended), `tests/test_doctor_plist.py` (extended).

**Impact:** A reading list of links becomes a set of cross-linked wiki source
pages in one command, retrievable on the next query. Provenance (`source-url`)
is preserved in frontmatter. Tests: 210 → 228.
```

- [ ] **Step 2: Commit**

```bash
git add IMPROVEMENTS.md
git commit -m "docs: changelog entry for web ingestion"
```

---

## Task 10: End-to-end manual verification (real servers)

The engine (:8080) and embed (:8081) servers are running manually this session.
This task confirms the feature works against a real page and the live chat model.

- [ ] **Step 1: Confirm servers are up**

Run: `.venv/bin/python -m src.cli doctor 2>&1 | grep -A3 "^Servers"`
Expected: both chat and embed servers `[OK]`. If down, restart per the session notes (manual `llama-server` launch).

- [ ] **Step 2: Ensure the wiki is initialized**

Run: `HERMES_VAULT_PATH=/Users/jaehunb/Documents/Obsidian .venv/bin/python -m src.cli wiki init`
Expected: bootstrap message or "already initialized".

- [ ] **Step 3: Single URL ingest against a real, stable page**

Run: `HERMES_VAULT_PATH=/Users/jaehunb/Documents/Obsidian .venv/bin/python -m src.cli ingest --url https://en.wikipedia.org/wiki/Attention_(machine_learning)`
Expected: `fetched ... summarizing` then `wrote <name>.md — <summary>`. Inspect the page:
Run: `ls /Users/jaehunb/Documents/Obsidian/wiki/sources/ && head -20 /Users/jaehunb/Documents/Obsidian/wiki/sources/*.md`
Expected: a managed page with `source-url:` frontmatter, `## Summary`, and `[[entities]]`.

- [ ] **Step 4: Batch ingest with one good + one dead link**

```bash
cat > /tmp/hermes-links.txt <<'EOF'
# my reading list
https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)
https://this-domain-does-not-exist-zzz.example/page
EOF
HERMES_VAULT_PATH=/Users/jaehunb/Documents/Obsidian .venv/bin/python -m src.cli ingest-links /tmp/hermes-links.txt
```
Expected: `[1/2] ok ...`, `[2/2] FAIL ...`, `Done: 1 ingested, 0 already present, 1 failed.`, a `/tmp/hermes-links.failed.txt` containing the dead URL, and an "indexing 1 new page(s)" line.

- [ ] **Step 5: Confirm the new page is retrievable**

Run: `HERMES_VAULT_PATH=/Users/jaehunb/Documents/Obsidian .venv/bin/python -m src.cli ask "what did I just read about transformers?"`
Expected: a streamed answer citing the new source page.

- [ ] **Step 6: Clean up the test artifacts (optional)**

```bash
rm -f /tmp/hermes-links.txt /tmp/hermes-links.failed.txt
```
Leave the wiki pages — they're real content. If you want them gone:
`rm /Users/jaehunb/Documents/Obsidian/wiki/sources/<name>.md` and re-run `hermes index --gc`.

- [ ] **Step 7: Final full-suite run**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all PASS (~228 tests).

---

## Self-Review Notes

**Spec coverage:** web.py (Task 2), ingest_text refactor (Task 3), batch driver (Task 6), CLI `--url` + `ingest-links` (Task 5), trafilatura dep (Task 1), doctor probe (Task 7), all tests (Tasks 2/4/6/7), edge cases (failure policy + `.failed.txt` in Task 6, hand-written guard preserved in Task 3, auto-index + `--no-index` in Tasks 5/6, fail-fast-before-fetch in Task 6). README + changelog (Tasks 8/9). E2E verify (Task 10). All spec sections map to a task.

**Type consistency:** `Article`, `WebError`, `fetch_article(url, *, timeout, _client)`, `IngestResult(status, page_path, summary)`, `ingest_text(body, *, page_name, source_label, extra_frontmatter, max_tokens, force)`, status constants `WROTE`/`ALREADY_EXISTS`/`REFUSED_HANDWRITTEN` — used identically across Tasks 2–6. `index_cmd.run(["--backfill"])` matches the real signature. `_name_from_url` defined in both cli.py and ingest_links_cmd.py (intentional — they're separate entry points; the duplication is 6 lines and avoids a cross-module import just for a helper).

**Placeholder scan:** no TBD/TODO; every code step shows complete code; every test step shows the assertion; every run step states the expected output.
