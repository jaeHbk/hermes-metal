# URL Ingestion — Design

**Date:** 2026-06-12
**Status:** Approved (design phase complete; ready for implementation plan)
**Branch:** `feat/phases-b-e`

## Goal

Let a user collect a series of web links (e.g. tabs/bookmarks from Google
Chrome), hand them to hermes as a plain list, and have hermes fetch each page,
extract the readable article, summarize it through the existing wiki-ingest
pipeline, and produce per-source wiki pages that auto-link to each other via
shared `[[entities]]` — "a wiki to read."

The back half of this already exists: `src/ingest_cmd.py` reads a local file,
summarizes it via the chat server, and writes `wiki/sources/<stem>.md` with a
page → index → log write-and-rollback sequence. The only true gap is
**URL → clean local text**, plus a **batch driver** over a list of URLs.

## Non-goals (v1)

- No PDF / image extraction. Non-HTML fetches that yield no extractable text
  are skipped with a clear reason (future extension).
- No synthesis/overview page across sources. Output is per-source pages only;
  cross-page connection happens through `[[wiki-link]]` entities in Obsidian's
  graph, exactly as file ingestion already does.
- No Chrome bookmarks.html parsing. Input is a plain text file of URLs.
- No parallel fetching. Sequential, because the 8B chat server summarizes one
  page at a time and is the bottleneck (~30–60s/page).
- No fetch retries. A failed URL is recorded to a `.failed.txt` file for a
  manual retry run.

## Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| HTML → text extraction | `trafilatura` (markdown output, with metadata) | Best boilerplate stripping; yields title + publish date. Pure-Python, arm64-safe wheel. |
| Input format | Plain text file, one URL per line (`#` comments + blank lines ignored). Single URL via `hermes ingest --url`. | Simplest to produce from Chrome; re-runnable; `.failed.txt` round-trips as input. |
| Output shape | One `wiki/sources/<name>.md` per URL; cross-link via shared `[[entities]]`. | Reuses the existing, tested ingest write-path. Ships now. |
| Failure mode (batch) | Skip-and-continue; end summary; write failed URLs to `<input>.failed.txt`. | Forgiving for "paste 20 links, walk away." |
| Indexing | Auto-index new pages at end of batch (and after single `--url`); `--no-index` to skip. | Watcher is the normal path but isn't guaranteed up; makes pages immediately searchable. |
| Integration | Approach A: extract a shared `ingest_text(...)` core; file + URL + batch all call it. | One tested write-path; no duplication of the bug-prone rollback logic. |

## Architecture

Three isolated, independently testable units plus CLI wiring.

```
src/web.py                          (NEW — fetch + extract)
  fetch_article(url, *, timeout=20.0) -> Article
  Article = dataclass(url, title, text, author, date)
  WebError(Exception)
  • httpx GET with a browser User-Agent, follow_redirects=True
  • trafilatura.extract(html, output_format='markdown', with_metadata=True)
  • raises WebError on HTTP error / timeout / empty (<200 char) extraction
  • imports: httpx, trafilatura.  Does NOT import wiki / server / llm.

src/ingest_cmd.py                   (REFACTOR — extract reusable core)
  ingest_text(body, *, page_name, source_label, extra_frontmatter=None,
              max_tokens=1024, force=False) -> IngestResult
  IngestResult = dataclass(status, page_path, summary)
      status in {WROTE, ALREADY_EXISTS, REFUSED_HANDWRITTEN}
  • the existing logic, parameterized: wiki-init check → hand-written guard →
    existing-managed skip → 32k truncate → chat summarize →
    write_page → (update_index_row → append_log) with page-unlink rollback
  • run(argv)  [file-path CLI] becomes: resolve path → read → ingest_text(...)
  • imports: wiki, server.client.  Write/rollback logic UNCHANGED.

src/ingest_links_cmd.py             (NEW — batch driver)
  run(argv) -> int
  • parse links file (strip, skip blank/`#`, validate http(s), de-dupe)
  • fail fast if wiki not initialized (BEFORE any fetch)
  • for each url: fetch_article → ingest_text, inside per-URL try/except
  • skip-and-continue; classify WROTE / ALREADY_EXISTS / failed
  • auto-index new pages unless --no-index
  • end summary; write <input>.failed.txt if any failed
  • imports: web, ingest_cmd, index_cmd.  No wiki logic of its own.

src/cli.py                          (WIRING)
  • add --url to the existing `ingest` subparser (mutually exclusive w/ path)
  • add `ingest-links <file> [--force] [--max-tokens N] [--no-index]`
  • delegate, matching the existing _cmd_ingest -> ingest_cmd.run pattern
```

### Boundaries

- `web.py` turns a URL into clean text and knows nothing about wikis or the
  LLM. Testable with a mocked `httpx` transport.
- `ingest_text(...)` is the single write-path. File, URL, and batch ingestion
  all funnel through it, so the rollback logic that adversarial review hardened
  lives in exactly one place.
- The batch driver is orchestration only — a loop with error handling over
  `fetch_article` + `ingest_text` + an index call. Zero persistence logic.

## Data flow (single URL)

`hermes ingest --url https://blog.foo.dev/attention-explained`

1. `cli._cmd_ingest` sees `args.url` → routes to the URL path.
2. `web.fetch_article(url)`:
   - `httpx.get(url, headers={User-Agent}, follow_redirects=True, timeout=20)`;
     HTTP error / timeout → `WebError`.
   - `trafilatura.extract(html, output_format='markdown', with_metadata=True)`;
     None / empty / <200 chars → `WebError("no extractable content")`.
   - → `Article(url, title, text, author, date)`.
3. Derive `page_name = _slugify(title or last-URL-segment or host)` (existing
   `_slugify`, including the index/log collision guard).
4. `ingest_text(body=Article.text, page_name=..., source_label=url,
   extra_frontmatter={source-url, source-title, source-date?, ingested-via:url},
   max_tokens, force)`:
   - wiki initialized? else error with `run: hermes wiki init`.
   - hand-written guard: page exists & not managed → refuse even with `--force`.
   - page exists & managed & not `--force` → skip → `ALREADY_EXISTS`.
   - truncate body to 32k chars.
   - `client.chat_sync(system=INGEST_PROMPT, user=_build_user_prompt(url, body))`.
   - `write_page` → try `update_index_row("Sources", …)` + `append_log(…)`;
     on failure unlink the page (rollback) and re-raise.
   - → `IngestResult(WROTE, page_path, summary)`.
5. Print result. If indexing enabled, run `index_cmd` backfill (indexes the new
   page, since it isn't in the index yet).

**Provenance:** the URL is stored in `source-url` frontmatter and is what the
LLM sees in its prompt (instead of a file path). The `## Entities and concepts`
section still emits `[[wiki-links]]`, so cross-page graph linking is identical
to file ingestion.

## Batch driver (`ingest-links <file>`)

1. Parse: one URL/line; strip; skip blank + `#`; validate scheme is http/https
   (else count malformed); de-dupe within file; file-not-found/empty → exit 2.
2. Fail fast if wiki not initialized (before any network call).
3. For each url (i of N), inside try/except:
   - `art = web.fetch_article(url)`  → `WebError` → skip, record reason.
   - `res = ingest_cmd.ingest_text(...)`  → `HermesError` → skip, record.
   - classify: `WROTE` → ok++; `ALREADY_EXISTS` → skipped++;
     `REFUSED_HANDWRITTEN` → failed++.
   - one bad URL never aborts the loop.
4. If any pages written and not `--no-index`: run `index_cmd` backfill. The
   existing `--backfill` already indexes only files not yet in the index, so it
   naturally embeds just the new pages without a special scoping mechanism.
5. End summary: `Done: <ok> ingested, <skipped> already present, <failed>
   failed.` If failed, print each url + reason and write the failed URLs to
   `<input>.failed.txt` (a valid links file → retry via
   `hermes ingest-links <input>.failed.txt`).

`--force` and `--max-tokens` pass through to every URL. `--force`
re-summarizes managed pages but never touches hand-written files.

## Edge cases

| Case | Behavior |
|---|---|
| No title from trafilatura | name falls back to last URL segment, then host; `_slugify` guard applies |
| Two URLs slug to same name | second gets a `-2` suffix; neither clobbers the other |
| Page exists & hand-written | refuse (never overwrite); count failed — same invariant as file ingest |
| Page exists & managed, no `--force` | skip as `ALREADY_EXISTS` (idempotent re-run) |
| Non-HTML (PDF/image) fetch | trafilatura empty → `WebError("no extractable content")` → skip |
| Network fully down | each URL fails fast; all reported failed; `.failed.txt` written |
| Wiki not initialized | fail fast before fetching anything, with `run: hermes wiki init` |
| Within-file duplicate URL | fetched once |

**Timeout & politeness:** 20s per-request timeout (matches `notify.py`); a real
browser User-Agent so sites don't 403 a bare Python client; follow redirects;
no retry in v1.

## Dependencies & config

- Add `trafilatura>=1.12.0` to `requirements.txt` (pure-Python, arm64 wheel).
  `httpx` is already a dependency.
- `doctor` gains a `trafilatura` import probe (like the existing lancedb/httpx
  probes): OK, or `fix: pip install trafilatura`. SKIP-friendly so a missing
  extractor reports clearly rather than crashing a batch.

## Testing

Ethos: pure-Python, no daemons, no network — `httpx` mocked via `MockTransport`
(as in `test_streaming.py`); chat client mocked (as in `test_ingest.py`).

**New `tests/test_web.py`** — `fetch_article` in isolation:
- clean HTML → correct title/text/date
- HTTP 403/404/500 → `WebError` with status in message
- timeout → `WebError("timeout …")`
- empty / boilerplate-only (trafilatura None or <200 chars) → `WebError`
- non-HTML (PDF bytes) → `WebError`
- redirect followed to final URL
- title-less page → falls back to URL segment / host

**New `tests/test_ingest_links_cmd.py`** — driver with `web.fetch_article` and
chat client monkeypatched:
- 3 good URLs → 3 pages, 3 index rows, summary "3 ingested"
- 2 good / 2 failing → 2 written, 2 in `.failed.txt`, loop did not abort
- re-run same file → all `ALREADY_EXISTS`, no duplicate index rows
- `.failed.txt` round-trips as a valid input file
- malformed lines (blank, `#`, non-http) skipped
- within-file duplicate fetched once
- hand-written collision → refused + counted failed; others still ingested
- `--no-index` skips index step; default triggers it (assert mocked index call)
- wiki-not-initialized → fails before any fetch (assert `fetch_article` not called)

**Extended `tests/test_ingest.py`**:
- existing file-path tests pass unchanged (pins the refactor)
- `ingest --url` path calls `ingest_text` with `source-url` frontmatter
- existing TOCTOU / rollback regression tests keep covering the shared core

**Extended doctor tests**: `trafilatura` import probe reports OK / `fix:` hint.

Target: every new code path covered; the refactor is pinned by existing ingest
tests so file-ingestion behavior cannot silently change.
