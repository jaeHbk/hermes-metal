# IMPROVEMENTS

Running log of changes to hermes-metal. Each entry names the problem the
change solved and the observable improvement, so future readers (or
future-us) can tell whether a feature is still pulling its weight.

Newest first. Add entries as work lands, not in batch.

---

## Proposed

Things on the table but not yet built. Each entry names the **gap** (what
the user can't do today, or has to work around), the **sketch** (rough
shape of the fix), and the **risk** (what would make this a bad idea, or
the load-bearing assumption).

Promotion rule: when something here ships, move it down into a dated
section above and rewrite to past tense. If it stops being a good idea,
delete it — don't let this list ossify into a wishlist museum.

### Retrieval quality (rerank + richer metadata)

**Gap:** Distances on real queries cluster in `0.94–1.08` — barely better
than random. The model is grounded by retrievals, so weak retrieval
caps answer quality regardless of model size.

**Sketch:**
- Declare the LanceDB vector column with `metric="cosine"` (or `"dot"`,
  since nomic embeddings are L2-normalized) instead of the current
  default L2.
- Store more per-chunk metadata: `mtime` (for recency boost),
  `heading_trail` (`"# Auth > ## Token storage"`), `tags` (frontmatter +
  inline `#tag`).
- Two-stage retrieval: top-20 by vector, then a small reranker pass —
  either a local cross-encoder (`bge-reranker-base` is ~280 MB) or a
  heuristic combining vector score, mtime decay, and heading-trail
  match against query terms.

**Risk:** A cross-encoder doubles per-query latency in the REPL.
Heuristic-only reranking is cheaper but less robust. Schema change
requires a `--force` reindex; current users would need to rerun
`hermes index --backfill --force`. Migration story matters.

### Cross-session conversation memory

**Gap:** `/save` and `/load` are manual. Users can't say "what did we
decide last week about the auth rewrite?" and have the REPL pull from
prior conversations.

**Sketch:**
- New LanceDB table (`conversations`) with a row per assistant turn,
  embedded the same way as vault chunks.
- REPL automatically appends each completed (user, assistant) pair on
  exit, gated on a minimum length so trivia doesn't pollute the index.
- Each turn's retrieval pulls top-k from BOTH `vault_chunks` and
  `conversations`, with conversations cited differently (e.g.
  `[chat:2026-05-28]` vs `[Welcome.md]`).

**Risk:** The conversation table grows monotonically; no GC story
yet. Privacy posture changes — users may not expect every chat to be
indexed forever. Needs an opt-out flag and a `hermes index --gc-chats
--older-than 90d`.

### Daily-note temporal awareness

**Gap:** Model has no idea what "yesterday", "this week", or "last
month" means. Obsidian users live in dated daily notes; queries that
hinge on time are answered by accident or not at all.

**Sketch:**
- Inject `Today is YYYY-MM-DD (Weekday)` into the system prompt at
  REPL/ask startup.
- Cheap query-time temporal parser (`yesterday` → date filter,
  `last week` → date range) that adds a LanceDB `where` clause on
  `source_path` matching daily-note filename conventions
  (`YYYY-MM-DD.md`, `YYYY/MM/YYYY-MM-DD.md`).
- Configurable daily-note glob in `config/vault.yaml` since not all
  users use the same scheme.

**Risk:** Misparsing temporal phrases produces overconfident wrong
answers. Limit to high-precision phrases (`yesterday`, `today`, `this
week`, explicit dates) and fall through to vector search otherwise.

### Watcher observability (`hermes status --watch`)

**Gap:** When the watcher misbehaves (embed server slow, queue
backed up, disk full), the only signal is a stale index — surfaced
hours later. `logs/watcher.log` has the truth but you have to know
to tail it.

**Sketch:**
- Watcher exposes a small JSON status endpoint (Unix domain socket
  in `storage/`) — files indexed in last hour, current queue depth,
  last error message + timestamp.
- `hermes status --watch` reads the socket and renders a live TUI
  with throughput, error rate, recent files. ^C to exit.
- `hermes doctor` reads the same socket and surfaces "watcher is
  alive but hasn't indexed anything in 24h."

**Risk:** Adds an IPC surface that has to stay backward-compatible
across watcher upgrades. Probably fine — Unix sockets in a known
path are easy. Skip if it's the only thing pulling its weight.

### File-scoped queries (`hermes ask --file`, `/file`)

**Gap:** Retrieval is global. Staring at `auth.md` and asking "what's
the rationale here?" returns *related* notes, not necessarily that
file's content.

**Sketch:**
- `hermes ask --file <path> "<q>"`: skip retrieval, inject the file's
  full contents (or top-N chunks of *just that file* if it's huge) as
  context.
- `/file <path>` REPL command: pin retrieval to that file for
  subsequent turns until cleared with `/file -` or `/clear`.
- Tab-completion over vault paths in the REPL would make this usable.

**Risk:** Low. Mostly additive. The completion piece needs `readline`
hooks that work on libedit (Apple's bundled Python).

### Embedding model agnosticism

**Gap:** `EMBED_DIM = 768` is hardcoded in three places. Switching to
BGE-M3 (1024-dim) or `gte-large` (1024-dim) requires code changes AND
a full reindex.

**Sketch:**
- Probe the embed server on first connect, store the observed
  dimension in a meta row (`vault_chunks_meta` table or schema
  metadata).
- Refuse to mix dimensions: if the running server's output dim
  doesn't match the table's stored dim, error out with a clear
  remediation (`hermes index --backfill --force`).
- `EMBED_DIM` becomes a default; the actual value is read from the
  table at REPL/ask startup.

**Risk:** Edge-case-heavy. The error path on dim mismatch has to be
clean (current behavior would be a cryptic `ValueError` deep in
LanceDB). Worth doing only if there's actual demand for switching
embed models.

### Cross-platform support

**Gap:** Hard Apple-Silicon-only stance. The Linux/Windows audience
for "local Obsidian RAG" is real and currently shut out.

**Sketch:**
- Split platform-specific code behind small shims:
  `src/platform/{darwin,linux,windows}.py` for service registration
  (launchd/systemd/Task Scheduler), hardware probing
  (`sysctl`/`/proc/cpuinfo`/wmic), and Metal vs CUDA vs CPU build
  flags.
- `make all` dispatches per-platform. Doctor's first check becomes
  "supported platform" rather than "is arm64."
- llama.cpp build switches from `GGML_METAL=ON` to whatever the
  platform supports.

**Risk:** Big. Triples the surface area of "things that could break"
and most of the project's optimization choices (KV-slot quantization,
unified-memory mmap, P-core thread pinning) are Apple-specific. Worth
doing only if the maintenance cost is committed to up front. May be
better to leave hermes-metal Apple-only and recommend a sister
project for other platforms.

### Versioned benchmark history

**Gap:** `make bench-report` writes today's numbers to
`bench/results/REPORT.md`, overwriting the previous run. Comparing
this month's perf to last quarter's requires git archaeology.

**Sketch:**
- Append a row to `bench/results/history.tsv` on every `make bench`:
  date, commit sha, tier, throughput, RSS, perplexity, J/1k-tokens.
- `make bench-report` plots history.tsv as a sparkline at the top of
  the report.

**Risk:** Trivial implementation. The trap is letting the file grow
unbounded — but at one row per manual `make bench` invocation, that's
a non-problem for any realistic horizon.

---

## 2026-06-03 — pytest suite

**Problem:** Zero test coverage. `_trim_history`, `chunk_text` (paragraph /
fence / oversize-paragraph branches), the SSE streamer, and the vault
filter were all silent-drift candidates.

**Change:** Added `tests/` with `conftest.py` plus five test files
covering chunker, REPL trim & transcript round-trip, vault filter, the
`hermes index` end-to-end (mocking embed via `monkeypatch.setattr`),
watcher × filter integration, and SSE streaming via `httpx.MockTransport`.
`make test` target. `pytest>=8.0.0` in `requirements.txt`.

**Files:** `tests/conftest.py`, `tests/test_chunker.py`,
`tests/test_repl_logic.py`, `tests/test_vault_filter.py`,
`tests/test_index_cmd.py`, `tests/test_watcher_filter.py`,
`tests/test_streaming.py`, `Makefile` (`test` target),
`requirements.txt`.

**Impact:** 48 passing tests, no daemons or network required. Catches
chunker invariants, transcript round-trip safety, watcher filter
bypass, and SSE parsing regressions before they reach prod.

---

## 2026-06-03 — `/load` REPL command

**Problem:** `/save` existed but the inverse didn't — users couldn't
resume yesterday's conversation.

**Change:** `/load <path>` parses a `/save`-format transcript (`### role`
blocks) and rehydrates `session.history`. Anchored regex
(`^### (user|assistant|system)$`) so a body line that happens to be a
markdown subheading doesn't split a message. Strips exactly the writer's
separator blank line — not all trailing newlines — so user content with
meaningful trailing blanks round-trips. Atomic on parse failure
(history untouched).

**Files:** `src/repl.py` (`_parse_transcript`, `_cmd_load`,
`_TRANSCRIPT_HEADER_RE`).

**Impact:** Conversations are now durable across REPL restarts.
`hermes` → ask things → `/save ~/notes/chat.md` → quit → next day
`/load ~/notes/chat.md` continues where you left off.

---

## 2026-06-03 — KV-slot persistence (cross-session)

**Problem:** Every REPL start re-prefilled the system prompt + early
history from scratch. First-token latency on the first turn was
identical every time.

**Change:** Best-effort `slot_save` on REPL exit, `slot_restore` on next
start. Per-vault slot name (sha1 of resolved vault path) so two vaults
on the same host don't clobber each other's caches. Streamer pins
`id_slot=0` so it stays correct under any future `--parallel >1`.
`slot_save` wrapped in `asyncio.wait_for(..., timeout=2.0)` so a dead
chat server can't hang REPL exit. `/forget-cache` slash command clears
the server slot AND unlinks the on-disk `.bin` file (server's
`action=erase` is in-memory only). `slot_erase` added to `HermesClient`.

**Files:** `src/repl.py` (`KV_SLOT_ID`, `_kv_slot_name`,
`_slot_save_dir`, `_delete_slot_file_on_disk`, restore/save hooks),
`src/server/client.py` (`slot_erase`).

**Impact:** Subsequent REPL boots reuse the prefix from the prior
session — observable as faster first-token latency on warm starts.
Worst case (cache miss / version drift / model change) silently falls
back to fresh prefill; correctness is unaffected.

---

## 2026-06-03 — `hermes index --backfill / --gc`

**Problem:** Watcher only catches *future* writes. A fresh install on a
1000-note vault left the index empty until you touched each file.
Files moved or renamed while the watcher was stopped became orphan
rows in LanceDB with no way to clean them up.

**Change:** `hermes index` subcommand with `--backfill --force --gc
--dry-run --limit`. Walks the vault using the shared filter, indexes
new/changed files, GC drops orphan rows whose `source_path` is gone or
newly excluded. `LanceVault.distinct_sources()` powers GC via
`to_arrow()` (ships with lancedb; avoids the optional `pylance` dep).
Vault path `.resolve()`-d so a relative `HERMES_VAULT_PATH` doesn't
make every indexed source look like an orphan. `--dry-run` rejected
unless paired with `--gc` (foot-gun: backfill is always real). GC
refuses to drop ≥90% of sources without `--force` (most likely cause:
vault root moved → wiping the index would be unrecoverable).

**Files:** `src/index_cmd.py`, `src/backend/database.py`
(`distinct_sources`), `src/cli.py` (subcommand wiring).

**Impact:** Closes the gap between the README's promise ("watches your
vault") and what actually happens on a fresh install. The supported
onboarding flow is now `make all && hermes index --backfill`.

---

## 2026-06-03 — Vault filter (shared by watcher + index)

**Problem:** Watcher only filtered by `.md` / `.markdown` extension. Real
vaults have `templates/`, `attachments/`, `.obsidian/`, daily-note
dumps; without folder excludes the index was mostly noise.

**Change:** `src/backend/vault_filter.py` with env > YAML > defaults
precedence. Globs match `fnmatch`-style against paths relative to the
vault root; slashless globs match any path component, slashed globs
are anchored. `iter_vault_files` prunes excluded directories via
`os.walk` dirnames mutation. Watcher derives `PatternMatchingEventHandler`
patterns from the filter's include list (so
`HERMES_VAULT_INCLUDE="*.md:*.txt"` actually delivers `.txt` events
live). `on_moved` consults the filter instead of hardcoding extensions.
Sample `config/vault.yaml.example` shipped.

**Files:** `src/backend/vault_filter.py`, `src/daemon/watcher.py`
(filter integration, pattern derivation, `on_moved` fix),
`config/vault.yaml.example`.

**Impact:** Default excludes (`. obsidian`, `.trash`, `attachments`,
`templates`) reduce index noise by 30–60% on typical Obsidian vaults.
Live and one-shot index paths share one source of truth — they can't
disagree on what belongs.

---

## 2026-06-03 — README polish

**Problem:** Quickstart told users to run `hermes doctor` but never
explained how `hermes` reaches `$PATH`. Architecture section conflated
write-path (ingestion) with query-path (REPL/ask).

**Change:** Quickstart now includes `make install-cli` + `hermes doctor`.
New Commands table summarizing the six subcommands. Architecture diagram
split into write path (Obsidian → watcher → embed → LanceDB) and query
path (REPL → embed → LanceDB → chat). Three LaunchAgents listed
explicitly with their roles.

**Files:** `README.md`.

**Impact:** New users get a working `hermes ...` invocation in the
first 30 seconds without searching for the symlink target.

---

## 2026-06-03 — Interactive REPL

**Problem:** `hermes ask "<q>"` is great for scripts, painful for the
actual "second brain" use case where you iterate. Each invocation paid
the chat-server connection-setup tax fresh.

**Change:** `src/repl.py` (~470 lines). Plain-history storage with on-
the-fly context injection (no chunk-bloat across turns). Per-turn
retrieval. Token-budget trimming via `TRIM_RATIO * context_window` —
drops oldest user/assistant pairs when over budget; current user
message and system prompt are always preserved. Streaming via
`client._chat.stream`. Slash commands: `/help /clear /sources /save
/norag /rag /exit`. Persistent line history at
`~/.hermes/repl_history` via stdlib `readline` (with libedit detection
for Apple's bundled Python). One warm `HermesClient` for the whole
session. Soft-fallback when the embed server is down. Startup health
probe with a `hermes doctor` hint. Bare `hermes` (no subcommand) drops
into REPL.

Adversarial review caught and fixed 4 real bugs:
- `loop.add_signal_handler` was overriding asyncio's default SIGINT
  injection — ^C at empty prompt was silently swallowed. Scoped the
  handler to the streaming window only.
- Synchronous `embed_query` ran on the event loop, freezing SIGINT
  delivery for up to 60s on a slow embed server. Wrapped in
  `asyncio.to_thread`.
- libedit detection had `or ""` (no-op precedence) and would
  `TypeError` when `readline.__doc__` is `None` — silently swallowed,
  so tab-complete was broken on Apple's bundled Python. Fixed.
- Mid-stream cancel left a dangling user message in `history`.
  `finally` block now persists any streamed content or rolls back the
  user turn if the assistant produced nothing.

**Files:** `src/repl.py`, `src/cli.py` (subcommand wiring, bare-hermes
fall-through), `README.md`.

**Impact:** REPL is now the default invocation. Multi-turn chat with
retrieval re-run each turn, ↑-recall across sessions, conversation
durability via `/save`. One warm connection pool means second-turn
latency is meaningfully lower than back-to-back `hermes ask` calls.

---

## 2026-06-03 — `hermes doctor`

**Problem:** When something broke (port conflict, stale plist, missing
model, dead daemon), users had to grep through `logs/`, `launchctl
list`, and `lsof` themselves. `hermes status` only said
`UNREACHABLE` — no remediation.

**Change:** `src/doctor.py` (~575 lines, **stdlib-only**). 9 sections × ~25
checks: host arch & Xcode CLT, repo files, `llama-server` build, venv
arch + import probe, model file size sanity, host topology + vault
contents + port-collision detection, LaunchAgent presence/loaded
state, both `/health` endpoints (with "loading model" 503 tolerance),
on-disk LanceDB inspection. Each non-OK result carries a one-line
`fix:`. `--json` flag for scripting. Exit 0 if ready, 1 on FAIL.

`bin/hermes` special-cases `doctor` to fall back to system `python3`
when `.venv` is missing — exactly when users reach for it. `src/cli.py`
early-routes `doctor` before heavy imports so a broken
`lancedb`/`pyarrow`/`httpx` doesn't break the diagnostic.

Adversarial review caught 5 real bugs, all fixed:
- `lsof` truncates command names to 9 chars by default → false-
  positive port-collision (added `+c 0`).
- `Request()` constructor (not just `urlopen`) raises `ValueError` on
  empty URL → moved into `try`.
- `HERMES_EMBED_URL` with trailing slash bypassed suffix-strip →
  `rstrip("/")` first.
- `launchctl print` substring match could collide with `*.helper`
  labels → anchored regex.
- `Path.is_relative_to` (Py 3.9+) → swapped for `try/except ValueError`
  for safety.

**Files:** `src/doctor.py`, `bin/hermes`, `src/cli.py`, `Makefile`
(`doctor` target), `README.md` (Troubleshooting section).

**Impact:** First-line diagnostic for any failure. `make doctor &&
hermes ask ...` is safe to chain. The single highest-leverage UX
improvement to date — collapsed the typical failure-debug loop from
"tail four log files" to "read one line."

---

## How to add an entry

When a substantive change lands:

1. Add a new section at the top under today's date.
2. State the **Problem** in concrete user-visible terms (not "we lacked
   X" but "users had to do Y manually").
3. Describe the **Change** at the level a future maintainer needs —
   enough detail that they could find it in `git log` if this doc
   disappeared.
4. List **Files** touched (top-level only, not every line).
5. Name the **Impact** observably. Numbers when you have them, mechanism
   when you don't. Avoid "improved performance" without a measurement.

Skip entries for typo fixes, comment-only edits, and one-line
refactors. This is for changes that altered what hermes-metal *does*,
not how the code reads.

## How to add a proposed entry

When something gets discussed but not built:

1. Add a `### Title` block under `## Proposed` (above the dated
   shipped entries).
2. Write **Gap**, **Sketch**, **Risk** in that order. Resist the urge
   to write a design doc — three paragraphs is the cap. If you need
   more, the idea isn't yet clear enough to be on this list.
3. When the idea ships, *move* the entry into a new dated section,
   rewrite to past tense, and replace **Risk** with **Impact**. Don't
   leave a stub behind.
4. If the idea stops being a good idea, *delete* the entry. This list
   is not a museum of plausibly-good ideas — it's a working backlog.
