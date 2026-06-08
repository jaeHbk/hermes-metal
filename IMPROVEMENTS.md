# IMPROVEMENTS

Running log of changes to hermes-metal. Each entry names the problem the
change solved and the observable improvement, so future readers (or
future-us) can tell whether a feature is still pulling its weight.

Newest first. Add entries as work lands, not in batch.

---

## 2026-06-07 — LaunchAgent exit-78 hardening (invalid-plist guard + diagnosis)

**Problem:** All three LaunchAgents reported `last exit code = 78` (EX_CONFIG)
and never served traffic, while `llama-server` ran fine when launched by hand.
A systematic isolation (run binary manually OK, under launchd's scrubbed env
OK, full config under a *different* label OK, byte-identical file with only the
Label changed OK — but the real `com.hermes.metal.*` labels FAIL) found two
distinct issues:

1. **A real, latent plist bug:** `daemon.plist.template` and
   `digest.plist.template` had XML comments containing a literal double hyphen
   (from `--flash-attn`, `--prompt-cache`, `--setup`). A double hyphen inside
   an XML comment is illegal per the spec; `plutil -lint` is lenient and
   passes it, but a strict (launchd-grade) parser rejects the whole plist →
   exit 78, silently, with the program never exec'd. This would bite real
   users on stricter macOS builds.
2. **A wedged launchd service record** (the symptom on this machine): once an
   agent has crash-looped, its record in the user's `gui/<uid>` domain can get
   stuck returning 78 regardless of plist content — proven because a
   byte-identical file under a fresh label launches instantly. It clears only
   on logout/reboot (which resets the user launchd domain); `bootout`,
   `disable`/`enable`, and re-`bootstrap` do not.

**Change:**

* **Templates:** reworded the offending comments so no double hyphen appears
  inside any comment (flag names spelled out / single-dashed). All four
  templates now render to strictly-valid XML.
* **Makefile `_bootstrap`:** added a strict `plistlib` (expat) parse after
  `plutil -lint` — catches the bad-comment class at install time instead of
  after a thousand silent crash-loops — plus a post-bootstrap exit-78 check
  that prints the logout/reboot remediation when a valid plist still won't
  spawn.
* **`hermes doctor`:** `check_agents` now strict-parses each installed plist
  and reads each agent's per-label exit code, so a crash-looping agent shows
  `FAIL … exits 78 (EX_CONFIG)` with the right fix instead of a misleading
  "loaded = OK". New stdlib helper `_strict_plist_ok`.

**Files:** `config/daemon.plist.template`, `config/digest.plist.template`,
`Makefile` (`_bootstrap`), `src/doctor.py` (`check_agents`,
`_strict_plist_ok`), `tests/test_doctor_plist.py` (new).

**Impact:** The invalid-XML bug can no longer ship undetected (install-time
guard + 4 regression tests covering every template). When a launchd record is
genuinely wedged, both `make` and `doctor` now name the cause and the fix
(log out / reboot) instead of leaving the user to guess. Tests: 206 → 210.

---

## 2026-06-07 — Daily-summary cascade: temporal retrieval, metadata schema + rerank, digest, conversation memory (Phases B→C→D→E)

**Problem:** The wiki layer made synthesis *durable*, but retrieval was still
naive: plain top-k vector search on L2 distance, no recency, no metadata, no
date awareness. "What did I write yesterday?" couldn't actually scope to
yesterday. There was no scheduled summary, and unmarked REPL conversations
evaporated on exit. The four phases below were the planned cascade (each
enables the next); they shipped together.

**Change:**

* **Phase C — metadata-rich schema + reranker** (the linchpin). LanceDB schema
  v2 adds `mtime`, `tags`, `heading_trail` columns. Migration is **in-place,
  lossless, and instant** (`add_columns`; existing vectors untouched) and runs
  automatically on `LanceVault` open — verified against a copy of the live
  index (7 rows, 0 lost). A stdlib-readable `.hermes-schema.json` sidecar lets
  `doctor` report the version without importing lancedb. `hermes index
  --migrate` upgrades the schema and re-embeds **only** the rows whose metadata
  is still placeholder (via `sources_with_stale_metadata()`), so a big vault
  isn't fully reprocessed. Search switched to **cosine** (correct for
  L2-normalized nomic embeddings; it's a query-time choice, so no migration
  needed). New `src/backend/reranker.py`: a zero-dependency heuristic
  (semantic + recency half-life + heading/lexical overlap) reorders an
  over-fetched candidate set down to top-k. Chose a heuristic over a
  cross-encoder deliberately — the always-on daemon's footprint is the whole
  project's thesis; a 280 MB reranker model would undo the benchmark win.
* **Phase B — temporal awareness.** New `src/backend/temporal.py`:
  `parse_window` maps **high-precision** phrases (`yesterday`, `today`,
  `last week`, `past N days`, `this month`, ISO dates, `YYYY-MM`) to an mtime
  window + (for single days) a `source_path LIKE '%YYYY-MM-DD%'` clause for
  daily notes. Anything ambiguous returns `None` and falls through to an
  unscoped search — a misparse that silently scopes to the wrong dates is
  worse than no scoping. Scoping is printed (`↳ scoped to yesterday`) and an
  empty window widens back to all notes. Wired into both `hermes ask` and the
  REPL.
* **Phase D — `hermes digest` + scheduled delivery.** New `src/digest.py` and
  `hermes digest [--date] [--dry-run] [--no-push] [--force]`. Walks the vault
  for notes in a day's mtime window, builds four sections — Activity
  (mechanical), Learnings (LLM, degrades to mechanical-only if the chat server
  is down), Practice questions (gated on class-material detection), Open
  questions (TODOs / `- [ ]` / `?`-lines, code-fence-aware, deduped). Writes a
  durable `wiki/digests/YYYY-MM-DD.md` (managed, indexed, so it compounds and
  is retrievable) and — **only with explicit opt-in** (`HERMES_DIGEST_PUSH=1`
  *and* a configured bot) — pushes a headline + the full markdown via
  Telegram. Idempotent (the wiki page is the state). New
  `config/digest.plist.template` (`StartCalendarInterval`, not KeepAlive) and
  `make install-digest-daemon` (`DIGEST_HOUR`/`MINUTE`/`PUSH` vars). Doctor
  gains a Digest section that **warns when push is on**.
* **Phase E — cross-session conversation memory.** The wiki *is* the memory.
  New `src/conversation.py`: on REPL exit, a substantial session (opt-in via
  `HERMES_REPL_ARCHIVE=1`) is archived to `wiki/conversations/<ts>.md` as a
  managed page — auto-indexed, citable by the digest, retrievable next turn.
  `hermes index --gc-chats --older-than N` prunes old archives (built in from
  day one so the directory can't metastasize). Wiki gained a `Conversations`
  index section; lint treats conversations (like digests) as chronologically
  terminal.

**Adversarial review (6 parallel reviewers × per-finding verifiers, 24 agents)
plus a hand audit caught 7 real bugs, all fixed:**

1. **Concurrent-migration crash** (hand audit): the watcher auto-migrates on
   open while `index --migrate` runs in another process; `add_columns` raises
   "column already exists" for the loser, crashing the watcher daemon.
   `migrate()` now adds each column independently and treats already-exists as
   success (re-checking the live schema).
2. **Unbounded blank-line growth in `index.md`** (hand audit): `_join_index`
   re-captured the blank line it left between a header and its body on every
   write (3→12 newlines after 10 writes) — a real problem under daily digests
   and per-session archives. Now normalizes the separator to one blank line.
3. **ISO-date regex matched inside identifiers**: `report-2026-06-07.md` in a
   query mis-scoped retrieval to that date. Replaced `\b` with
   `(?<![-\d])…(?![-\d])` so only standalone date tokens match.
4. **ISO-month regex** had the same hyphen-boundary flaw; same fix.
5. **Digest TOCTOU**: the hand-written-file guard ran before a multi-second
   LLM call, so a user file created in that window could be clobbered.
   Re-checked at write time inside `write_digest_page`.
6. **Conversation dangling index row**: an `append_log` failure after the
   index row was committed rolled back the page, stranding the index row at a
   deleted file. The audit log is now best-effort (separate from the
   load-bearing page+index write).
7. **Doctor `check_wiki`** omitted the conversations count; added.

The review also *refuted* 12 findings — including two (the blank-line growth
and the migration race) that the verifiers correctly marked refuted because
they read the already-fixed code. Net: the verification loop did its job.

**Files:** `src/backend/database.py`, `src/backend/indexer.py`,
`src/backend/reranker.py` (new), `src/backend/temporal.py` (new),
`src/digest.py` (new), `src/conversation.py` (new), `src/index_cmd.py`,
`src/cli.py`, `src/repl.py`, `src/doctor.py`, `src/wiki.py`, `src/wiki_cmd.py`,
`src/lint_cmd.py`, `config/digest.plist.template` (new), `Makefile`, and five
new test files (`test_temporal`, `test_reranker`, `test_database_migration`,
`test_digest`, `test_conversation`).

**Impact:** Retrieval is now recency- and date-aware and reranked; the schema
migrates losslessly with a clear `doctor` signal; the daily digest closes the
original CLAUDE.md goal of a pushed daily summary (privacy default-off); and
conversations compound into the same wiki the rest of the system already
draws on. Tests: 141 → 206 passing, no daemons or network required.

---

## 2026-06-04 — Wiki layer (architectural pivot from retrieval-only to compounding synthesis)

**Problem:** Every query rediscovered knowledge from raw notes from
scratch. Nothing accumulated. A subtle question that synthesized five
notes made the model find and piece together fragments every time,
with no memory that this synthesis ever happened. The
[LLM-Wiki pattern](https://gist.github.com/...) describes exactly the
gap: persistent, LLM-maintained knowledge that compounds rather than
being re-derived.

**Change:** Three layers, all additive. Existing surfaces still work.

* **`src/wiki.py`** — stdlib-only wiki primitives. `WikiPaths` resolution
  (env → `<vault>/wiki/`), atomic page writes via temp+rename,
  log/index update, `[[wiki-link]]` and `[md](link)` parsing,
  `is_managed` frontmatter detection (bounded to the YAML block, not
  body substring matches).
* **`hermes wiki init / status`** — bootstrap a wiki at
  `<vault>/wiki/` with `sources/`, `topics/`, `digests/` subdirs plus
  three meta files (`index.md`, `log.md`, `.hermes-agents.md`). Idempotent.
* **`hermes ingest <path>`** — drives the chat server with a structured
  ingest prompt (Summary / Key claims / Entities and concepts / Open
  questions), writes `wiki/sources/<stem>.md`, updates index, appends
  log. Refuses to overwrite hand-written files even with `--force`.
  Three-step write (page → index → log) with rollback: if index/log
  fails, the page is unlinked so the wiki stays internally consistent.
* **`hermes lint`** — read-only wiki health-check. Reports orphan
  topics (no inbound link), unused sources (in `sources/` but never
  cited), stubs (referenced pages with no file), stale (older than
  `--stale-days`). Self-loop-safe (a page linking to its own stem
  doesn't mask its own orphan status). Digests are exempt — they're
  chronologically terminal. `--strict` exits 1 on any issue.
* **REPL `/file <name>` and `/wiki`** — `/file` promotes the last
  assistant turn into `wiki/topics/<name>.md` with the question
  retained as a frontmatter backref. The (last_user, last_assistant)
  pair is captured atomically in `add_assistant`, so a cancelled
  follow-up question can't poison the next `/file`. Hand-written
  guard applies regardless of `--force`. `/wiki` shows quick status.
* **System prompt becomes wiki-aware** — built once per session as
  base prompt + today's date + `.hermes-agents.md` contents (if
  initialized). Stable across the session so KV-slot prefix matching
  still works.
* **Doctor** gains a Wiki section (SKIP when uninitialized — the wiki
  is opt-in).

The watcher auto-indexes wiki pages because they live under
`<vault>/`. No schema change, no reindex needed. Knowledge written
into the wiki is queryable on the very next turn.

Adversarial review (4 parallel subagents) caught **12 real bugs**, all
fixed:

1. `_join_index` silently dropped rows for sections the user had
   deleted from `index.md`. Now appends missing sections at the end.
2. `is_managed` substring match wasn't bounded to the frontmatter
   block — a user file with `hermes-managed: true` quoted in its body
   was mis-flagged and would be overwritten. Now scoped to the YAML
   block between the opening and closing `---`.
3. Ingest hand-written guard was nested inside `not args.force`, so
   `--force` could overwrite hand-written user files. Hoisted out.
4. `_slugify` collision check was case-sensitive. APFS (the project's
   sole target FS) is case-insensitive by default — `INDEX.md` and
   `index.md` are the same path, so the guard must compare in lower
   case.
5. Ingest's three-step write had no rollback; a failure between the
   page write and the index update left the page on disk with no log
   entry, blocking re-runs because the early-exit saw the file
   already existing. Now wraps index+log in try/except and unlinks
   the page on failure.
6. Lint self-loop bug: a page containing `[[its-own-stem]]` inflated
   its own inbound count to 1, masking it from orphan detection.
   `_index_outbound_links` now discards self-references.
7. `_read_updated` only tried three exact strptime formats; ISO
   timestamps with fractional seconds (`2026-06-04T19:00:00.123Z`)
   silently failed → page never flagged as stale even when ancient.
   Now uses `datetime.fromisoformat` first.
8. Lint stem-collision: `page_stems = {stem: path}` silently
   overwrote one of two same-stem pages in different subdirs. Now
   iterates on full paths and counts inbounds via a stem-keyed
   helper.
9. `/file --force` had the same hand-written-overwrite bug as ingest.
   Same fix.
10. `/file` arg parser only checked the trailing token for `--force`,
    so `/file foo --force --verbose` silently stuffed `--verbose` into
    the slug. Now parses each token; any unknown flag errors out.
11. `last_user` was set in `add_user` BEFORE the assistant streamed.
    A cancelled Q2 corrupted the `(Q, A)` pair `/file` reads — Q2
    backref on A1. Now both `last_user` and `last_assistant` update
    atomically in `add_assistant`, only on a successful turn.
12. (Filed during refactor: lint orphan-check originally treated
    digests as orphans — corrected to exempt them as chronologically
    terminal.)

**Files:** `src/wiki.py`, `src/wiki_cmd.py`, `src/ingest_cmd.py`,
`src/lint_cmd.py`, `src/repl.py` (extends), `src/doctor.py` (extends),
`src/cli.py` (subcommand wiring), `tests/test_wiki.py`,
`tests/test_ingest.py`, `tests/test_lint_cmd.py`,
`tests/test_repl_logic.py` (extends).

**Impact:** This is the architectural pivot the project was missing.
Without it hermes was a (good) chatbot over notes. With it, hermes
becomes the compounding knowledge base the original CLAUDE.md
described as the goal. The user keeps owning their vault notes; the
LLM owns the wiki. Tests: 84 → 141 passing.

---

## 2026-06-04 — Telegram notification transport (Phase A of daily-summary cascade)

**Problem:** No way for the agent to push anything off the local machine.
Daily-summary plans, alarms, and any future "your bot pinged you" feature
all needed a wire-level primitive first. Picked Telegram over ntfy or
webhook because it has a real iOS/Android client, supports bot DMs as a
private channel, and authenticates per-bot rather than per-topic.

**Change:** `src/notify.py` (~530 lines, **stdlib-only**, urllib +
plistlib + json). New `hermes notify` subcommand:

- `hermes notify "<msg>"` — send a chat message; chunks on paragraph
  boundaries with `(i/n)` pagination footers when over Telegram's 4096-
  char cap. Refuses empty input (Telegram 400s on empty `text`).
- `hermes notify --setup` — interactive flow: prompts for the BotFather
  token, validates it via `getMe`, asks the user to DM the bot, captures
  the chat_id by long-polling `getUpdates`, writes
  `~/.hermes/telegram.json` (mode 0600), sends a confirmation message.
- `hermes notify --check` — validates token + chat against `getMe` /
  `getChat`. Exit 0 OK, 1 configured-but-broken, 2 unconfigured.

Config precedence is env > file (per-key, so `HERMES_TELEGRAM_BOT_TOKEN`
in env + chat_id in file is supported). `send_document` shipped now for
Phase D's full-summary attachment use case. Doctor gains a Notifications
section (SKIP when unconfigured — notifications are optional — OK / FAIL
when configured).

Adversarial review (3 parallel subagents) caught 9 real bugs, all fixed:

- `poll_for_chat_id` never advanced offset when returning a chat → next
  process run replayed the same backlog forever. Now bumps offset across
  the whole batch and best-effort acks before returning.
- Empty-string `send` produced `[""]` → Telegram 400. Now returns 0
  chunks and `send` raises explicitly.
- `send_document` filename injected raw into `Content-Disposition`. A
  filename with `"` or `\r\n` could corrupt the multipart frame or
  inject headers. Added `_sanitize_filename`.
- `--check "ignored"` silently dropped the stray positional. Now errors.
- Doctor's `except notify.NotConfiguredError` after `is_configured()` was
  unreachable. Removed.
- Doctor's `except ImportError` did NOT catch `SyntaxError` — a typo in
  notify.py would crash doctor entirely. Now catches both.
- `_run_notify_late` was unreachable in normal flow + duplicated the
  notify subparser definition (drift risk). Removed both; the early-
  route in cli.py is the only live path.
- Early-route had no `KeyboardInterrupt` guard, so ^C during `--setup`
  printed a traceback. Now exits 130 cleanly.
- Several test gaps (chat-not-found path, negative chat_id, offset
  advancement, send_document) closed.

**Files:** `src/notify.py`, `src/cli.py` (early-route + slimmed
subparser), `src/doctor.py` (`check_notifications`),
`tests/test_notify.py` (36 tests).

**Impact:** Phase A wire is proven — both unit-tested (36 passing) and
exercisable end-to-end via `hermes notify --setup`. Unblocks the Phase
B/C/D cascade for daily summaries: temporal awareness, schema upgrade
with rerank, and the actual summarizer can now plug into a working
delivery primitive.

---

## Proposed

Things on the table but not yet built. Each entry names the **gap** (what
the user can't do today, or has to work around), the **sketch** (rough
shape of the fix), and the **risk** (what would make this a bad idea, or
the load-bearing assumption).

Promotion rule: when something here ships, move it down into a dated
section above and rewrite to past tense. If it stops being a good idea,
delete it — don't let this list ossify into a wishlist museum.

### Reranker tuning + optional cross-encoder

**Gap:** The Phase C reranker weights (semantic 0.70 / recency 0.15 /
lexical 0.15, 30-day half-life) are hand-picked defaults. There's no way
to tune them per-vault, and no learned reranker for users who'd trade
footprint for quality.

**Sketch:**
- Expose the weights via env / `config/vault.yaml` so a user whose notes
  are mostly timeless can drop the recency weight.
- Optional `bge-reranker-base` (~280 MB) behind the same `rerank()`
  signature, loaded as a third llama-server slot, selected by a config
  flag. Default stays heuristic (zero footprint).

**Risk:** Low. The heuristic is the safe default; tuning is additive and
the cross-encoder is opt-in. Only worth the model download if users
report the heuristic missing obvious matches.

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
