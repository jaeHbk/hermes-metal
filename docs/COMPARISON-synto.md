# hermes-metal vs. synto — an honest comparison

> Issue: **hm-3c6**. Compares this project against
> [`kytmanov/synto`](https://github.com/kytmanov/synto) and records what we
> chose to adopt, what we deferred, and why — measured against hermes-metal's
> core value: **minimal RAM, zero-lag background operation on-device.**

Both projects start from the same Karpathy-flavored premise — *treat your
notes as source material, not the final artifact, and let a local LLM build
something on top* — and then make almost opposite engineering bets. That makes
the comparison genuinely useful rather than a homer exercise: each is better at
a different thing.

---

## 1. What each project is

| | **hermes-metal** | **synto** |
| --- | --- | --- |
| One-liner | Always-on background "second brain" that watches a vault, embeds it, and answers questions with cited RAG. | A pipeline that compiles raw notes into a cross-linked, self-improving wiki of articles. |
| Mental model | **Retrieval engine.** Your notes stay primary; the LLM answers *about* them on demand. | **Compiler.** Your notes are input; the LLM *writes new artifacts* (wiki articles) as output. |
| Primary loop | edit note → watcher embeds → ask → retrieve top-k → stream cited answer | `add` source → ingest (small model) → compile articles (big model) → review → publish |
| Interaction | REPL / `ask` (live chat), plus an LLM-authored wiki that compounds. | CLI batch pipeline + `query` + interactive `review` TUI + MCP server. |

They overlap in the middle (both maintain an LLM-authored wiki with
`[[wikilinks]]`, both ingest sources, both lint/maintain), but the **center of
gravity differs**: hermes optimizes *answering*, synto optimizes *compiling*.

---

## 2. Feature-by-feature

| Axis | hermes-metal | synto | Honest verdict |
| --- | --- | --- | --- |
| **Retrieval** | Vector (LanceDB, nomic-embed 768-d) + heuristic rerank (semantic+recency+lexical) + temporal scoping. | **BM25 over source segments**, index-routed via `INDEX.json`. *No embeddings, no vector DB.* | **Split.** hermes wins on paraphrase/synonym recall and recency-aware ranking. synto wins on zero-model simplicity and exact-term/rare-token queries, and runs retrieval on *any* machine with no model loaded. |
| **Memory / RAM** | Vector index on disk; embed server (~0.3 GB) + chat 8B (Q4_K_M ~4.7 GB) resident as background LaunchAgents (`Nice=15`). Heuristic reranker is deliberately *not* a model. Peak RSS ~298 MiB for the engine in bench. | Works on 16 GB with 4B + 14B; **no resident model at query time** (Ollama loads on demand); no vector index in RAM at all. | **synto's steady-state floor is lower** — its query path needs no resident model. hermes pays RAM for an always-warm, zero-latency answer. Different trade: hermes = instant + heavier; synto = lean + cold-start. |
| **Two-tier models** | Single 8B for everything; reranker is a heuristic, not a model. | **4B for analysis, 14B for article-writing** — cheap pattern-matching vs. expensive synthesis. | synto's split is genuinely clever for a *compile* workload. hermes doesn't have a compile workload heavy enough to justify a second resident model — it would *raise* steady-state RAM, which we will not do. |
| **Ingestion** | `ingest <file>` / `--url` / `ingest-links <file>` (batch URL list, auto-indexed, failures spooled for retry). | `add` imports PDF/md/txt with heading-aware chunking; **source-type prompts** (`paper`, `textbook`, `api_docs`, …) tune analysis. | synto's **source-type-aware prompts** and PDF support are a real edge. hermes is markdown/URL-first (matches its Obsidian thesis) and has no PDF path. |
| **Wiki** | `sources/ topics/ digests/ conversations/`, LLM-managed, `hermes-managed` frontmatter, atomic index/log, hand-edit refusal. | `draft → verified → published` lifecycle, **SHA-256 hand-edit protection**, confidence scores, rejection-feedback loop, concept identity (rename/merge/split + `undo`). | synto's wiki lifecycle is **more mature**: explicit verification states, reversible curation, confidence-gated approval. hermes' wiki is simpler but tightly integrated with live retrieval (the watcher indexes the wiki, so synthesis is instantly queryable). |
| **Incremental work** | Watcher embeds only changed files; `index --backfill/--gc/--migrate`. | **Incremental compile** — changing one note recompiles only its derived articles; LLM response cache. | Comparable in spirit; synto's compile-lineage (`trace`) and response cache are nicer for a batch compiler. hermes' watcher is better for *live* ingestion. |
| **UX** | REPL with streaming, slash commands, temporal phrasing, daily digest, Telegram opt-in. | `review` TUI, `compare` (A/B models), `eval` (coverage/citation metrics), `doctor --backlog`, git auto-commit `[synto]`, `undo`. | synto has more **operator tooling** (eval, compare, undo, trace). hermes has a better **conversational** surface (streaming multi-turn REPL, digests). |
| **Agent surface** | OpenAI-compatible local HTTP (chat :8080 / embed :8081). | **MCP server** (`serve`) with 12 tools + license-gated verbatim source access. | synto is **more agent-native** today (MCP, quality signals for self-filtering). hermes exposes raw model endpoints, not a curated tool surface. |
| **Platform** | **Apple Silicon-specific** (Metal llama.cpp, launchd, `sysctl` tiering). Deeply optimized, not portable. | Cross-platform Python; 20+ OpenAI-compatible providers incl. cloud. | synto is **far more portable**. hermes trades portability for a hardware-native, battery-friendly background daemon — that's the whole point of "metal." |
| **Privacy** | On-device by default; digest push to Telegram is opt-in and doctor-flagged. | Local by default; cloud providers optional; warns when verbatim gate defaults open. | Both local-first and honest about their escape hatches. Tie. |

---

## 3. Verdict — which is better, and for whom

There is no single winner; they serve different jobs.

- **Choose hermes-metal** if you want an **always-on, zero-latency answer
  machine** glued to one machine you control, where memory footprint and
  battery matter more than raw throughput, and you live in a live edit→ask
  loop. Its vector retrieval is better when you don't remember the exact
  words you wrote, and its background-QoS design means it never makes your
  laptop stutter.

- **Choose synto** if you want to **compile a durable, reviewed wiki** out of
  heterogeneous sources (PDFs, papers, specs), value an explicit
  draft→verified→published workflow with reversible curation, need it to run
  **anywhere** (including cloud models / non-Apple hardware), or want a
  **first-class MCP tool surface** for other agents to consume.

**Where synto is honestly better than hermes today:** (1) lower steady-state
RAM floor on the query path — no resident model; (2) source-type-aware
ingestion + PDF support; (3) a more mature wiki lifecycle (verification
states, confidence scores, `undo`/`trace`, hand-edit SHA protection);
(4) portability and provider breadth; (5) a built-in MCP server and eval
tooling.

**Where hermes is better:** (1) live, streaming, multi-turn conversation with
fresh per-turn retrieval; (2) semantic recall that survives paraphrase;
(3) recency- and date-aware ranking; (4) a genuinely background, battery-aware
daemon tuned to one architecture; (5) 6× lower engine RSS and far better
perplexity than MLX at the same 4 bpw (see `bench/`).

---

## 4. What we adopted

### ✅ BM25 lexical search (`hermes search --lexical`)

synto's defining bet — *"no embeddings, no vector database"* — is the single
most RAM-virtuous idea it has, and it slots into hermes as a **pure-upside,
opt-in fallback** rather than a replacement.

**What it is.** A new `src/backend/lexical.py` implements standard Okapi BM25
(k1=1.5, b=0.75, +0.5/+1 smoothed IDF) over the chunk text *already stored in
LanceDB*. `hermes search --lexical` builds a transient index from that text,
ranks, and returns the top-k — then drops the index. `LanceVault.all_chunks()`
projects just the text + locator columns (never the 768-float vector) so the
read stays cheap.

**Why it passes the RAM bar (the key judgement).**

- **Zero steady-state cost.** There is *no persisted lexical index* and *no
  resident model*. The default three-LaunchAgent footprint
  (`ProcessType=Background`, `Nice=15`) is byte-for-byte unchanged. The BM25
  index exists only for the microseconds of a single query and is garbage-
  collected immediately.
- **Opt-in, off by default.** Vector search remains the default `hermes
  search` path. `--lexical` is something the user reaches for deliberately.
- **Transient peak is tiny.** For a personal vault (low thousands of chunks)
  the token lists + DF map are a few MB of short-lived Python objects — far
  below the embed model it replaces for that query.

**Why it's worth having (the upside).**

- **Works with the embed server down — or absent.** This directly serves the
  CPU-host constraint: on a box where the Metal llama/embed servers don't run,
  vector search is impossible but `--lexical` still answers from indexed text.
  The vector path's error now points users at it.
- **Better for exact-term queries** — error codes, file names, identifiers,
  quoted phrases — where literal match beats semantic nearness.

**Memory impact: none at steady state; a few MB transient per query.**
Tests: `tests/test_lexical.py` (14), `tests/test_cli_lexical.py` (4), plus
`all_chunks()` coverage in `tests/test_database_migration.py` (2). All
pure-Python — no daemon, no network — so they pass on a CPU host.

---

## 5. What we deliberately did NOT adopt, and why

| Candidate from synto | Why deferred |
| --- | --- |
| **Two-tier 4B + 14B models** | **Deferred (RAM).** hermes has no compile workload heavy enough to justify a *second resident model*. Adding a 14B writer would multiply steady-state RAM — a direct violation of the core value. hermes' synthesis (ingest, digest, `/file`) is occasional and the existing 8B handles it. If a heavy compile mode is ever wanted, it must be a spawn-on-demand subprocess that exits, never a resident agent. |
| **Persisted BM25 / FTS5 index** | **Deferred (RAM/disk + redundancy).** A standing inverted index would add disk and, if mmapped/cached hot, steady-state memory. The transient-index approach gets ~all the benefit at zero resident cost. Revisit only if vault size makes per-query rebuild slow (it isn't, at personal scale). |
| **Heavy wiki lifecycle** (draft→verified→published, confidence scores, rejection-feedback, concept merge/split/undo) | **Deferred (scope, not RAM).** Genuinely good, but it's a large behavioral surface that reshapes hermes' simpler "the wiki page is the state" model. Worth a dedicated design pass, not a drive-by. No RAM objection — purely scope/risk. |
| **SHA-256 hand-edit protection** | **Partially already covered / deferred (scope).** hermes already refuses to overwrite hand-written files via `hermes-managed` frontmatter (`wiki.is_managed`). A content-hash guard is stricter but additive; reasonable future work, not required now. |
| **Source-type-aware ingest prompts** (`paper`, `textbook`, `api_docs`, …) | **Deferred (scope).** Attractive and low-RAM (just prompt variants). A good candidate for a *future* additive feature; left out of this push to keep the change focused on the headline RAM-positive adoption. |
| **MCP server / `eval` / `compare` / `undo` / `trace`** | **Deferred (scope).** Operator tooling that doesn't conflict with the RAM ethos but is a separate body of work. MCP in particular is a promising future direction for agent consumption. |
| **PDF ingestion** | **Deferred (dependency weight).** Robust PDF parsing pulls a heavier dependency; hermes' thesis is markdown/URL. Low priority unless users need it. |

**Bottom line:** we adopted the one feature that is *strictly* RAM-positive and
opt-in (BM25 lexical fallback), and explicitly declined everything that would
raise steady-state memory (notably the two-tier resident models). hermes-metal
stays true to its identity: zero always-resident footprint increase by default.
