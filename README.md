<img width="577" height="433" alt="Gemini_Generated_Image_vv8jqpvv8jqpvv8j-removebg-preview" src="https://github.com/user-attachments/assets/7cbe30b5-06ba-46ba-b526-94cb2db820ad" />

# hermes-metal

A zero-lag, background "second brain" for Apple Silicon. It watches a Markdown
vault, embeds it locally, serves an 8B model over a local OpenAI-compatible
API, and answers questions about your notes — entirely on-device, under macOS
QoS so foreground apps stay snappy.

Beyond plain retrieval, hermes maintains an **LLM-authored wiki** between your
raw notes and your queries. Sources you ingest become summary pages;
explorations you `/file` become topic pages; daily **digests** and archived
**conversations** accumulate too. The watcher indexes all of it, so future
queries draw on compounding synthesis instead of rediscovering everything each
time.

```
write path (background, debounced):
  Obsidian vault ─► watcher ─► nomic-embed (:8081) ─► LanceDB

query path (per turn):
  ask / repl ─► temporal scope ─► nomic-embed (:8081) ─► LanceDB top-N
                                                            │
                                              rerank ◄──────┘
                                                │
                                hermes-8b (:8080) ─► streamed answer
```

Three always-on LaunchAgents (`engine` = chat on `:8080`, `embed` on `:8081`,
`watcher`) run under `ProcessType=Background` + `Nice=15`. An optional fourth
(`digest`) fires once a day.

## Requirements

- Apple Silicon (M1–M4), macOS 13+
- Python 3.11+, Xcode Command Line Tools
- ~10 GB disk (models + build + index)

## Quickstart

```sh
git clone --recursive https://github.com/jaeHbk/hermes-metal
cd hermes-metal

export HERMES_VAULT_PATH=~/Documents/Obsidian   # where your notes live

make all          # arch check → build llama.cpp → venv → fetch models →
                  # install 3 LaunchAgents. First run: 5–15 min (models ~5 GB).
make install-cli  # symlink `hermes` onto PATH (needs ~/.local/bin on PATH)

hermes doctor                 # confirm everything's healthy
hermes index --backfill       # index notes that existed before install
```

Endpoints once running: chat `http://127.0.0.1:8080/v1/chat/completions`,
embeddings `http://127.0.0.1:8081/v1/embeddings`. Stop with `make stop-daemon`;
remove agents with `make uninstall`.

## Commands

| Command | Purpose |
| --- | --- |
| `hermes` | Interactive REPL (default). Multi-turn, streaming, fresh retrieval per turn. |
| `hermes ask "<q>"` | One-shot RAG question; streams a cited answer. |
| `hermes search "<q>"` | Retrieval only; prints matched chunks + scores. |
| `hermes digest` | Build a daily activity digest; file it in the wiki, optionally push it. |
| `hermes wiki init` / `status` | Bootstrap / inspect the LLM-authored wiki. |
| `hermes ingest <path>` | Summarize a raw source into `wiki/sources/`. |
| `hermes lint` | Wiki health-check: orphans, stubs, stale, unused sources. |
| `hermes index` | Backfill / GC / migrate the index (see below). |
| `hermes notify` | Send a Telegram message; `--setup` configures, `--check` validates. |
| `hermes status` / `doctor` | Health probe / end-to-end self-diagnostic with `fix:` hints. |

## Talking to your vault

Bare `hermes` drops into the REPL — multi-turn chat with retrieval re-run each
turn and streamed output:

```
$ hermes
hermes>  what did I write yesterday about the auth rewrite?
  ↳ scoped to yesterday
  ↳ retrieved 5 chunk(s): 2026-06-06-daily.md #chunk2  d=0.41 ...
The auth-rewrite notes from yesterday flag two open questions [2026-06-06-daily.md]: ...
hermes>  remind me which one I said was blocking
```

History is kept across turns; retrieval re-runs against your latest question.
Ctrl-C cancels a generation (Ctrl-C at an empty prompt, or Ctrl-D, exits).
↑ recalls past prompts. Slash commands worth knowing:

- `/file <name>` — promote the last answer into `wiki/topics/<name>.md`.
- `/wiki` — page counts. `/sources` — last turn's retrievals.
- `/save <path>` / `/load <path>` — round-trippable transcripts.
- `/norag` / `/rag` — toggle vault retrieval. `/clear` — wipe history.
- `/forget-cache` — drop the persisted KV slot.

### Retrieval: temporal scoping + reranking

Retrieval is date-aware and reranked. A high-precision date phrase
(`yesterday`, `today`, `last week`, `past 7 days`, `this month`, `2026-06-01`,
`2026-06`) scopes the search to that window and prints `↳ scoped to …`;
anything ambiguous falls through to an unscoped search, and an empty window
widens back automatically. Candidates are then reranked by a blend of semantic
similarity, recency (30-day half-life), and heading/term overlap before the
top-k reaches the model. All of this degrades gracefully on an un-migrated
index.

## The wiki: compounding synthesis

The wiki is a structured `<vault>/wiki/` subtree the LLM owns; you keep owning
your raw notes. Because it lives under your vault, the watcher indexes it — so
synthesis becomes retrievable the moment it's written.

```
<vault>/wiki/
├── .hermes-agents.md   ← schema you edit; the LLM reads it into its prompt
├── index.md            ← catalog (LLM-maintained)   log.md ← audit trail
├── sources/            ← one summary per ingested source
├── topics/             ← concept pages built from /file
├── digests/            ← daily summaries (Phase D)
└── conversations/      ← archived REPL sessions (Phase E)
```

```sh
hermes wiki init                 # idempotent bootstrap
hermes ingest path/to/article.md # summarize a source → wiki/sources/<stem>.md
hermes lint                      # health report (read-only; --strict for CI)
```

In the REPL, `/file quant-comparison` files the last answer as a topic page;
the next query about quantization pulls in that synthesis, not just raw notes.
Ingest and `/file` never overwrite hand-written files, and update `index.md` /
`log.md` atomically with rollback.

## Daily digest

`hermes digest` summarizes a day's vault activity into a durable wiki page at
`wiki/digests/YYYY-MM-DD.md`:

- **Activity** — which notes changed (mechanical).
- **Learnings** — an LLM synthesis (degrades to mechanical-only if the chat
  server is down).
- **Practice questions** — only when the day's notes are class material
  (`#class/*` tag or a `class/`-like path).
- **Open questions** — TODOs, unchecked `- [ ]` boxes, and `?`-lines.

```sh
hermes digest --dry-run          # preview yesterday's digest, no writes
hermes digest --date 2026-06-06  # a specific day
make install-digest-daemon       # schedule it (default 07:30 daily)
```

It's idempotent (the wiki page is the state) and **never pushes off-device by
default**. To deliver digests to Telegram, opt in explicitly:

```sh
hermes notify --setup                          # one-time bot setup
make install-digest-daemon DIGEST_PUSH=1       # enable daily push
```

`hermes doctor` warns whenever push is on, since it sends summarized vault
content to Telegram daily.

## Conversation memory

Substantial REPL sessions can be archived to `wiki/conversations/` on exit, so
past chats become retrievable and citable by future digests. It's opt-in:

```sh
export HERMES_REPL_ARCHIVE=1                  # archive sessions on exit
hermes index --gc-chats --older-than 90       # prune old archives
```

## Indexing

The watcher catches **future** writes. For an existing vault, or after files
moved while the daemon was stopped:

```sh
hermes index --backfill          # embed anything missing
hermes index --backfill --force  # re-embed everything (e.g. after a model change)
hermes index --gc                # drop rows whose source file is gone
hermes index --migrate           # upgrade an old index to the metadata schema
```

`--migrate` adds the `mtime`/`tags`/`heading_trail` columns in place (instant,
lossless) and re-embeds only the rows whose metadata is still placeholder.
`hermes doctor` flags a pre-migration index with the exact fix. GC refuses to
drop ≥90% of sources without `--force` (usually means the vault moved).

**Vault filter.** Indexes `*.md` / `*.markdown`; excludes `.obsidian/`,
`.trash/`, `attachments/`, `templates/`. Override per-session with
`HERMES_VAULT_INCLUDE` / `HERMES_VAULT_EXCLUDE` (colon-separated globs), or
commit `config/vault.yaml` (see the `.example`). The watcher and `hermes index`
share one filter, so live and one-shot indexing always agree.

## Configuration

Env-driven; defaults work.

| Variable | Default | Notes |
| --- | --- | --- |
| `HERMES_VAULT_PATH` | `~/Documents/Obsidian` | |
| `HERMES_LANCEDB_PATH` | `<repo>/storage/lancedb` | |
| `HERMES_CHAT_URL` | `http://127.0.0.1:8080` | |
| `HERMES_EMBED_URL` | `http://127.0.0.1:8081/v1/embeddings` | |
| `HERMES_DIGEST_PUSH` | `0` | `1` + a configured bot enables daily Telegram push |
| `HERMES_REPL_ARCHIVE` | `0` | `1` archives substantial REPL sessions to the wiki |

Hardware tier (context window, threads) is auto-detected via `sysctl` into
`config/host_topology.env`. See `CLAUDE.md` for the full spec.

## Troubleshooting

```sh
hermes doctor          # or: make doctor
```

Walks the install end-to-end — host arch, build, venv, models, agents, ports,
both `/health` endpoints, LanceDB schema version, wiki, notifications, digest —
and prints a one-line `fix:` for anything that isn't OK. Stdlib-only and falls
back to system `python3` when the venv is broken. Exit 0 if ready, 1 on FAIL;
`--json` for machine output.

## Tests

```sh
make test     # pytest tests/ -v — pure-Python, no daemons or network
```

206 tests cover the load-bearing pieces: chunker, schema migration, reranker,
temporal parser, digest, conversation archive, vault filter, wiki, REPL trim,
and SSE streaming (mocked via `httpx.MockTransport`).

## Benchmarks

Head-to-head vs MLX 4-bit on the same Hermes-3-8B (`temperature=0`, `seed=42`).
See `bench/README.md`.

| Axis | hermes (llama.cpp Q4_K_M) | MLX 4-bit | Winner |
| --- | --- | --- | --- |
| Decode tok/s (medium) | 19.3 | **28.8** | MLX +49% |
| Prefill tok/s (long) | 320.1 | **357.3** | MLX +12% |
| **Peak RSS** (long) | **298 MiB** | 1,806 MiB | **llama.cpp 6×** |
| **Perplexity** (WikiText-2) | **8.40** | 11.72 | **llama.cpp** |

MLX wins raw throughput; llama.cpp wins memory by 6× and perplexity by a wide
margin at the same 4 bpw. For an always-on background daemon, memory and
quality are the right axes — which is also why the reranker is a heuristic, not
another model to keep resident.

## Changelog

Substantive changes — problem, change, impact — are tracked newest-first in
[`IMPROVEMENTS.md`](IMPROVEMENTS.md).

## License

MIT. See `LICENSE`.
