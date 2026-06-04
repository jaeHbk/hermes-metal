# hermes-metal

A zero-lag, background "second brain" for Apple Silicon. Watches a Markdown
vault, embeds it locally with `nomic-embed-text-v1.5`, indexes it in LanceDB,
and serves Hermes-3-8B (Q4_K_M, GGUF) over a local OpenAI-compatible chat API
via `llama.cpp` + Metal — all under macOS QoS so foreground apps stay snappy.

## Requirements

- Apple Silicon (M1/M2/M3/M4), macOS 13+
- Python 3.11 / 3.12
- Xcode Command Line Tools
- ~10 GB disk (model + build + index)

## Quickstart

```sh
git clone --recursive https://github.com/jaeHbk/hermes-metal
cd hermes-metal

# 1. Tell hermes where your vault is. (No vault yet? `mkdir -p
#    ~/Documents/Obsidian && echo "# hello" > ~/Documents/Obsidian/Welcome.md`)
export HERMES_VAULT_PATH=~/Documents/Obsidian

# 2. Full install: arch check → build llama.cpp → venv → fetch models
#    → install + boot 3 LaunchAgents (engine, embed, watcher).
#    First run takes 5–15 min depending on network (models are ~5 GB).
make all

# 3. Make `hermes` available in your shell. ~/.local/bin must be on PATH;
#    override with PREFIX=/usr/local/bin if you prefer system-wide.
make install-cli

# 4. Confirm everything's healthy. Should print
#    "hermes-metal: ready — N ok, 0 warn, 0 fail" and exit 0.
hermes doctor

# 5. (Existing vault only.) The watcher only catches future writes — for
#    notes that already existed before install, run a one-shot backfill:
hermes index --backfill
```

Endpoints once running:
- Chat: `http://127.0.0.1:8080/v1/chat/completions` (OpenAI-compatible)
- Embeddings: `http://127.0.0.1:8081/v1/embeddings`

Stop with `make stop-daemon`; remove agents with `make uninstall`.

## First run — sanity test

A 5-step verification you can do in 60 seconds end-to-end. Each step has
a clear pass condition; if any one fails, `hermes doctor` is the next
move.

**1. Doctor is green.**

```sh
hermes doctor
```

Expect every section OK except possibly Notifications (SKIP if you
haven't run `--setup` yet — that's fine, notifications are optional).
Bottom line should say `hermes-metal: ready`.

**2. Watcher catches a live write.**

```sh
echo "# Smoke test\n\nThe capital of France is Paris." \
  > "$HERMES_VAULT_PATH/_smoke.md"
sleep 8                       # debounce window is 5s
hermes status | grep rows     # row count should be > 0
```

If `rows` is 0, the watcher didn't fire — check
`logs/watcher.log` and rerun `hermes doctor`.

**3. Retrieval finds it.**

```sh
hermes search "capital of France"
```

Should print `_smoke.md #chunk0` near the top.

**4. End-to-end RAG answer.**

```sh
hermes ask "What's the capital of France according to my notes?"
```

Should stream an answer that cites `[_smoke.md]`.

**5. Interactive REPL.**

```sh
hermes
# At the prompt, ask the same question, then ask a follow-up like
# "remind me what we just discussed" — multi-turn context should hold.
# Ctrl-D or /exit when done.
```

Cleanup:

```sh
rm "$HERMES_VAULT_PATH/_smoke.md"
sleep 8
hermes index --gc           # drops the smoke-test row from the index
```

If all five pass, you're ready to test Phase A (Telegram notifications).

## Commands

| Command                | Purpose                                                    |
| ---------------------- | ---------------------------------------------------------- |
| `hermes`               | Drop into the interactive REPL (default).                  |
| `hermes ask "<q>"`     | One-shot RAG question; streams the answer.                 |
| `hermes search "<q>"`  | Retrieval only; prints matched chunks and their sources.   |
| `hermes status`        | Probe both `/health` endpoints and the index size.         |
| `hermes doctor`        | End-to-end self-diagnostic with one-line `fix:` hints.     |
| `hermes index`         | Backfill / GC the vault index (one-shot; see below).       |
| `hermes notify`        | Send a Telegram message; `--setup` configures, `--check` validates. |
| `hermes repl`          | Same as bare `hermes`, with `-k`, `--max-tokens`, `--no-rag`. |

## Talking to your vault

Bare `hermes` (no subcommand) drops into an interactive REPL — multi-turn
chat, fresh retrieval each turn, streaming output:

```
$ hermes
hermes-metal REPL — local second-brain chat.
  /help              show commands     /clear   reset conversation
  /sources           show last hits    /save P  save transcript to P
  /norag             disable RAG       /rag     re-enable RAG
  Ctrl-C             cancel generation / exit at empty prompt
  Ctrl-D             exit

hermes>  what did I write yesterday about the auth rewrite?
  ↳ retrieved 5 chunk(s):
     - 2026-06-02-daily.md #chunk2  d=0.412
     - design/auth.md #chunk0  d=0.487
     ...
The auth rewrite notes from yesterday flag two open questions [2026-06-02-daily.md]: ...
hermes>  remind me which one I said was blocking
...
```

History is kept across turns; retrieval is re-run each turn against your
latest question (no stale-context bloat). Ctrl-C aborts the current
generation but keeps you in the REPL; Ctrl-C at an empty prompt (or Ctrl-D)
exits. Up-arrow recalls past prompts via `~/.hermes/repl_history`.

For one-shot scripted use, `hermes ask "<question>"` still works.

REPL slash commands worth knowing:

- `/save <path>` / `/load <path>` — round-trippable transcripts (markdown).
- `/sources` — show retrievals from the last turn.
- `/clear` — wipe conversation; retrieval still works on the next turn.
- `/norag` / `/rag` — toggle vault retrieval on the fly.
- `/forget-cache` — drop the persisted KV cache (the REPL silently
  saves it to slot 0 on exit and restores on next start; see
  `storage/slots/`).

## Notifications (Telegram) — Phase A test guide

Optional, but this is the **wire for the upcoming daily-summary feature**
(Phase D). Setting it up now means you can verify Phase A end-to-end and
the daily-summary phases will plug straight in.

### Setup (one time)

**Step 1 — Create a Telegram bot.**

Open Telegram (phone or web), search for [`@BotFather`](https://t.me/BotFather),
start a chat, send `/newbot`, follow the prompts. BotFather gives you:

- A bot username like `@my_hermes_bot`.
- A token like `1234567890:ABCDef-ghIJklMnOPqrsTUvwxYZ`.

**Step 2 — Run the setup wizard.**

```sh
hermes notify --setup
```

It will:

1. Prompt for the BotFather token. Paste it.
2. Validate the token via `getMe` (you'll see `✓ token OK — bot is @your_bot`).
3. Ask you to DM the bot. Open Telegram → search for your bot's username
   → tap **Start** (or just send any message).
4. Capture your chat_id via long-polling `getUpdates` (`✓ chat_id captured`).
5. Write `~/.hermes/telegram.json` (mode 0600).
6. Send a confirmation message — **check your phone, you should see
   "hermes-metal: notifications are wired up."**

If step 6 doesn't land in your Telegram client, setup didn't work. See
troubleshooting below.

### Verify Phase A

Three quick checks. All three should pass before you consider Phase A
done.

**A. Doctor reports the bot healthy.**

```sh
hermes doctor
```

The Notifications section should now show:

```
Notifications (Telegram)
  [OK  ] telegram token  bot @your_bot_name
  [OK  ] telegram chat   chat_id=12345 (Your Name)
```

**B. Manual send round-trips.**

```sh
hermes notify "ping from $(hostname) at $(date)"
```

Message should arrive in your Telegram chat within a second or two.

**C. `--check` exits 0.**

```sh
hermes notify --check && echo OK
```

Should print `hermes notify: OK — bot @your_bot ...` followed by `OK`.

### What this proves

- Bot token works against Telegram's API (`getMe`).
- Bot can reach your chat (`getChat`).
- Long messages chunk and deliver correctly (`sendMessage`).
- `~/.hermes/telegram.json` was written with secure perms (`stat -f %p
  ~/.hermes/telegram.json` should end in `600`).

If all three checks pass, **Phase A is verified** and Phase B/C/D can
start building on top.

### Troubleshooting

- **Setup hangs on "Waiting for your message".** You haven't DM'd the
  bot yet, or you sent the message before pressing **Start** so Telegram
  didn't deliver it. Open the bot's profile, tap **Start**, then send
  any message. Setup polls for 60s before timing out.
- **Doctor's Notifications section says SKIP after setup.** Config
  wasn't written — check that `~/.hermes/telegram.json` exists. If you
  ran setup as a different user, the file is in that user's home dir.
- **`hermes notify "..."` returns "transport error".** Network issue
  reaching `api.telegram.org`. Try `curl -sS https://api.telegram.org`
  to isolate. Some corporate networks block Telegram.
- **Bot was working, now `--check` returns 1 with "Unauthorized".**
  Token was revoked (most likely you regenerated it via BotFather).
  Run `hermes notify --setup` again to re-capture.
- **Negative chat_id (Telegram supergroup).** Supported and stored as a
  string. If you want notifications going to a group chat instead of a
  DM, add the bot to the group, send any message there, and re-run
  `--setup` — the wizard captures whichever chat sees the message first.

### Config file format

```json
{
  "bot_token": "1234567890:ABCDef-ghIJklMnOPqrsTUvwxYZ",
  "chat_id": "987654321"
}
```

Env vars (`HERMES_TELEGRAM_BOT_TOKEN`, `HERMES_TELEGRAM_CHAT_ID`)
override the file per-key — useful for one-off testing without
rewriting the config.

## Indexing

The watcher daemon catches **future** writes. For an existing vault, or to
clean up after files were moved while the daemon was stopped:

```sh
hermes index --backfill           # walk the vault, embed anything missing
hermes index --backfill --force   # re-embed everything (e.g. after upgrading the embed model)
hermes index --gc --dry-run       # show orphan rows that would be removed
hermes index --gc                 # actually remove them
hermes index --backfill --gc      # both, in order
```

Hot tip: a fresh install on an existing vault should be `make all && hermes
index --backfill` — `make all` brings the daemons up but the index starts
empty, so the watcher would otherwise need every file touched once.

GC refuses to drop ≥90% of sources without `--force` — the most common
cause of "everything looks orphan" is moving the vault root, and wiping
the whole index in that case is recoverable only by re-running backfill.

### Vault filter

By default the index includes `*.md` and `*.markdown`, and excludes
`.obsidian/`, `.trash/`, `attachments/`, and `templates/`. Override per-
session with env vars (colon-separated globs, replacing — not merging —
the defaults):

```sh
export HERMES_VAULT_INCLUDE="*.md:*.txt"
export HERMES_VAULT_EXCLUDE=".obsidian:.git:Archive/*"
```

Or commit a permanent override at `config/vault.yaml` (see
`config/vault.yaml.example`). The watcher and `hermes index` both
consume the same filter, so the live and one-shot paths always agree.

## Troubleshooting

Stuck? Run the self-diagnostic:

```sh
hermes doctor          # or: make doctor
```

It walks the install end-to-end (host arch, Xcode CLT, `llama.cpp` build,
`.venv`, models, agents, ports, both `/health` endpoints, LanceDB on disk)
and prints a one-line `fix:` for everything that isn't OK. It exits `0` if
ready, `1` on FAIL — safe to chain in scripts:

```sh
make doctor && hermes ask "what did I write yesterday?"
```

Doctor is stdlib-only and falls back to `/usr/bin/env python3` when the venv
is broken — exactly when you need it. Use `--json` for machine-readable
output.

## Architecture

```
write path (background, debounced):
  Obsidian Vault ─► watchdog daemon ─► nomic-embed (:8081) ─► LanceDB

query path (per turn):
  hermes repl / ask ─► nomic-embed (:8081) ─► LanceDB top-k
                                                      │
                                                      ▼
                                       hermes-8b (:8080) ─► streamed tokens
```

Three cooperating LaunchAgents:

- **`com.hermes.metal.engine`** — `llama-server` with Hermes-3-8B (Q4_K_M)
  on `:8080`. Chat + completions, KV-slot persistence, OpenAI-compatible.
- **`com.hermes.metal.embed`** — `llama-server --embedding` with
  `nomic-embed-text-v1.5` on `:8081`. Mean-pooled, 768-dim.
- **`com.hermes.metal.watcher`** — Python `watchdog` daemon that debounces
  Obsidian save bursts, hashes content (skip no-op writes), chunks the
  Markdown, and upserts into LanceDB.

All three run under `ProcessType=Background` + `Nice=15` so foreground apps
keep the CPU. KeepAlive respawns crashed agents; `make uninstall` is the
clean stop.

## Configuration

Env-driven. Defaults work; override as needed.

| Variable              | Default                              |
| --------------------- | ------------------------------------ |
| `HERMES_VAULT_PATH`   | `~/Documents/Obsidian`               |
| `HERMES_LANCEDB_PATH` | `<repo>/storage/lancedb`             |
| `HERMES_CHAT_URL`     | `http://127.0.0.1:8080`              |
| `HERMES_EMBED_URL`    | `http://127.0.0.1:8081/v1/embeddings`|

Hardware tier (context window, threads) is auto-detected via `sysctl` and
written to `config/host_topology.env`. See `CLAUDE.md` for the full spec.

## Tests

Pure-Python coverage of the load-bearing pieces (chunker, REPL trim &
transcript parser, vault filter, index command, watcher × filter
integration, SSE streaming):

```sh
make test          # runs `pytest tests/ -v` in the project venv
```

No daemons or network required — `tests/test_streaming.py` mocks
`llama-server` via `httpx.MockTransport`.

## Benchmarks

Head-to-head against MLX 4-bit on the same Hermes-3-Llama-3.1-8B — same
weights, same prompts, `temperature=0`, `seed=42`. See `bench/README.md`
for the harness.

```sh
make bench              # throughput + perplexity (no sudo)
sudo make bench-power   # adds powermetrics telemetry
make bench-report       # write bench/results/REPORT.md
```

Measured on M3 Pro (36 GiB, 6P+6E, macOS 26.4):

| Axis | hermes-metal (llama.cpp Q4_K_M) | MLX 4-bit | Winner |
| --- | --- | --- | --- |
| Decode tok/s (medium prompt)   | 19.3   | **28.8**   | MLX +49% |
| Prefill tok/s (long prompt)    | 320.1  | **357.3**  | MLX +12% |
| **Peak RSS** (long prompt)     | **298 MiB**  | 1,806 MiB  | **llama.cpp 6×** |
| **Perplexity** (WikiText-2)    | **8.40**     | 11.72      | **llama.cpp Δ −3.32** |

The honest framing: MLX wins raw throughput; **llama.cpp wins memory by
6× and perplexity by a wide margin** at the same nominal 4 bpw. Q4_K_M's
K-quants super-blocks preserve quality far better than MLX's group-wise
affine quantization, and llama.cpp's `mmap`'d weights + `q8_0` KV cache
keep the always-on daemon's footprint dramatically smaller. For an
always-on background "second brain," memory and quality are the right
axes to win on.

(RSS caveat: llama.cpp `mmap`s the GGUF, so shared file-backed pages are
counted differently from MLX's MLX-buffer allocations. The structural
gap is still large — q8_0 KV cache alone is the biggest lever.)

## Changelog

Substantive changes are tracked in [`IMPROVEMENTS.md`](IMPROVEMENTS.md) —
problem, change, impact for each one, newest first.

## License

MIT. See `LICENSE`.
