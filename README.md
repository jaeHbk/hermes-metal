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
export HERMES_VAULT_PATH=~/Documents/Obsidian   # your vault root
make all
```

`make all` runs: `check-env` → `build-engine` (CMake llama.cpp w/ Metal) →
`setup-venv` → fetch chat + embed models → install three LaunchAgents
(engine, embed, watcher) and bootstrap them.

Endpoints once running:
- Chat: `http://127.0.0.1:8080/v1/chat/completions` (OpenAI-compatible)
- Embeddings: `http://127.0.0.1:8081/v1/embeddings`

Stop with `make stop-daemon`; remove agents with `make uninstall`.

## Architecture

```
Obsidian Vault ──► watchdog daemon ──► nomic-embed (127.0.0.1:8081)
                                          │
                                          ▼
                                       LanceDB ──► hermes-8b (127.0.0.1:8080)
```

1. Edit Markdown in your vault.
2. Watcher daemon catches modify events, chunks the doc.
3. Chunks embedded via local `llama-server --embedding`, written to LanceDB.
4. Chat client retrieves context, queries the chat `llama-server`.

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

## Benchmarks

Head-to-head against MLX 4-bit on the same Hermes-3-8B — measures
throughput, peak RSS, WikiText-2 perplexity, and joules-per-1k-tokens via
`powermetrics`. See `bench/README.md`.

```sh
make bench              # throughput + perplexity (no sudo)
make bench-power        # adds power telemetry (sudo)
make bench-report       # write bench/results/REPORT.md
```

## License

MIT. See `LICENSE`.
