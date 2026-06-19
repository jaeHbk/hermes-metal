#!/usr/bin/env bash
# Manually start the hermes-metal chat + embed servers WITHOUT launchd.
#
# Use this when the LaunchAgents are wedged (exit 78 / EX_CONFIG) and you don't
# want to log out / reboot yet. It mirrors the ProgramArguments baked into
# ~/Library/LaunchAgents/com.hermes.metal.{engine,embed}.plist (PRO tier:
# ctx 32768, 8 threads, flash-attn, q8_0 KV cache).
#
# Idempotent: if a port is already serving, it leaves that server alone.
# Logs go to logs/{engine,embed}.manual.log. Stop them with:
#   pkill -f 'llama-server .*hermes-8b'      # engine
#   pkill -f 'llama-server .*nomic-embed'    # embed
#
# NOTE: these are plain background processes — they do NOT auto-restart and the
# watcher daemon is still not running (so live vault auto-indexing is off).
# The permanent fix is: log out / reboot, then `make start-daemon`.

set -euo pipefail

# Resolve the repo root from this script's location, so it works from anywhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SERVER="$REPO_ROOT/third_party/llama.cpp/build/bin/llama-server"
CHAT_MODEL="$REPO_ROOT/models/hermes-8b-q4_k_m.gguf"
EMBED_MODEL="$REPO_ROOT/models/nomic-embed-text-v1.5.f16.gguf"
SLOT_DIR="$REPO_ROOT/storage/slots"
CTX=32768
THREADS=8

mkdir -p logs "$SLOT_DIR"

# Preflight: the binary and models must exist (else `make build-engine` /
# `make fetch-models` was never run).
for f in "$SERVER" "$CHAT_MODEL" "$EMBED_MODEL"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing required file: $f" >&2
        echo "       run \`make all\` to build the engine and fetch models." >&2
        exit 1
    fi
done

# True if something is already LISTENing on the given TCP port.
port_up() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

# Wait up to ~30s for a port to come up; return 1 on timeout.
wait_for_port() {
    local port="$1" tries=0
    while ! port_up "$port"; do
        tries=$((tries + 1))
        [ "$tries" -gt 60 ] && return 1
        sleep 0.5
    done
    return 0
}

start_embed() {
    if port_up 8081; then
        echo "embed  (:8081) already up — leaving it alone."
        return 0
    fi
    echo "embed  (:8081) starting..."
    nohup "$SERVER" -m "$EMBED_MODEL" --host 127.0.0.1 --port 8081 \
        --embedding --pooling mean -c 8192 -t "$THREADS" \
        > logs/embed.manual.log 2>&1 &
    if wait_for_port 8081; then
        echo "embed  (:8081) up (pid $!)."
    else
        echo "embed  (:8081) FAILED to come up — see logs/embed.manual.log" >&2
        return 1
    fi
}

start_engine() {
    if port_up 8080; then
        echo "engine (:8080) already up — leaving it alone."
        return 0
    fi
    echo "engine (:8080) starting (loads a 4.6 GB model — give it ~15-20s)..."
    nohup "$SERVER" -m "$CHAT_MODEL" --host 127.0.0.1 --port 8080 \
        -c "$CTX" -t "$THREADS" -fa on -ctk q8_0 -ctv q8_0 \
        --slot-save-path "$SLOT_DIR" \
        > logs/engine.manual.log 2>&1 &
    if wait_for_port 8080; then
        echo "engine (:8080) up (pid $!)."
    else
        echo "engine (:8080) FAILED to come up — see logs/engine.manual.log" >&2
        return 1
    fi
}

start_embed
start_engine

echo
echo "Servers ready. Verify with:  hermes doctor"
echo "Now you can run, e.g.:        hermes ingest-links ~/reading.txt"
