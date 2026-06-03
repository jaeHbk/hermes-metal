#!/usr/bin/env bash
# fetch_embed_model.sh — download the canonical nomic-embed-text-v1.5 GGUF
# (f16, native 768-dim, 8192 ctx) into ./models/.
#
# Behavior:
#   - Sources config/engine_flags.env for $EMBED_MODEL_URL and $EMBED_MODEL_FILE.
#   - Skips download if models/$EMBED_MODEL_FILE already exists, unless --force
#     is passed as the first argument.
#   - Sanity-checks the resulting file is > 200 MB (the canonical
#     nomic-embed-text-v1.5.f16.gguf is ~262 MiB).
#
# Run from anywhere; the script cd's to the repo root before doing work.
#
# This is the embedding model used by src/backend/indexer.py via a second
# llama-server instance (axiom-aligned: same llama.cpp binary as the chat
# server, second process on port 8081, native ARM Metal). Per the Nomic
# model card, every input must be prefixed with a task instruction
# ("search_document: " for indexing, "search_query: " for queries) — that
# prefixing happens in the Python client, not here.

set -euo pipefail

# Resolve repo root = parent of the directory holding this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ENV_FILE="config/engine_flags.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "fetch_embed_model.sh: missing ${ENV_FILE}; run \`make check-env\` first." >&2
    exit 1
fi

# shellcheck disable=SC1090
set -a
source "${ENV_FILE}"
set +a

# Defaults if engine_flags.env is silent on these — pin to the canonical
# Nomic-published GGUF, f16 variant.
: "${EMBED_MODEL_URL:=https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf}"
: "${EMBED_MODEL_FILE:=nomic-embed-text-v1.5.f16.gguf}"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

mkdir -p models
DEST="models/${EMBED_MODEL_FILE}"

if [[ -f "${DEST}" && "${FORCE}" -eq 0 ]]; then
    echo "fetch_embed_model.sh: ${DEST} already exists; pass --force to overwrite."
else
    echo "fetch_embed_model.sh: downloading ${EMBED_MODEL_URL}"
    echo "                   -> ${DEST}"
    # --fail   : exit non-zero on HTTP >=400 instead of writing an error page.
    # -L       : follow redirects (HF resolve URLs redirect to a CDN).
    # --progress-bar : human-friendly bar; replaces the noisy default meter.
    curl -L --fail --progress-bar "${EMBED_MODEL_URL}" -o "${DEST}.part"
    mv "${DEST}.part" "${DEST}"
fi

# Sanity-check size. Use stat in BSD (macOS) form first, GNU as fallback.
if SIZE=$(stat -f%z "${DEST}" 2>/dev/null); then
    :
else
    SIZE=$(stat -c%s "${DEST}")
fi

MIN_BYTES=200000000  # 200 MB; canonical f16 is ~262 MiB / ~274 MB.
if (( SIZE < MIN_BYTES )); then
    echo "fetch_embed_model.sh: ${DEST} is only ${SIZE} bytes (< ${MIN_BYTES})." >&2
    echo "                   Likely a truncated download or an HTML error page." >&2
    echo "                   Re-run with --force to retry." >&2
    exit 1
fi

echo "fetch_embed_model.sh: OK — ${DEST} (${SIZE} bytes)"
