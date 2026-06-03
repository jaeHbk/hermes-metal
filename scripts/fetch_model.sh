#!/usr/bin/env bash
# fetch_model.sh — download the canonical Hermes 8B GGUF (Q4_K_M) into ./models/.
#
# Behavior:
#   - Sources config/engine_flags.env for $MODEL_URL and $MODEL_FILE.
#   - Skips download if models/$MODEL_FILE already exists, unless --force
#     is passed as the first argument.
#   - Sanity-checks the resulting file is > 4 GB (the canonical
#     Hermes-3-Llama-3.1-8B.Q4_K_M.gguf is ~4.92 GB / 4_920_733_824 bytes).
#
# Run from anywhere; the script cd's to the repo root before doing work.

set -euo pipefail

# Resolve repo root = parent of the directory holding this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ENV_FILE="config/engine_flags.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "fetch_model.sh: missing ${ENV_FILE}; run \`make check-env\` first." >&2
    exit 1
fi

# shellcheck disable=SC1090
set -a
source "${ENV_FILE}"
set +a

: "${MODEL_URL:?MODEL_URL not set in ${ENV_FILE}}"
: "${MODEL_FILE:?MODEL_FILE not set in ${ENV_FILE}}"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

mkdir -p models
DEST="models/${MODEL_FILE}"

if [[ -f "${DEST}" && "${FORCE}" -eq 0 ]]; then
    echo "fetch_model.sh: ${DEST} already exists; pass --force to overwrite."
else
    echo "fetch_model.sh: downloading ${MODEL_URL}"
    echo "             -> ${DEST}"
    # --fail   : exit non-zero on HTTP >=400 instead of writing an error page.
    # -L       : follow redirects (HF resolve URLs redirect to a CDN).
    # --progress-bar : human-friendly bar; replaces the noisy default meter.
    curl -L --fail --progress-bar "${MODEL_URL}" -o "${DEST}.part"
    mv "${DEST}.part" "${DEST}"
fi

# Sanity-check size. Use stat in BSD (macOS) form first, GNU as fallback.
if SIZE=$(stat -f%z "${DEST}" 2>/dev/null); then
    :
else
    SIZE=$(stat -c%s "${DEST}")
fi

MIN_BYTES=4000000000  # 4 GB; canonical Q4_K_M is ~4.92 GB.
if (( SIZE < MIN_BYTES )); then
    echo "fetch_model.sh: ${DEST} is only ${SIZE} bytes (< ${MIN_BYTES})." >&2
    echo "             Likely a truncated download or an HTML error page." >&2
    echo "             Re-run with --force to retry." >&2
    exit 1
fi

echo "fetch_model.sh: OK — ${DEST} (${SIZE} bytes)"
