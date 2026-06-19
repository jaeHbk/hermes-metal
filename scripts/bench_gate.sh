#!/usr/bin/env bash
# bench_gate.sh — definition-of-done for a perf experiment.
#   --quick : pytest + short-prompt throughput + RSS (fast iteration, no ppl)
#   --full  : adds perplexity (mandatory pre-PR / Refinery validation)
# exit 0 PASS · 1 FAIL · 2 ERROR. Always writes bench/results/gate-<sha>.json.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 2

MODE="${1:---full}"
BASELINE="bench/results/baseline.json"
SHA="$(git rev-parse --short HEAD)"
GATE_OUT="bench/results/gate-${SHA}.json"
PY=".venv/bin/python"
RSS_CEILING="${BENCH_GATE_RSS_CEILING_MIB:-1024}"   # absolute MiB ceiling; default 1024 (36GiB tier). A bead trading RSS (e.g. E5) sets this lower/explicitly.

fail()  { echo "GATE ERROR: $*" >&2; exit 2; }
[ -f "$BASELINE" ] || fail "no baseline.json — run gate make-baseline first"

# 1. correctness
echo "== gate: pytest =="
$PY -m pytest tests/ -q || { echo "GATE FAIL: tests"; exit 1; }

# 2. build (only if engine inputs changed vs baseline branch)
if git diff --name-only HEAD~1 2>/dev/null | grep -qE '^(third_party/llama.cpp|config/engine_flags.env)'; then
  echo "== gate: build-engine =="
  make build-engine || { echo "GATE FAIL: build"; exit 1; }
fi

# 3. throughput (llama_cpp only is enough for the gate)
echo "== gate: throughput =="
$PY -m bench.bench_throughput --backend llama_cpp || fail "throughput run"
T=$(ls -t bench/results/throughput-*.json | head -1)
[ -n "$T" ] || fail "no throughput json produced"

# 4. perplexity (full mode only)
if [ "$MODE" = "--full" ]; then
  echo "== gate: perplexity =="
  $PY -m bench.bench_perplexity --backend llama_cpp || fail "perplexity run"
  P=$(ls -t bench/results/perplexity-*.json | head -1)
  [ -n "$P" ] || fail "no perplexity json produced"
else
  P="$BASELINE"; PFLAG_NOTE="(quick: ppl skipped)"
fi

# 5/6. verdict
echo "== gate: verdict ${PFLAG_NOTE:-} =="
ARGS=(check --current-throughput "$T" --baseline "$BASELINE" --out "$GATE_OUT")
if [ "$MODE" = "--full" ]; then ARGS+=(--current-perplexity "$P"); else ARGS+=(--current-perplexity "$BASELINE"); fi
ARGS+=(--rss-ceiling-mib "$RSS_CEILING")
$PY -m bench.gate "${ARGS[@]}"; RC=$?
if [ "$RC" -ne 0 ] && [ ! -f "$GATE_OUT" ]; then
  fail "gate.py errored without producing a verdict (exit $RC) — bad input or IO, not a regression"
fi
exit $RC
