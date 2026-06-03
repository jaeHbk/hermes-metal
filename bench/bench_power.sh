#!/usr/bin/env bash
# bench/bench_power.sh — wall-clock power telemetry around an inference run.
#
# Wraps `powermetrics` (CPU + GPU power samplers) around a single backend run
# of bench_throughput.py and reports joules + Wh per 1k generated tokens.
# Requires sudo because powermetrics is privileged.
#
# Usage:
#   sudo bench/bench_power.sh llama_cpp [prompt_id]
#   sudo bench/bench_power.sh mlx       [prompt_id]
#
# The "battery cost" claim of hermes-metal lives or dies in this number. We
# report joules/1k-tokens (a backend-agnostic unit) so the comparison holds
# whether the host is on battery or wall power.

set -euo pipefail

BACKEND="${1:-}"
PROMPT_ID="${2:-medium_summary}"

if [ -z "$BACKEND" ] || { [ "$BACKEND" != "llama_cpp" ] && [ "$BACKEND" != "mlx" ]; }; then
    echo "usage: sudo $0 {llama_cpp|mlx} [prompt_id]" >&2
    exit 2
fi

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: powermetrics requires sudo. Re-run as: sudo $0 $BACKEND $PROMPT_ID" >&2
    exit 1
fi

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$BENCH_DIR/.." && pwd)"
RESULTS_DIR="$BENCH_DIR/results"
mkdir -p "$RESULTS_DIR"

STAMP=$(date +%Y%m%dT%H%M%S)
PM_LOG="$RESULTS_DIR/power-${BACKEND}-${STAMP}.txt"
RUN_LOG="$RESULTS_DIR/power-${BACKEND}-${STAMP}.run.log"

# Pick the right venv per backend.
if [ "$BACKEND" = "mlx" ]; then
    PY="$BENCH_DIR/.venv-mlx/bin/python"
else
    PY="$REPO_ROOT/.venv/bin/python"
fi

if [ ! -x "$PY" ]; then
    echo "ERROR: $PY not found. Run 'bench/setup.sh' first." >&2
    exit 1
fi

# Sample CPU + GPU power at 500ms cadence. (ANE residency is exposed via a
# separate sampler in powermetrics, but the sampler name varies by macOS
# release — and the on-device 8B inference doesn't actually use ANE since
# llama.cpp/MLX both target Metal. So we skip ane to keep the suite portable.)
# powermetrics emits one sample group per --sample-interval.
echo "==> bench_power: $BACKEND ($PROMPT_ID)  log=$PM_LOG"
powermetrics \
    --samplers cpu_power,gpu_power,thermal \
    --sample-interval 500 \
    --output-file "$PM_LOG" \
    --hide-cpu-duty-cycle \
    >/dev/null 2>"$RESULTS_DIR/power-${BACKEND}-${STAMP}.pm.err" &
PM_PID=$!

# Trap so we always stop powermetrics, even on Ctrl-C / failure.
cleanup() {
    if kill -0 "$PM_PID" 2>/dev/null; then
        kill "$PM_PID" 2>/dev/null || true
        wait "$PM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Verify powermetrics actually started writing samples; fail fast if it didn't
# (wrong sampler name on this macOS release, sip blocking it, etc.).
sleep 2
if [ ! -s "$PM_LOG" ]; then
    echo "ERROR: powermetrics produced no output. stderr:" >&2
    cat "$RESULTS_DIR/power-${BACKEND}-${STAMP}.pm.err" >&2
    exit 1
fi

# Run the workload non-root by dropping back to the invoking user. SUDO_USER is
# set when invoked via `sudo`; fall back to whoami if missing.
RUN_USER="${SUDO_USER:-$(whoami)}"
echo "==> running bench_throughput as user=$RUN_USER backend=$BACKEND prompt=$PROMPT_ID"

T0=$(python3 -c 'import time; print(time.perf_counter())')
# `sudo -u "$RUN_USER" sh -c '...'` puts the redirect inside the dropped-priv
# shell so the run log (and the throughput-*.json it writes via cwd) are owned
# by the invoking user, not root.
sudo -u "$RUN_USER" sh -c \
    "cd '$REPO_ROOT' && '$PY' -m bench.bench_throughput --backend '$BACKEND' --prompt-id '$PROMPT_ID' >'$RUN_LOG' 2>&1"
T1=$(python3 -c 'import time; print(time.perf_counter())')

cleanup

WALL_S=$(python3 -c "print(round($T1 - $T0, 3))")
echo "==> workload wall: ${WALL_S}s"

# ---- post-process: integrate watts over the wall window ----------------------
python3 - "$PM_LOG" "$RUN_LOG" "$BACKEND" "$PROMPT_ID" "$STAMP" "$WALL_S" <<'PY'
import json, re, sys, time
from pathlib import Path

pm_path, run_log, backend, prompt_id, stamp, wall_s = sys.argv[1:7]
wall_s = float(wall_s)

# powermetrics emits "CPU Power: NNN mW" and "GPU Power: NNN mW" at the
# configured sample interval (500ms => 0.5s integration step). Output format
# is a release-stable plain-text dump.
SAMPLE_S = 0.5
total_j = {"cpu": 0.0, "gpu": 0.0}
samples = {"cpu": 0, "gpu": 0}

text = Path(pm_path).read_text(errors="replace")
for line in text.splitlines():
    m = re.match(r"\s*(CPU|GPU) Power:\s*([0-9]+)\s*mW", line)
    if not m:
        continue
    key = m.group(1).lower()
    mw = int(m.group(2))
    total_j[key] += (mw / 1000.0) * SAMPLE_S
    samples[key] += 1

# Pull tokens-generated from the run log so we can compute per-1k-token energy.
# bench_throughput.py prints lines like:
#   "    ttft=...ms  prefill=...t/s  decode=...t/s  rss=...MiB  wall=...s"
# but we want the n_out token count; parse the structured JSON it just wrote
# instead. The most recent throughput-*.json in bench/results/ is ours.
results_dir = Path(__file__).resolve().parent.parent / "bench" / "results" \
    if False else Path(run_log).parent
results = sorted(results_dir.glob("throughput-*.json"))
output_tokens = 0
input_tokens = 0
if results:
    payload = json.loads(results[-1].read_text())
    for r in payload.get("runs", []):
        if r.get("prompt_id") == prompt_id and r.get("backend") == backend:
            output_tokens = r["output_tokens"]
            input_tokens = r["input_tokens"]
            break

total_pkg_j = sum(total_j.values())
per_1k = (total_pkg_j / output_tokens * 1000.0) if output_tokens else 0.0
wh = total_pkg_j / 3600.0

out = {
    "kind": "power",
    "stamp": stamp,
    "backend": backend,
    "prompt_id": prompt_id,
    "wall_s": wall_s,
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "samples": samples,
    "energy_j": total_j,
    "energy_total_j": total_pkg_j,
    "energy_total_wh": wh,
    "joules_per_1k_output_tokens": per_1k,
}
out_path = Path(pm_path).with_suffix(".json")
out_path.write_text(json.dumps(out, indent=2))
print(f"==> {out_path}")
print(f"    cpu: {total_j['cpu']:.1f} J   gpu: {total_j['gpu']:.1f} J   "
      f"total: {total_pkg_j:.1f} J ({wh*1000:.1f} mWh)")
print(f"    output tokens: {output_tokens}   joules/1k tokens: {per_1k:.2f}")
PY
