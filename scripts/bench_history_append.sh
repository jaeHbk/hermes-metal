#!/usr/bin/env bash
# Append one row to history.tsv from a gate JSON. Args: <experiment-label> <gate-json>
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
[ $# -eq 2 ] || { echo "usage: $0 <experiment-label> <gate-json>" >&2; exit 2; }
LABEL="$1"; GATE="$2"
[ -f "$GATE" ] || { echo "error: gate-json not found: $GATE" >&2; exit 2; }
.venv/bin/python - "$LABEL" "$GATE" <<'PY'
import json, sys, subprocess, datetime
label = sys.argv[1].replace("\t", " ").replace("\n", " ")
gate_path = sys.argv[2]
g = json.loads(open(gate_path).read())
sha = subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
d = g.get("deltas", {})
def dec(pid): return f"{d.get(pid,{}).get('decode_tps',0):+.1f}"
row = "\t".join([
    datetime.date.today().isoformat(), sha, label,
    dec("short_qa"), dec("medium_summary"), dec("long_context"),
    f"{g.get('max_rss_bytes',0)/1048576:.0f}",
    f"{g.get('current_ppl',0):.4f}",
    "ok" if g.get("ok") else "FAIL",
])
open("bench/results/history.tsv","a").write(row+"\n")
print("appended:", row)
PY
