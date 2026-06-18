# hermes-metal Gastown Performance Slice — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a reusable Gastown multi-agent harness on hermes-metal and use it to land performance gains, each guarded by a bench-gate that forbids RSS/perplexity regressions.

**Architecture:** A small testable Python module (`bench/gate.py`) holds the regression/tolerance logic; `scripts/bench_gate.sh` is a thin orchestrator that runs tests + the existing `bench/` harness, then calls the module to produce a pass/fail verdict + delta JSON. Gastown (already installed) drives a Beads backlog of independent perf experiments; polecats work in isolated worktrees and submit merge requests to a Refinery whose validation IS the gate; merges land on an epic integration branch (`perf/gastown-run-1`), never `main`/`feat/phases-b-e`. The E1 experiment is run first as an end-to-end spike (plus a deliberate negative test) before fanning out.

**Tech Stack:** Python 3.11 (stdlib + existing `bench/` modules; no new deps), bash, pytest, Gastown `gt` 1.1.0 + Beads `bd`, git worktrees, llama.cpp (Metal), macOS launchd. Apple-Silicon-only; everything runs on this Mac.

**Spec:** [`../specs/2026-06-18-hermes-metal-gastown-perf-design.md`](../specs/2026-06-18-hermes-metal-gastown-perf-design.md)

---

## File structure

| File | New? | Responsibility |
| --- | --- | --- |
| `bench/gate.py` | new | Pure logic: load result JSON, build baseline, compare current vs baseline with tolerances + per-bead RSS ceiling, emit verdict. The testable contract. |
| `tests/test_gate.py` | new | pytest coverage of `bench/gate.py` (the only new test file; keeps the gate honest). |
| `scripts/bench_gate.sh` | new | Thin orchestrator: pytest → (build if needed) → bench throughput/perplexity → `python -m bench.gate check`. Modes `--quick`/`--full`. |
| `bench/results/baseline.json` | new | Seeded reference numbers (per-prompt decode/prefill/RSS + perplexity). |
| `bench/results/history.tsv` | new | Append-only versioned benchmark history. |
| `scripts/bench_history_append.sh` | new | Append one row to history.tsv from a gate JSON (used on accepted merge). |
| `.beads/` | new | Beads backlog E1–E7 (created by `bd init`). |
| `docs/gastown-harness.md` | new | How to run/reuse the harness (for later slices). |
| `IMPROVEMENTS.md` | modify | Slice writeup at the end (problem/change/impact). |

`bench/gate.py` deliberately splits the load-bearing comparison logic out of the shell so it is unit-testable; `scripts/bench_gate.sh` stays dumb. This is a refinement of the spec's "one shell script" — same contract, testable core.

---

### Task 1: Create the work branch

**Files:** none (git only)

- [ ] **Step 1: Branch off the current feature branch**

Run:
```bash
cd ~/Documents/hermes-metal
git checkout feat/phases-b-e
git checkout -b feat/gastown-bench-gate
```
Expected: `Switched to a new branch 'feat/gastown-bench-gate'`

- [ ] **Step 2: Confirm clean tree**

Run: `git status --short`
Expected: empty output.

---

### Task 2: `bench/gate.py` — loaders (TDD)

**Files:**
- Create: `bench/gate.py`
- Test: `tests/test_gate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gate.py`:
```python
import json
from pathlib import Path
from bench import gate


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def test_load_throughput_indexes_by_prompt(tmp_path):
    payload = {
        "kind": "throughput",
        "runs": [
            {"backend": "llama_cpp", "prompt_id": "short_qa",
             "decode_tps": 23.8, "prefill_tps": 199.8, "peak_rss_bytes": 190_840_832},
            {"backend": "mlx", "prompt_id": "short_qa",
             "decode_tps": 29.9, "prefill_tps": 142.6, "peak_rss_bytes": 1_900_000_000},
        ],
    }
    path = _write(tmp_path, "throughput-x.json", payload)
    got = gate.load_throughput(path)            # llama_cpp only
    assert set(got) == {"short_qa"}
    assert got["short_qa"]["decode_tps"] == 23.8
    assert got["short_qa"]["peak_rss_bytes"] == 190_840_832


def test_load_perplexity_picks_llama_cpp(tmp_path):
    payload = {"kind": "perplexity", "runs": [
        {"backend": "llama_cpp", "perplexity": 8.4047},
        {"backend": "mlx", "perplexity": 11.7232},
    ]}
    path = _write(tmp_path, "perplexity-x.json", payload)
    assert gate.load_perplexity(path) == 8.4047
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'bench.gate'`.

- [ ] **Step 3: Write minimal implementation**

Create `bench/gate.py`:
```python
"""bench/gate.py — regression gate logic for the Gastown perf harness.

Pure, unit-testable core. `scripts/bench_gate.sh` is the thin orchestrator
that runs pytest + the bench harness, then calls this module to decide
pass/fail. We only ever judge the llama.cpp backend (the daemon's runtime);
MLX rows in the result JSON are ignored.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LLAMA_CPP = "llama_cpp"


def load_throughput(path: str | Path) -> dict[str, dict[str, float]]:
    payload = json.loads(Path(path).read_text())
    out: dict[str, dict[str, float]] = {}
    for r in payload.get("runs", []):
        if r.get("backend") != LLAMA_CPP:
            continue
        out[r["prompt_id"]] = {
            "decode_tps": float(r["decode_tps"]),
            "prefill_tps": float(r["prefill_tps"]),
            "peak_rss_bytes": int(r["peak_rss_bytes"]),
        }
    return out


def load_perplexity(path: str | Path) -> float:
    payload = json.loads(Path(path).read_text())
    for r in payload.get("runs", []):
        if r.get("backend") == LLAMA_CPP:
            return float(r["perplexity"])
    raise ValueError(f"no llama_cpp perplexity run in {path}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gate.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add bench/gate.py tests/test_gate.py
git commit -m "feat(bench): gate.py result loaders + tests"
```

---

### Task 3: `bench/gate.py` — `make_baseline` (TDD)

**Files:**
- Modify: `bench/gate.py`
- Test: `tests/test_gate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gate.py`:
```python
def test_make_baseline_shape(tmp_path):
    tput = _write(tmp_path, "throughput-x.json", {
        "host": {"chip": "Apple M3 Pro"},
        "runs": [{"backend": "llama_cpp", "prompt_id": "short_qa",
                  "decode_tps": 23.8, "prefill_tps": 199.8, "peak_rss_bytes": 190_840_832}],
    })
    ppl = _write(tmp_path, "perplexity-x.json", {
        "runs": [{"backend": "llama_cpp", "perplexity": 8.4047}],
    })
    base = gate.make_baseline(tput, ppl)
    assert base["perplexity"] == 8.4047
    assert base["prompts"]["short_qa"]["decode_tps"] == 23.8
    assert base["host"]["chip"] == "Apple M3 Pro"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gate.py::test_make_baseline_shape -q`
Expected: FAIL — `AttributeError: module 'bench.gate' has no attribute 'make_baseline'`.

- [ ] **Step 3: Write minimal implementation**

Append to `bench/gate.py`:
```python
def make_baseline(throughput_path: str | Path, perplexity_path: str | Path) -> dict[str, Any]:
    host = json.loads(Path(throughput_path).read_text()).get("host", {})
    return {
        "host": host,
        "prompts": load_throughput(throughput_path),
        "perplexity": load_perplexity(perplexity_path),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gate.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add bench/gate.py tests/test_gate.py
git commit -m "feat(bench): gate.make_baseline"
```

---

### Task 4: `bench/gate.py` — `compare` verdict (TDD, the core)

**Files:**
- Modify: `bench/gate.py`
- Test: `tests/test_gate.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gate.py`:
```python
BASE = {
    "prompts": {
        "short_qa":       {"decode_tps": 23.8, "prefill_tps": 199.8, "peak_rss_bytes": 190_840_832},
        "medium_summary": {"decode_tps": 19.3, "prefill_tps": 321.9, "peak_rss_bytes": 263_192_576},
    },
    "perplexity": 8.40,
}


def _cur(decode=24.0, rss=190_840_832, ppl=8.40):
    return (
        {"short_qa": {"decode_tps": decode, "prefill_tps": 199.8, "peak_rss_bytes": rss},
         "medium_summary": {"decode_tps": 19.3, "prefill_tps": 321.9, "peak_rss_bytes": 263_192_576}},
        ppl,
    )


def test_compare_passes_on_improvement():
    cur_t, cur_p = _cur(decode=26.0)
    v = gate.compare(cur_t, cur_p, BASE)
    assert v.ok and v.improved
    assert v.deltas["short_qa"]["decode_tps"] > 0


def test_compare_fails_on_perplexity_regression():
    cur_t, cur_p = _cur(ppl=8.55)            # +0.15 > tol 0.10
    v = gate.compare(cur_t, cur_p, BASE)
    assert not v.ok and "perplexity" in v.reason


def test_compare_passes_within_perplexity_noise():
    cur_t, cur_p = _cur(ppl=8.49)            # +0.09 < tol 0.10
    v = gate.compare(cur_t, cur_p, BASE)
    assert v.ok


def test_compare_fails_on_rss_regression_default():
    cur_t, cur_p = _cur(rss=int(190_840_832 * 1.2))   # +20% > factor 1.10
    v = gate.compare(cur_t, cur_p, BASE)
    assert not v.ok and "rss" in v.reason.lower()


def test_compare_rss_ceiling_override_allows_memory_trade():
    # Speculative decoding: 600 MiB resident, way past x1.10, but under the
    # absolute ceiling — must PASS so the human sees the speed win.
    cur_t, cur_p = _cur(decode=40.0, rss=600 * 1024 * 1024)
    v = gate.compare(cur_t, cur_p, BASE, rss_ceiling_mib=700)
    assert v.ok and v.improved


def test_compare_rss_ceiling_still_a_hard_floor():
    cur_t, cur_p = _cur(decode=40.0, rss=900 * 1024 * 1024)   # over 700 ceiling
    v = gate.compare(cur_t, cur_p, BASE, rss_ceiling_mib=700)
    assert not v.ok and "ceiling" in v.reason.lower()


def test_compare_no_gain_passes_but_flags_not_improved():
    cur_t, cur_p = _cur(decode=23.8)         # identical
    v = gate.compare(cur_t, cur_p, BASE)
    assert v.ok and not v.improved
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_gate.py -q`
Expected: FAIL — `AttributeError: module 'bench.gate' has no attribute 'compare'`.

- [ ] **Step 3: Write minimal implementation**

Append to `bench/gate.py`:
```python
from dataclasses import dataclass, field

PPL_TOL = 0.10          # REPORT.md: <0.1 is tokenizer noise
RSS_FACTOR = 1.10       # default relative ceiling


@dataclass
class Verdict:
    ok: bool
    improved: bool
    reason: str
    deltas: dict[str, dict[str, float]] = field(default_factory=dict)
    current_ppl: float = 0.0
    baseline_ppl: float = 0.0
    max_rss_bytes: int = 0


def compare(
    current_throughput: dict[str, dict[str, float]],
    current_ppl: float,
    baseline: dict[str, Any],
    rss_ceiling_mib: int | None = None,
    ppl_tol: float = PPL_TOL,
    rss_factor: float = RSS_FACTOR,
) -> Verdict:
    base_prompts = baseline["prompts"]
    base_ppl = float(baseline["perplexity"])

    # --- deltas (throughput is the objective, never a hard fail) ---
    deltas: dict[str, dict[str, float]] = {}
    improved = False
    for pid, cur in current_throughput.items():
        b = base_prompts.get(pid)
        if not b:
            continue
        d_dec = cur["decode_tps"] - b["decode_tps"]
        d_pre = cur["prefill_tps"] - b["prefill_tps"]
        d_rss = cur["peak_rss_bytes"] - b["peak_rss_bytes"]
        deltas[pid] = {"decode_tps": d_dec, "prefill_tps": d_pre, "peak_rss_bytes": d_rss}
        if d_dec > 0.5 or d_pre > 1.0:        # beyond run-to-run jitter
            improved = True

    max_rss = max((c["peak_rss_bytes"] for c in current_throughput.values()), default=0)

    # --- quality guardrail ---
    if current_ppl > base_ppl + ppl_tol:
        return Verdict(False, improved,
                       f"perplexity regression: {current_ppl:.4f} > {base_ppl:.4f}+{ppl_tol}",
                       deltas, current_ppl, base_ppl, max_rss)

    # --- memory guardrail ---
    if rss_ceiling_mib is not None:
        ceiling = rss_ceiling_mib * 1024 * 1024
        if max_rss > ceiling:
            return Verdict(False, improved,
                           f"rss over absolute ceiling: {max_rss} > {ceiling} ({rss_ceiling_mib} MiB)",
                           deltas, current_ppl, base_ppl, max_rss)
    else:
        for pid, cur in current_throughput.items():
            b = base_prompts.get(pid)
            if b and cur["peak_rss_bytes"] > b["peak_rss_bytes"] * rss_factor:
                return Verdict(False, improved,
                               f"rss regression on {pid}: {cur['peak_rss_bytes']} "
                               f"> {b['peak_rss_bytes']}x{rss_factor}",
                               deltas, current_ppl, base_ppl, max_rss)

    return Verdict(True, improved, "ok", deltas, current_ppl, base_ppl, max_rss)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gate.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add bench/gate.py tests/test_gate.py
git commit -m "feat(bench): gate.compare verdict with ppl/rss guardrails + per-bead rss ceiling"
```

---

### Task 5: `bench/gate.py` — CLI (`make-baseline` / `check`)

**Files:**
- Modify: `bench/gate.py`
- Test: `tests/test_gate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gate.py`:
```python
import subprocess, sys


def test_cli_check_exit_codes(tmp_path):
    base = tmp_path / "baseline.json"
    base.write_text(json.dumps(BASE))
    tput = _write(tmp_path, "throughput-x.json", {
        "runs": [{"backend": "llama_cpp", "prompt_id": "short_qa",
                  "decode_tps": 26.0, "prefill_tps": 199.8, "peak_rss_bytes": 190_840_832},
                 {"backend": "llama_cpp", "prompt_id": "medium_summary",
                  "decode_tps": 19.3, "prefill_tps": 321.9, "peak_rss_bytes": 263_192_576}],
    })
    ppl = _write(tmp_path, "perplexity-x.json", {
        "runs": [{"backend": "llama_cpp", "perplexity": 8.40}]})
    out = tmp_path / "gate.json"
    r = subprocess.run(
        [sys.executable, "-m", "bench.gate", "check",
         "--current-throughput", str(tput), "--current-perplexity", str(ppl),
         "--baseline", str(base), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text())["ok"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gate.py::test_cli_check_exit_codes -q`
Expected: FAIL — non-zero return / `No module named bench.gate.__main__`-style error.

- [ ] **Step 3: Write minimal implementation**

Append to `bench/gate.py`:
```python
def _cli(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="bench.gate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    mb = sub.add_parser("make-baseline")
    mb.add_argument("--throughput", required=True)
    mb.add_argument("--perplexity", required=True)
    mb.add_argument("--out", required=True)

    ck = sub.add_parser("check")
    ck.add_argument("--current-throughput", required=True)
    ck.add_argument("--current-perplexity", required=True)
    ck.add_argument("--baseline", required=True)
    ck.add_argument("--rss-ceiling-mib", type=int, default=None)
    ck.add_argument("--out", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "make-baseline":
        base = make_baseline(args.throughput, args.perplexity)
        Path(args.out).write_text(json.dumps(base, indent=2))
        print(f"wrote baseline: {args.out}")
        return 0

    baseline = json.loads(Path(args.baseline).read_text())
    cur_t = load_throughput(args.current_throughput)
    cur_p = load_perplexity(args.current_perplexity)
    v = compare(cur_t, cur_p, baseline, rss_ceiling_mib=args.rss_ceiling_mib)
    Path(args.out).write_text(json.dumps({
        "ok": v.ok, "improved": v.improved, "reason": v.reason,
        "deltas": v.deltas, "current_ppl": v.current_ppl,
        "baseline_ppl": v.baseline_ppl, "max_rss_bytes": v.max_rss_bytes,
    }, indent=2))
    # human-readable card
    print(f"gate: {'PASS' if v.ok else 'FAIL'}  ({v.reason})")
    for pid, d in v.deltas.items():
        print(f"  {pid}: decode {d['decode_tps']:+.1f} t/s  "
              f"prefill {d['prefill_tps']:+.1f} t/s  rss {d['peak_rss_bytes']/1048576:+.0f} MiB")
    print(f"  ppl {v.current_ppl:.4f} (base {v.baseline_ppl:.4f})  "
          f"max rss {v.max_rss_bytes/1048576:.0f} MiB")
    return 0 if v.ok else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gate.py -q`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
git add bench/gate.py tests/test_gate.py
git commit -m "feat(bench): gate CLI (make-baseline / check)"
```

---

### Task 6: Seed `bench/results/baseline.json`

**Files:**
- Create: `bench/results/baseline.json`

- [ ] **Step 1: Ensure the daemon is up (baseline must be measured on current code)**

Run: `hermes doctor` (or `make start-daemon` then `hermes doctor`)
Expected: chat server healthy on :8080. If not, fix before continuing — a baseline measured against a down daemon is invalid.

- [ ] **Step 2: Generate fresh result JSONs on the CURRENT (unchanged) engine**

Run:
```bash
make bench-throughput
make bench-perplexity
```
Expected: writes `bench/results/throughput-<ts>.json` and `perplexity-<ts>.json`.

- [ ] **Step 3: Build baseline.json from the newest results**

Run:
```bash
T=$(ls -t bench/results/throughput-*.json | head -1)
P=$(ls -t bench/results/perplexity-*.json | head -1)
.venv/bin/python -m bench.gate make-baseline --throughput "$T" --perplexity "$P" \
  --out bench/results/baseline.json
```
Expected: `wrote baseline: bench/results/baseline.json`.

- [ ] **Step 4: Sanity-check the values match REPORT.md ballpark**

Run: `cat bench/results/baseline.json`
Expected: per-prompt `decode_tps` ≈ 19–25, `peak_rss_bytes` ≤ ~315 MiB, `perplexity` ≈ 8.40. If wildly off, the daemon was misconfigured — re-measure.

- [ ] **Step 5: Commit**

```bash
git add bench/results/baseline.json
git commit -m "chore(bench): seed performance baseline.json"
```

---

### Task 7: `scripts/bench_gate.sh` — the orchestrator

**Files:**
- Create: `scripts/bench_gate.sh`

- [ ] **Step 1: Write the script**

Create `scripts/bench_gate.sh`:
```bash
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
RSS_CEILING="${BENCH_GATE_RSS_CEILING_MIB:-}"   # set by a bead that trades RSS

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

# 3. throughput (llama_cpp only is enough for the gate; 'both' also fine)
echo "== gate: throughput =="
$PY -m bench.bench_throughput --backend llama_cpp || fail "throughput run"
T=$(ls -t bench/results/throughput-*.json | head -1)

# 4. perplexity (full mode only)
if [ "$MODE" = "--full" ]; then
  echo "== gate: perplexity =="
  $PY -m bench.bench_perplexity --backend llama_cpp || fail "perplexity run"
  P=$(ls -t bench/results/perplexity-*.json | head -1)
else
  # quick mode reuses the baseline ppl so compare() sees no quality change
  P="$BASELINE"; PFLAG_NOTE="(quick: ppl skipped)"
fi

# 5/6. verdict
echo "== gate: verdict ${PFLAG_NOTE:-} =="
ARGS=(check --current-throughput "$T" --baseline "$BASELINE" --out "$GATE_OUT")
if [ "$MODE" = "--full" ]; then ARGS+=(--current-perplexity "$P"); else ARGS+=(--current-perplexity "$BASELINE"); fi
[ -n "$RSS_CEILING" ] && ARGS+=(--rss-ceiling-mib "$RSS_CEILING")
$PY -m bench.gate "${ARGS[@]}"
exit $?
```

Note: in `--quick` mode we pass `--current-perplexity "$BASELINE"`; `load_perplexity` reads `runs[].perplexity`, so baseline.json needs a compatible shape. Add a `runs` shim in Step 2.

- [ ] **Step 2: Make baseline.json readable by `load_perplexity` in quick mode**

Edit `bench/gate.py` `make_baseline` to also embed a `runs` shim so the same file works as a `--current-perplexity` input:
```python
def make_baseline(throughput_path: str | Path, perplexity_path: str | Path) -> dict[str, Any]:
    host = json.loads(Path(throughput_path).read_text()).get("host", {})
    ppl = load_perplexity(perplexity_path)
    return {
        "host": host,
        "prompts": load_throughput(throughput_path),
        "perplexity": ppl,
        "runs": [{"backend": "llama_cpp", "perplexity": ppl}],  # so baseline.json doubles as a ppl input in --quick
    }
```
Run: `.venv/bin/python -m pytest tests/test_gate.py -q`
Expected: PASS (still 11 passed — the extra key doesn't break `make_baseline` shape test).

Then regenerate baseline.json (Task 6 Step 3) so it carries the shim, and re-commit it.

- [ ] **Step 3: chmod + syntax check**

Run:
```bash
chmod +x scripts/bench_gate.sh
bash -n scripts/bench_gate.sh && echo "SYNTAX OK"
```
Expected: `SYNTAX OK`.

- [ ] **Step 4: Smoke-test the gate on the unchanged tree (must PASS, not improved)**

Run: `scripts/bench_gate.sh --quick`
Expected: ends `gate: PASS  (ok)` with `~+0` deltas; exit 0. (`echo $?` → 0.)

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_gate.sh bench/gate.py bench/results/baseline.json
git commit -m "feat(bench): bench_gate.sh orchestrator (quick/full)"
```

---

### Task 8: `history.tsv` append helper

**Files:**
- Create: `scripts/bench_history_append.sh`
- Create: `bench/results/history.tsv` (header row)

- [ ] **Step 1: Create the header**

Run:
```bash
printf 'date\tsha\texperiment\tdecode_short\tdecode_medium\tdecode_long\tmax_rss_mib\tperplexity\tok\n' \
  > bench/results/history.tsv
```

- [ ] **Step 2: Write the append helper**

Create `scripts/bench_history_append.sh`:
```bash
#!/usr/bin/env bash
# Append one row to history.tsv from a gate JSON. Args: <experiment-label> <gate-json>
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
LABEL="$1"; GATE="$2"
.venv/bin/python - "$LABEL" "$GATE" <<'PY'
import json, sys, subprocess, datetime
label, gate_path = sys.argv[1], sys.argv[2]
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
```
Run: `chmod +x scripts/bench_history_append.sh && bash -n scripts/bench_history_append.sh && echo OK`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench_history_append.sh bench/results/history.tsv
git commit -m "feat(bench): history.tsv + append helper"
```

---

### Task 9: Register the rig + integration branch

**Files:** none (Gastown state lives under `~/gt`)

- [ ] **Step 1: Push the gate branch so worktrees can fork from it**

Run:
```bash
git push -u origin feat/gastown-bench-gate
```
Expected: branch on GitHub.

- [ ] **Step 2: Create the integration branch (merges land here, NOT main)**

Run:
```bash
git checkout -b perf/gastown-run-1
git push -u origin perf/gastown-run-1
git checkout feat/gastown-bench-gate
```
Expected: `perf/gastown-run-1` exists locally + remote.

- [ ] **Step 3: Register the rig with Gastown**

Run:
```bash
gt rig add hermes-metal ~/Documents/hermes-metal
gt rig list
```
Expected: `hermes-metal` appears in the rig list.

- [ ] **Step 4: Point the rig's merge target at the integration branch**

Run:
```bash
gt rig config hermes-metal           # inspect available keys
```
Set the main/merge-target branch to `perf/gastown-run-1` via the key shown (Gastown 1.1.0 exposes this under `gt rig config`; if it uses an epic integration branch instead, use `gt mq integration` — see Step 5). Confirm with `gt rig config hermes-metal` that the merge target is `perf/gastown-run-1`, never `main`/`feat/phases-b-e`.

- [ ] **Step 5: Wire the gate into Refinery validation**

Run:
```bash
gt rig config hermes-metal     # find the validation/check-command key
```
Set the Refinery validation command to `scripts/bench_gate.sh --full`. Verify it's recorded. (If 1.1.0 has no per-rig validation hook, the gate runs as the polecat's pre-`gt done` step instead — Task 11 Step 3 covers that fallback; either way no MR merges without a green gate JSON.)

- [ ] **Step 6: Boot witness + refinery**

Run:
```bash
gt rig boot hermes-metal
gt agents
```
Expected: witness + refinery sessions listed for the rig.

---

### Task 10: Seed the Beads backlog (E1–E7)

**Files:** `.beads/` (created by Gastown rig add or `bd init`)

- [ ] **Step 1: Confirm Beads is initialized for the rig**

Run: `bd list 2>/dev/null || (cd ~/Documents/hermes-metal && bd init)`
Expected: an empty (or rig) issue list, no error.

- [ ] **Step 2: Create the epic + E1 (spike)**

Run:
```bash
bd create --title "Perf slice: close llama.cpp decode/prefill gap" --type epic --priority 1
bd create --title "E1: verify -fa + --prompt-cache are live in engine flags" \
  --type task --priority 1 \
  --description "Audit config/engine_flags.env + the running engine LaunchAgent. CLAUDE.md mandates -fa and --prompt-cache; confirm both are actually passed. Fix drift if found. DoD: scripts/bench_gate.sh --full PASS. Spike: proves the whole loop."
```
Expected: two bead IDs printed (note the epic ID as `<EPIC>` and E1 as `<E1>`).

- [ ] **Step 3: Create E2–E7**

Run (one `bd create` each — full text so the polecat needs no other context):
```bash
bd create --title "E2: thread tuning (--threads / --threads-batch vs P/E split)" --type task --priority 2 \
  --description "Set --threads-batch to physical P-core count, --threads to a sensible decode value, in config/engine_flags.env. Hypothesis: prefill is compute-bound; matching threads-batch to P-cores lifts prefill t/s. DoD: bench_gate --full PASS, prefill improved, no ppl/rss regression."
bd create --title "E3: batch/ubatch tuning (-b/-ub) for prefill" --type task --priority 2 \
  --description "Raise -b/-ub in config/engine_flags.env for long-context prefill. Risk: RSS up — gate's x1.10 catches it. DoD: bench_gate --full PASS."
bd create --title "E4: KV-cache quantization (--cache-type-k/v q8_0)" --type task --priority 2 \
  --description "Add --cache-type-k q8_0 --cache-type-v q8_0. Hypothesis: frees memory headroom at ~0 quality cost. Risk: perplexity — gate catches >+0.10. DoD: bench_gate --full PASS."
bd create --title "E5: speculative decoding with tiny draft model" --type task --priority 3 \
  --description "Add a ~0.5-1B draft model (--model-draft) for speculative decoding. Biggest DECODE lever. Trades RSS for speed: set BENCH_GATE_RSS_CEILING_MIB=700 for this bead so the gate uses the absolute ceiling, not x1.10. DoD: bench_gate --full PASS with ceiling, decode improved, human approves RSS trade at merge."
bd create --title "E6: bump llama.cpp submodule" --type task --priority 3 \
  --description "Update third_party/llama.cpp to a newer tag; rebuild. Newer Metal kernels often raise decode/prefill. Risk: build/API drift — tests + gate catch. DoD: bench_gate --full PASS."
bd create --title "E7: quant variant probe (Q4_K_M vs IQ4_XS vs Q5_K_M)" --type task --priority 4 \
  --description "INFORMATIONAL: fetch alt-quant GGUFs into models/, bench each, report the speed/quality/RSS frontier in IMPROVEMENTS.md. Likely reports rather than merges. DoD: a written comparison; merge only if a variant strictly dominates."
```
Expected: five more bead IDs.

- [ ] **Step 4: Chain dependencies — E2–E7 blocked by the E1 spike**

Run (use the IDs from Steps 2–3):
```bash
for B in <E2> <E3> <E4> <E5> <E6> <E7>; do bd link "$B" --blocked-by <E1>; done
bd ready
```
Expected: `bd ready` shows only `<E1>` (the spike). Everything else is blocked until it lands.

- [ ] **Step 5: Commit the backlog**

```bash
git add .beads
git commit -m "chore(gastown): seed E1-E7 perf experiment backlog"
git push
```

---

### Task 11: Run the E1 spike end-to-end

**Files:** `config/engine_flags.env` (only if drift is found)

- [ ] **Step 1: Sling E1 to the rig (auto-spawns a polecat in its own worktree)**

Run:
```bash
gt sling <E1> hermes-metal --merge=mr
gt agents
```
Expected: a polecat session appears; its worktree lives under `~/gt/.../polecats/...`, NOT in `~/Documents/hermes-metal`.

- [ ] **Step 2: Confirm your working tree is untouched**

Run: `cd ~/Documents/hermes-metal && git status --short && git branch --show-current`
Expected: clean tree, still on `feat/gastown-bench-gate`. (The polecat works in its own worktree.)

- [ ] **Step 3: Let the polecat finish; it must pass the gate before `gt done`**

The polecat audits engine flags, fixes drift if any, then runs `scripts/bench_gate.sh --full` and signals `gt done` only on PASS. Watch:
```bash
gt trail            # recent agent activity
gt mq list          # the MR once submitted
```
Expected: an MR for E1 appears in the queue with a green `gate-<sha>.json`.

- [ ] **Step 4: Gate the merge (human)**

Run:
```bash
gt mq next                 # see the top MR + its gate card
gt mq status <MR-ID>       # diff + deltas
```
Review: tests 233/233, ppl within noise, RSS within +10%, decode/prefill deltas. If good:
```bash
gt mq submit               # or the rig's approve path; Refinery merges to perf/gastown-run-1
```
Expected: E1 merges to `perf/gastown-run-1`. Verify nothing landed on main/feat:
```bash
git log --oneline -1 origin/main; git log --oneline -1 origin/feat/phases-b-e
```
Expected: both unchanged.

- [ ] **Step 5: Update baseline + history from the accepted run**

Run (from the integration branch):
```bash
git checkout perf/gastown-run-1 && git pull
T=$(ls -t bench/results/throughput-*.json | head -1)
P=$(ls -t bench/results/perplexity-*.json | head -1)
.venv/bin/python -m bench.gate make-baseline --throughput "$T" --perplexity "$P" --out bench/results/baseline.json
scripts/bench_history_append.sh E1 "$(ls -t bench/results/gate-*.json | head -1)"
git add bench/results/baseline.json bench/results/history.tsv
git commit -m "chore(bench): update baseline + history after E1"
git push
```
Expected: baseline reflects E1; one new history row.

---

### Task 12: Negative test — prove the gate has teeth

**Files:** none kept (throwaway)

- [ ] **Step 1: Create a throwaway bad-change bead**

Run:
```bash
bd create --title "NEG: drop -fa to confirm gate rejects regressions" --type task --priority 0 \
  --description "Throwaway. Remove -fa from config/engine_flags.env. EXPECT the gate to FAIL (throughput/ppl regression) and NO MR to merge. Do not fix; this validates the safety net."
```

- [ ] **Step 2: Sling it and let it run the gate**

Run: `gt sling <NEG> hermes-metal --merge=mr`
Expected: the polecat runs `scripts/bench_gate.sh --full`, which **exits 1**; the polecat does NOT `gt done`, OR the Refinery validation rejects the MR.

- [ ] **Step 3: Confirm rejection**

Run: `gt mq list`
Expected: no merged MR for NEG (either never submitted, or shows rejected). `git log origin/perf/gastown-run-1` has no NEG commit.

- [ ] **Step 4: Clean up**

Run:
```bash
gt unsling <NEG> 2>/dev/null || true
bd close <NEG>
```
Expected: bead closed; no residue on the integration branch.

- [ ] **Step 5: Decision gate (per spec §7A)**

If E1 merged cleanly AND the negative test was rejected → harness is trusted; proceed to fan-out. If either failed → STOP and reconsider (fall back to plain Kiro/meshclaw subagents). Record the outcome in the next commit message.

---

### Task 13: Fan out E2–E4 (safe parallel batch)

**Files:** `config/engine_flags.env` (per experiment, in polecat worktrees)

- [ ] **Step 1: Confirm they're now ready (E1 unblocked them)**

Run: `bd ready`
Expected: E2, E3, E4 listed (E5/E6 still gated by priority/your call; E7 informational).

- [ ] **Step 2: Sling the safe batch (respect concurrency cap = 3)**

Run:
```bash
gt sling <E2> hermes-metal --merge=mr
gt sling <E3> hermes-metal --merge=mr
gt sling <E4> hermes-metal --merge=mr
gt agents          # confirm <= 3 active polecats
```
Expected: up to 3 polecats, each in its own worktree. If RAM pressure shows (each loads an 8B for bench), sling fewer at once.

- [ ] **Step 3: Gate each MR as it arrives (loop of Task 11 Step 4–5)**

For each MR: `gt mq next` → review card → approve if green → after merge, refresh baseline + append history (Task 11 Step 5 commands, label E2/E3/E4). Reject any that regress; the bead stays open for a retry with a new hypothesis.

- [ ] **Step 4: Checkpoint commit**

```bash
git checkout perf/gastown-run-1 && git pull
git log --oneline origin/feat/phases-b-e -1   # confirm still untouched
```

---

### Task 14: Fan out E5–E6 (higher ceiling, gated carefully)

**Files:** `config/engine_flags.env`, `models/` (E5 draft model)

- [ ] **Step 1: E6 submodule bump first (broad, but mechanical)**

Run: `gt sling <E6> hermes-metal --merge=mr`
Gate as usual; the build phase + 233 tests are the safety net for API drift.

- [ ] **Step 2: E5 speculative decoding with the RSS ceiling**

The E5 bead's description already instructs the polecat to export `BENCH_GATE_RSS_CEILING_MIB=700` so the gate uses the absolute ceiling. Sling it:
```bash
gt sling <E5> hermes-metal --merge=mr
```

- [ ] **Step 2b: Verify the ceiling actually applied (don't trust, check)**

When E5's MR lands in the queue, open its `gate-<sha>.json` and confirm `max_rss_bytes` is reported and the verdict honored the 700 MiB ceiling (not the x1.10 default). If the polecat forgot the env var, the gate would have failed on RSS — reject and re-sling with the ceiling set.

- [ ] **Step 3: Human-gate the RSS/speed trade explicitly**

Run: `gt mq next`
Review: decode delta should be large (the point of E5); `max rss` must be ≤ 700 MiB. Approve only if the speed win justifies the memory, and hermes still reads as a background daemon. Refresh baseline/history with label E5.

---

### Task 15: E7 informational probe + slice writeup

**Files:** `IMPROVEMENTS.md`, `docs/gastown-harness.md`

- [ ] **Step 1: Run E7 as a report, not a merge**

Run: `gt sling <E7> hermes-metal --merge=local`
Expected: the polecat fetches alt-quant GGUFs, benches each, and writes a comparison. `--merge=local` keeps it off the integration branch (it's informational).

- [ ] **Step 2: Write the harness doc**

Create `docs/gastown-harness.md` documenting: rig registration, the bench-gate contract, `gt sling`/`gt mq` loop, the RSS-ceiling mechanism, and how to reuse it for the usability/accessibility slices (point at `ROADMAP-gastown-slices.md`). Keep it under ~1 page; it's a runbook, not prose.

- [ ] **Step 3: Write the IMPROVEMENTS.md entry**

Prepend a dated entry (newest-first, matching the repo's problem/change/impact format) summarizing: the Gastown harness + bench-gate, which experiments landed, the measured decode/prefill deltas from `history.tsv`, and that RSS/perplexity held. Cite the final numbers from `bench/results/REPORT.md`.

- [ ] **Step 4: Refresh REPORT.md from the final integration-branch numbers**

Run: `make bench-report`
Expected: `bench/results/REPORT.md` reflects the improved numbers.

- [ ] **Step 5: Commit**

```bash
git checkout perf/gastown-run-1
git add docs/gastown-harness.md IMPROVEMENTS.md bench/results/REPORT.md bench/results/history.tsv
git commit -m "docs: Gastown perf slice writeup + harness runbook"
git push
```

---

### Task 16: Final cumulative PR (human merges by hand)

**Files:** none (git/PR)

- [ ] **Step 1: Open the integration PR**

Run:
```bash
gh pr create --base feat/phases-b-e --head perf/gastown-run-1 \
  --title "perf: Gastown slice — close decode/prefill gap (RSS/ppl held)" \
  --body "See docs/superpowers/specs/2026-06-18-hermes-metal-gastown-perf-design.md and history.tsv. All landed experiments passed the bench-gate; RSS and perplexity within guardrails."
```
Expected: a PR `perf/gastown-run-1 → feat/phases-b-e`.

- [ ] **Step 2: Final review (human)**

Review the cumulative diff + `history.tsv` + `REPORT.md`. Run `make test` and `scripts/bench_gate.sh --full` once more on the integration branch tip.
Expected: 233 green, gate PASS.

- [ ] **Step 3: Merge by hand**

Merge the PR yourself (not via Refinery). This is the one merge that reaches your feature branch.

- [ ] **Step 4: Park Gastown**

Run: `gt mayor detach 2>/dev/null || true; gt rig dock hermes-metal 2>/dev/null || true`
Expected: agents parked; Beads state preserved for the next slice.

---

## Self-review

**Spec coverage:**
- §3 architecture / bench-gate-as-contract → Tasks 2–7, 9.
- §4 gate (modes, tolerances, baseline seeding, per-bead RSS ceiling) → Tasks 2–7 (compare logic + ceiling in Task 4; modes in Task 7; baseline in Task 6).
- §4 versioned history (history.tsv) → Task 8, refreshed Tasks 11/13/15.
- §5 backlog E1–E7 + spike-first ordering + dependency chain → Task 10 (deps in Step 4), executed Tasks 11/13/14/15.
- §6 operator workflow (rig add, integration branch, gated merges, concurrency cap, stop/resume) → Tasks 9, 11, 13, 16 Step 4.
- §7A harness validation + negative test + decision gate → Tasks 11, 12.
- §7B per-experiment validation (gate is the test) → enforced every sling via Task 7 gate.
- §7C definition of done (harness doc, net improvement, cumulative PR by hand, IMPROVEMENTS entry) → Tasks 15, 16.
- §8 out-of-scope → not implemented (correct); roadmap already committed.
- §9 artifacts table → every listed path has a creating/modifying task.

**Placeholder scan:** No TBD/TODO. The only "confirm against 1.1.0" notes (Task 9 Steps 4–5) are deliberate — Gastown's per-rig config-key spelling must be read from the installed binary, and a concrete fallback (gate as polecat pre-`gt done` step) is specified so the task can't stall.

**Type/name consistency:** `gate.compare(...) -> Verdict(ok, improved, reason, deltas, current_ppl, baseline_ppl, max_rss_bytes)` used consistently in Tasks 4/5/7/8. CLI subcommands `make-baseline`/`check` consistent (Task 5 def, Tasks 6/11 use). `scripts/bench_gate.sh` modes `--quick`/`--full` consistent (Task 7 def, Tasks 11/12/13/14/16 use). `BENCH_GATE_RSS_CEILING_MIB` env var consistent (Task 7 reads, Task 10 E5 bead sets, Task 14 verifies). Integration branch `perf/gastown-run-1` consistent throughout.

**Known deviation from spec (intentional, noted):** spec §4 said "one shell script"; plan splits testable logic into `bench/gate.py` + thin `scripts/bench_gate.sh`. Same contract, TDD-able, fits the repo's pytest-first convention. Flagged here so review isn't surprised.
