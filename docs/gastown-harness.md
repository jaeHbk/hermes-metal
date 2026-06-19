# Gastown performance harness — runbook

A reusable multi-agent harness for running **gated performance experiments** on
hermes-metal. Each experiment runs as an isolated Gastown *polecat* that must
pass `scripts/bench_gate.sh --full` before its work can merge. Built and proven
during the 2026-06-18/19 performance slice (spec + plan under
`docs/superpowers/{specs,plans}/2026-06-18-hermes-metal-gastown-perf*`).

> **This is local-Mac only.** hermes-metal is Apple-Silicon/Metal-bound, so every
> polecat builds/benches on *this* machine. Gastown's coordinator could live
> elsewhere, but the workers cannot. Single Metal GPU ⇒ experiments are
> effectively **serial** (~10–15 min each), not parallel.

## The two pieces

1. **The bench-gate** (`scripts/bench_gate.sh` + `bench/gate.py`) — the
   definition-of-done. Pure-logic verdict in `bench/gate.py` (unit-tested,
   `tests/test_gate.py`); thin orchestrator in `scripts/bench_gate.sh`.
   - `--quick`: pytest + throughput + RSS (~1 min; reuses baseline ppl).
   - `--full`: adds perplexity (~4 min). **Mandatory before merge.**
   - Exit `0`=PASS, `1`=FAIL (real regression, verdict written), `2`=ERROR.
   - Guardrails: **perplexity** ≤ baseline+0.10 (hard), **RSS** ≤ absolute
     ceiling (default 1024 MiB; per-bead override via
     `BENCH_GATE_RSS_CEILING_MIB`). Throughput is the *objective*, never a hard
     fail. Baseline in `bench/results/baseline.json`; accepted runs append to
     `bench/results/history.tsv` (`scripts/bench_history_append.sh`).

2. **Gastown** (`gt` 1.1.0, Town HQ at `~/gt`, rig `hermes_metal`) — Mayor /
   polecats (isolated worktrees) / Refinery (merge queue) / Witness. Beads
   (`bd`, Dolt-backed on :3307) hold the experiment backlog. **Overseer = you**:
   approve/reject every merge.

## Run an experiment

```sh
# 0. Prereqs: chat daemon + Dolt up
curl -sf http://127.0.0.1:8080/health        # else: bash scripts/start-servers-manual.sh
gt dolt status                                # else: gt dolt start

# 1. File the experiment as a bead (one independent change each)
cd ~/gt/hermes_metal
bd create "E<n>: <hypothesis>" --type task -p 2 -d "<what to change + DoD: bench_gate --full PASS>"

# 2. Sling it → spawns a polecat in an isolated worktree
gt sling <bead-id> hermes_metal --merge=mr

# 3. Watch (polecat ~10-15 min: symlinks runtime, edits flags, runs gate, gt done)
gt polecat list hermes_metal
tmux -L gt-034423 capture-pane -p -t hm-<polecat> | tail -30   # live session

# 4. Overseer review when the MR lands
gt mq list hermes_metal
gt mq status <mr-id>                          # diff + gate verdict
#   approve → Refinery merges to perf/gastown-run-1
#   reject  → gt mq reject hermes_metal <mr-id> --reason "..."

# 5. After a real accepted change: refresh baseline + history
T=$(ls -t bench/results/throughput-*.json|head -1); P=$(ls -t bench/results/perplexity-*.json|head -1)
.venv/bin/python -m bench.gate make-baseline --throughput "$T" --perplexity "$P" --out bench/results/baseline.json
scripts/bench_history_append.sh E<n> "$(ls -t bench/results/gate-*.json|head -1)"
```

## MANDATORY per-bead guard (paste into every experiment's description)

Polecats start in a **fresh clone** missing the gitignored `.venv`,
`third_party/llama.cpp/build`, and `models/*.gguf` (~5 GB). They must **symlink**
those from `/Users/jaehunb/Documents/hermes-metal` — never rebuild, never commit
the symlinks. Before `gt done`, `git status` and ensure the MR diff is
**change-only** (config/engine_flags.env, submodule pointer, etc.); unstage
`.venv` / `.claude` / `.runtime` / `models` / `CLAUDE.local.md` / `bench/results/*.json`.
Also: **do not restart the :8080 chat daemon** (see RSS-warmth gotcha) and **do
not run open-ended A/B sweeps** — make the change, run the gate **once**, submit.

## Hard-won gotchas (all cost real time during the slice)

- **mmap RSS warmth (the big one).** llama.cpp mmaps the GGUF, so a daemon's
  *sampled* RSS ranges ~140 MiB (long-idle, pages evicted) → ~6.9 GB (freshly
  restarted, all resident) for the **identical** model+config. The bench samples
  daemon RSS, so **restarting the daemon mid-experiment false-fails the RSS
  gate**. This bit us 3×. Mitigations: absolute ceiling (not relative); don't
  restart the daemon; for a no-runtime-change commit the RSS fail is irrelevant
  (judge on the diff). A real fix (sample RSS only after a warmup/settle, or use
  a daemon-age-independent metric) is open follow-up.
- **Polecats over-run.** They treat bounded experiments as open research
  (observed 35+ min / 110k tokens on a one-line finding). Put an explicit scope
  budget in the bead ("change → gate once → submit; stop at 15 min") and be
  ready to `gt polecat nuke <rig>/<name>` after capturing the diff.
- **Cruft in MRs.** Polecats commit their `.venv` symlink / `CLAUDE.local.md`.
  The rig `.gitignore` now covers Gas Town overlay files; still verify diffs.
- **Stale resubmits.** An idle/zombie polecat can resubmit an already-closed
  bead's cruft MR. `gt polecat nuke` it once its bead is closed.
- **Service fragility.** Running two 8B-class servers (e.g. a temp A/B server +
  the daemon) can evict the daemon and even the Dolt server under memory
  pressure. Restore with `bash scripts/start-servers-manual.sh` and
  `gt dolt start`. Don't `kill` a llama-server without confirming which is :8080.
- **gt/bd specifics:** rig names can't contain hyphens (`hermes_metal`);
  `gt rig add` clones from the **remote** (push first); `gt mq list/reject` need
  the rig as first arg; no per-rig custom validation hook in 1.1.0 → the gate is
  enforced as the polecat's pre-`gt done` step + your `gt mq` approval.

## Reusing for the next slice (usability, accessibility)

Same loop; only the backlog and the gate's pass/fail axes change. For
usability/accessibility, swap the perf guardrails for the relevant checks (e.g.
REPL snapshot tests, axe a11y lint) and keep the perf guardrail as a
non-regression floor. See `docs/ROADMAP-gastown-slices.md`.

## Teardown

```sh
gt down                       # stop rig agents (witness/refinery/deacon)
gt dolt stop                  # stop the Dolt server (:3307)
# shell integration added a block to ~/.zshrc during `gt install` — remove if unwanted
```
