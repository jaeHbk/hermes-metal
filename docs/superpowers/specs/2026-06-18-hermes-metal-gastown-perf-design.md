# Design — Gastown-driven performance slice for hermes-metal

Status: **approved (brainstorm)** · Date 2026-06-18 · Owner: jaehunb
Branch: `feat/phases-b-e` · Repo: github.com/jaeHbk/hermes-metal

Roadmap for later slices: [`../../ROADMAP-gastown-slices.md`](../../ROADMAP-gastown-slices.md)

---

## 1. Goal & framing

Two **co-equal deliverables**:

1. A reusable **Gastown multi-agent harness** running on this repo (a real,
   reusable outcome — not just a means).
2. Measurable **performance gains** in hermes-metal, used as the proving ground.

This spec covers **only the performance slice**. Usability and accessibility are
deliberately deferred and parked in the roadmap doc; they will reuse this same
harness.

"Seamless" was specified in three senses, all in scope:
- **Operator:** one command (`gt mayor attach`) launches the swarm; you only
  review/approve merges.
- **Integration:** Gastown plugs into the repo's *existing* `make` / `pytest` /
  git / PR workflow — no bespoke scaffolding leaks into the project shape.
- **Product:** the end result is a faster `hermes`.

### Non-negotiable guardrail

hermes-metal chose llama.cpp over MLX **on purpose**: ~6× lower peak RSS and
materially better perplexity at the same 4 bpw — the right axes for an
always-on background daemon. Therefore the performance goal is:

> **Improve decode and/or prefill tok/s WITHOUT regressing peak RSS or
> perplexity** beyond tolerance.

Any "improvement" that trades away RSS or quality (e.g. swapping in MLX) is **out
of scope** and would be correctly rejected by the gate. This is a product-quality
floor, not a preference.

### Hard constraints

- **Apple-Silicon-only.** Metal-compiled llama.cpp (`GGML_METAL=1`), `sysctl`
  tier detection, `launchd` agents, arm64 guard in the Makefile. All
  build/test/bench agents **must run on this Mac**, not the Linux cloud desktop.
- **Gastown is already installed** (`gt`/`gastown` 1.1.0 via Homebrew; Go 1.26,
  Node 18 present). Setup is configuration, not installation.

---

## 2. Baseline (measured, M3 Pro / 36 GiB, from `bench/results/REPORT.md`)

| prompt | decode t/s (llama.cpp) | decode t/s (MLX) | prefill t/s (llama.cpp) | peak RSS |
| --- | --- | --- | --- | --- |
| short_qa | 23.8 | 29.9 | 199.8 | 182 MiB |
| medium_summary | 19.3 | 28.8 | 321.9 | 251 MiB |
| long_context | 24.4 | 28.7 | 320.1 | 298 MiB |

Perplexity (WikiText-2): llama.cpp **8.40** vs MLX 11.72. MLX leads decode by
~+20–50%; llama.cpp wins RSS ~6× and perplexity decisively. The slice targets
that decode/prefill gap from the llama.cpp side, holding RSS and ppl.

---

## 3. Architecture & the stack

Three layers, matching Gastown's own model, with **one new repo-side
component** (the bench-gate).

```
Gastown (gt 1.1.0)         Mayor → decompose backlog
                           Polecats → 1 experiment each, isolated git worktree
                           Refinery → merge queue (serialized)
                           Witness → stuck/loop detection
                           Overseer = YOU → gated merge approval
Beads (.beads/, git-backed) the perf-experiment backlog (1 bead per experiment)
hermes-metal repo          NEW: scripts/bench_gate.sh (definition-of-done)
                           existing: bench/, pytest (233), Makefile, llama.cpp submodule
                           ── all on this Mac (Apple Silicon) ──
```

**Central idea — the bench-gate is the contract.** Every polecat must pass
`scripts/bench_gate.sh --full` before it may open a PR. The gate is what makes
all three "seamless" senses real at once and mechanically enforces the
guardrail. It is also reusable by the later slices and outlives Gastown.

---

## 4. The bench-gate (the one new component)

`scripts/bench_gate.sh` — one shell script, stdlib + existing `bench/` modules,
**no new dependencies**, lives in `scripts/` (orchestration glue, not product
code, so it does not ship inside the `hermes` package or count against its test
surface).

```
bench_gate.sh [--quick|--full] [--baseline bench/results/baseline.json]
  exit 0 → PASS  (safe to open PR)
  exit 1 → FAIL  (correctness or guardrail violation — must not PR)
  exit 2 → ERROR (could not run; treated as fail)
  always writes bench/results/gate-<sha>.json (machine-readable delta)
```

| Phase | Action | Fail condition |
| --- | --- | --- |
| 1. Correctness | `pytest -q` (all 233) | any test fails |
| 2. Build | `make build-engine` if submodule/flags touched | build error |
| 3. Throughput | `bench/bench_throughput.py` on the 3 fixed prompts | measures only |
| 4. Quality guardrail | `bench/bench_perplexity.py` | ppl > baseline + 0.1 |
| 5. Memory guardrail | peak RSS from throughput run | RSS > baseline × 1.10 |
| 6. Verdict | write delta JSON: Δdecode/Δprefill t/s, ΔRSS, Δppl | exit per above |

**Two modes** (perplexity is ~150s):
- `--quick` — tests + short throughput + RSS (~30s). What a polecat runs *while
  iterating*.
- `--full` — adds perplexity. **Mandatory pre-PR gate**, and what Refinery
  re-runs before queueing a merge.

**Baseline seeding.** Commit `bench/results/baseline.json` from current
REPORT.md numbers (decode 19.3–24.4 t/s by prompt; RSS ≤ 298 MiB; ppl 8.40) as
the reference. Each accepted merge that improves numbers **updates the
baseline** (later experiments must beat the new bar) and appends a row to
`bench/results/history.tsv` — delivering the "versioned benchmark history"
roadmap item as a side effect.

**Tolerances (explicit, to remove ambiguity):**
- Perplexity regression threshold: **> baseline + 0.10** fails (REPORT.md calls
  < 0.1 tokenizer-noise).
- RSS regression threshold: **absolute ceiling**, default **1024 MiB** on the
  36 GiB Pro tier (see the amendment below).
- Throughput is *not* a pass/fail axis by itself — it is the *objective*. A
  bead that passes guardrails but shows no throughput gain is reported and left
  for the human to accept or reject (it may still be worth landing, e.g. a
  submodule bump that's neutral now but unblocks later work).

> **AMENDMENT (2026-06-18, discovered by the E1-readiness spike — RSS is mmap-noisy).**
> The original design gated RSS on a relative `× 1.10` factor. The spike proved
> that unworkable: llama.cpp **mmaps** the GGUF (default on), so the benchmark's
> sampled peak RSS reflects only the *currently-resident* page set, which (a)
> never approaches the true ~4.7 GB weight footprint (the live server shows
> ~140 MiB RSS) and (b) varies run-to-run by **several-fold** (observed
> 37 → 154 → 294 MiB for the same prompt) as pages fault in lazily. A ×1.10
> relative gate therefore fires on pure measurement noise, not regressions.
> **Resolution:** the gate uses an **absolute** RSS ceiling for ALL beads. The
> `compare()` function already supports this via `rss_ceiling_mib`; the
> orchestrator (`scripts/bench_gate.sh`) supplies a tier default of **1024 MiB**
> when a bead sets none. 1024 MiB is comfortably above the observed noisy peak
> (<300 MiB for the base config) yet far below MLX's ~1.8 GiB path, so it still
> catches a genuine memory balloon. E5 (speculative decoding) overrides this to
> a *tighter* **700 MiB** via `BENCH_GATE_RSS_CEILING_MIB`, since its resident
> draft model is the one deliberate memory trade and deserves the closer watch.
> The relative `× rss_factor` code path is retained in `compare()` (still unit-
> tested) but is no longer the gate's default.

**Per-bead RSS budget (resolves the E5 tension).** A few experiments — chiefly
E5 speculative decoding — trade memory *for* decode speed by design; a resident
draft model blows past `baseline × 1.10`, so the default RSS gate would
`exit 1` and the human would never see the speed win. To keep the human in the
loop without weakening the floor for everyone else, the gate reads an optional
per-bead `rss_ceiling_mib` (absolute) from the bead metadata:
- If set, the RSS axis uses that **absolute background-friendly ceiling**
  instead of the relative `× 1.10` rule.
- The ceiling is still a hard floor (the daemon must stay background-light): set
  per hardware tier — e.g. **≤ 700 MiB** on the 36 GiB Pro tier — so even a
  memory/speed trade cannot make hermes behave like the 1.8 GiB MLX path.
- Beads without `rss_ceiling_mib` use the default `× 1.10` rule. E5 is the only
  backlog item expected to set it; the human approves the ceiling when seeding
  the bead, so "human judgment" happens at backlog time *and* at merge time.

---

## 5. Performance experiment backlog (Beads)

One bead → one polecat → one worktree → one PR. Deliberately independent (no
shared-file contention); judged purely by the bench-gate.

| # | Experiment | Hypothesis | Decode/prefill ceiling | Guardrail risk |
| --- | --- | --- | --- | --- |
| **E1** | Verify `-fa` + `--prompt-cache` are live in the running agents | CLAUDE.md mandates both; running LaunchAgent flags may have drifted — free win if so | low–med | none (audit) |
| **E2** | Thread tuning (`--threads` / `--threads-batch` vs detected P/E split) | Prefill is compute-bound; match `threads-batch` to P-cores | med (prefill) | none |
| **E3** | Batch / ubatch (`-b`, `-ub`) for prefill | Larger ubatch lifts long-context prefill | med (prefill) | RSS ↑ (gate catches) |
| **E4** | KV-cache quantization (`--cache-type-k/v q8_0`) | Frees memory headroom; enables bigger batch; ~0 quality cost at q8 | low direct; enables others | ppl (gate catches) |
| **E5** | Speculative decoding w/ tiny draft model (~0.5–1B) | **Biggest decode lever** — the axis MLX leads; draft proposes, 8B verifies | **high (decode)** | **RSS ↑** (resident draft) — gate + human judgment |
| **E6** | llama.cpp submodule bump | Newer Metal kernels often raise decode/prefill for free | med, broad | build/API drift; gate + tests catch |
| **E7** | Quant variant probe (Q4_K_M → IQ4_XS / Q5_K_M) | Map speed/quality/RSS frontier; may find a better operating point | med | explicit tradeoff — likely **informational**, not a merge |

**Order (spike-first):**
1. **E1 is the spike** — lowest risk, audit-style; it forces the entire loop
   (worktree → change a flag → `bench_gate --full` → PR → gated merge → baseline
   update) to work end-to-end on something safe.
2. After E1 lands clean: fan out **E2–E4** (safe parallel batch).
3. Then **E5–E6** (higher ceiling, gated more carefully).
4. **E7** as an informational probe that likely reports rather than merges.

**Honesty notes (carried into the plan):**
- These are *hypotheses*. The gate decides; some will regress and be rejected —
  that is the system working, not failing. No specific tok/s number is promised.
- **E5 closes the real decode gap** but is the one that can cost RSS — exactly
  the quality the project exists to protect. It is intentionally **not** the
  spike. Because a resident draft model exceeds the default `× 1.10` RSS gate by
  design, E5's bead carries an explicit `rss_ceiling_mib` (the absolute
  background-friendly ceiling from §4); the human sets that ceiling when seeding
  the bead and approves the trade with ΔRSS visible at merge.

---

## 6. Operator workflow

**One-time setup (run once, ~5 min):**
```sh
gt rig add hermes-metal ~/Documents/hermes-metal
bd init && bd onboard
# seed E1–E7 as beads; commit baseline.json + bench_gate.sh on a setup branch
gt config set merge.gate "scripts/bench_gate.sh --full"
gt config set merge.policy gated      # human approves every merge
```
(Exact `gt`/`bd` subcommand spelling to be confirmed against installed 1.1.0
during planning; intent is fixed.)

**Daily loop:**
```sh
gt mayor attach     # one command — tmux cockpit
```
tmux shows the Mayor assigning beads, each polecat's live progress, Witness
flags, the Refinery queue. The only required human action is the **gate**:

```
MERGE REQUEST: E2 thread-tuning
  tests 233/233 ✓   build ✓
  decode +2.1 t/s (19.3→21.4)   prefill +18 t/s ✓
  RSS +4 MiB (251→255, within +10%) ✓
  ppl +0.02 (8.40→8.42, noise) ✓
  diff: config/engine_flags.env (+3 −2)
  [a]pprove  [r]eject  [d]iff  [h]old
```
Approve → Refinery merges to integration branch `perf/gastown-run-1`, updates
baseline, appends `history.tsv`.

**Boundaries (deliberate):**
- Worktrees live **outside** the repo (`~/gt/worktrees/…`); your checkout is
  never disturbed (verified by `git status` staying clean).
- Merges land on **`perf/gastown-run-1`**, never `main` or `feat/phases-b-e`
  unattended.
- **Concurrency cap** (each polecat loads an 8B for bench = real RAM/CPU):
  default **3** on the 36 GiB M3 Pro; confirmed against host before launch.
- Stop anytime: `gt mayor detach` parks state in Beads (git-backed);
  `gt mayor attach` resumes. Nothing lost.

---

## 7. Validation & definition of done

**A. Harness validation — proven by the E1 spike before any fan-out:**
- `gt mayor attach` launches; Mayor assigns E1 to a polecat in an isolated
  worktree (checkout untouched).
- Polecat runs `bench_gate.sh --full`, opens a PR; Refinery shows the gate card;
  human approves.
- Merge lands on `perf/gastown-run-1`; `baseline.json` + `history.tsv` update.
- **Negative test:** seed one throwaway bead with a knowingly-bad change (drop
  `-fa`) and confirm the gate **rejects** it (regression → exit 1 → no PR). A
  safety net not observed catching something is not yet trusted.
- If the spike loop or negative test fails → **stop and reconsider** (fall back
  to plain Kiro/meshclaw subagent orchestration). That is the spike's job.

**B. Per-experiment validation:** the bench-gate IS the test — 233 green, ppl ≤
baseline + 0.1, RSS ≤ baseline × 1.10, tok/s delta recorded. No bead merges on
green-tests-alone; guardrails are mandatory.

**C. Whole-slice definition of done:**
- Harness reproducible (`gt mayor attach` → swarm → gated merges) and documented
  in `docs/gastown-harness.md` for reuse by later slices.
- A net **decode and/or prefill tok/s improvement** on `perf/gastown-run-1` with
  **zero RSS or perplexity regression** vs the seeded baseline, captured in
  `history.tsv` and a refreshed `bench/results/REPORT.md`. **No specific number
  promised**; the gate reports the honest delta and rejects regressions.
- A final cumulative PR `perf/gastown-run-1` → `feat/phases-b-e` that the
  **human merges by hand**, plus an `IMPROVEMENTS.md` entry in the repo's
  problem/change/impact format.

---

## 8. Out of scope (this slice) — see roadmap

- **Usability/features** and **accessibility** slices — reuse this harness;
  parked in [`../../ROADMAP-gastown-slices.md`](../../ROADMAP-gastown-slices.md).
- **Heavy operator dashboard** (Approach B) — YAGNI until thin glue proves
  insufficient. (Versioned-bench-history piece ships here via `history.tsv`.)
- **Cross-platform / MLX backend swap** — conflicts with the guardrail; only
  revisit as a deliberate product-direction change, never as "performance."

---

## 9. New / touched artifacts

| Path | New? | Purpose |
| --- | --- | --- |
| `scripts/bench_gate.sh` | new | definition-of-done gate (the contract) |
| `bench/results/baseline.json` | new | seeded reference numbers |
| `bench/results/history.tsv` | new | versioned benchmark history (append-only) |
| `.beads/` | new | Gastown/Beads backlog (E1–E7) |
| `docs/gastown-harness.md` | new | how to run/reuse the harness |
| `config/engine_flags.env` | touched (by experiments) | flag changes (E1–E3) |
| `third_party/llama.cpp` submodule | touched (E6) | version bump |
| `models/` | touched (E5/E7) | draft / variant model fetch |
| `IMPROVEMENTS.md` | touched (end) | slice writeup |

Worktrees, integration branch, and gating ensure none of these touch `main` or
`feat/phases-b-e` until the human merges the final cumulative PR.
