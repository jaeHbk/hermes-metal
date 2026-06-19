# Roadmap — Gastown improvement slices

Status: **planned** · Created 2026-06-18 · Owner: jaehunb

This file parks work that is **deliberately out of scope for the first
(performance) slice** but that we intend to pick up in the near future. The
first slice is specced in
[`superpowers/specs/2026-06-18-hermes-metal-gastown-perf-design.md`](superpowers/specs/2026-06-18-hermes-metal-gastown-perf-design.md).

The premise across all slices: a **Gastown multi-agent harness** (Mayor →
polecats in isolated git worktrees → Refinery merge queue → gated merges by the
human Overseer) drives the work, and a reusable **bench/quality gate** is the
definition-of-done every polecat must pass before opening a PR. Each slice below
**reuses that same harness** — only the backlog and the gate's pass/fail axes
change.

---

## Slice 2 — Usability / features

**Goal:** make `hermes` more capable and pleasant to use day-to-day.

**Why deferred:** features tend to touch shared modules (`repl.py`, `cli.py`,
`wiki.py`), so they create more merge contention than the perf experiments —
a harder test of the harness. Better to prove the harness on the independent
perf backlog first.

**Candidate backlog (refine when we start this slice):**
- Richer REPL (better history search, inline citations UX, command discovery).
- Onboarding polish (`hermes init` wizard, first-run vault detection).
- New commands surfaced from the `## Proposed` section of
  [`../IMPROVEMENTS.md`](../IMPROVEMENTS.md): **Watcher observability
  (`hermes status --watch`)**, **File-scoped queries (`hermes ask --file`)**,
  **Reranker tuning + optional cross-encoder**, **Embedding model
  agnosticism**. (Cross-referenced, not duplicated — those entries are the
  source of truth for their own sketches/risks.)

**Gate axes for this slice:** correctness (pytest) + UX-regression checks
(e.g. REPL snapshot tests, `--help` completeness) rather than tok/s. The perf
guardrail still applies as a non-regression floor.

---

## Slice 3 — Accessibility

**Goal:** make the CLI/REPL and the bundled web UI usable with assistive tech.

**Why deferred:** well-bounded and independent (good harness fit), but smaller
in scope and lower urgency than perf/usability — a gentle later slice.

**Candidate backlog:**
- Screen-reader-friendly streaming output (avoid mid-token ANSI churn that
  breaks readers; a `--plain`/line-buffered mode).
- Colorblind-safe palette + honor `NO_COLOR` / `--no-color` everywhere.
- ARIA / semantic-HTML pass on llama.cpp's bundled web UI
  (`third_party/llama.cpp/tools/ui/`) — note this is **submodule code**, so
  upstream first or carry a documented patch; don't fork silently.
- Keyboard-only navigation audit of the web UI.

**Gate axes:** automated a11y linting where possible (e.g. axe on the web UI),
plus a manual screen-reader checklist. Perf guardrail still applies.

---

## Deferred tooling (not a slice — pull in if/when it pays for itself)

- **Heavy operator dashboard** (was "Approach B" in the perf brainstorm): perf
  dashboard, auto-generated experiment matrix, Witness tuned for stuck builds,
  auto-rollback. Deferred as YAGNI until the thin-glue harness proves
  insufficient. Note: the "versioned benchmark history" piece of this
  (`history.tsv` + sparkline) is being delivered *inside* the perf slice, so
  it's already partly done.

---

## Explicitly NOT planned (tension flagged)

- **Cross-platform support / MLX backend swap.** This conflicts with the
  perf-slice guardrail (llama.cpp is chosen precisely for 6× lower RSS and
  better perplexity as an always-on daemon). It also exists as a `## Proposed`
  entry in [`../IMPROVEMENTS.md`](../IMPROVEMENTS.md). If we ever revisit it,
  it must be framed as a *deliberate product-direction change*, not a
  "performance improvement" — otherwise the gate would (correctly) reject it.

---

_When starting a slice: copy its backlog into Beads, set the gate axes, and run
the same `gt mayor attach` loop. Each slice gets its own spec under
`docs/superpowers/specs/`._
