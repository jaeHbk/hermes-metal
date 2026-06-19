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


def make_baseline(throughput_path: str | Path, perplexity_path: str | Path) -> dict[str, Any]:
    host = json.loads(Path(throughput_path).read_text()).get("host", {})
    return {
        "host": host,
        "prompts": load_throughput(throughput_path),
        "perplexity": load_perplexity(perplexity_path),
    }


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
                       f"perplexity regression: {current_ppl:.4f} > {base_ppl + ppl_tol:.4f}",
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
                               f"> {int(b['peak_rss_bytes'] * rss_factor)} (baseline x{rss_factor})",
                               deltas, current_ppl, base_ppl, max_rss)

    return Verdict(True, improved, "ok", deltas, current_ppl, base_ppl, max_rss)


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
