"""
bench/aggregate.py — merge the most recent throughput / perplexity / power
JSON outputs into a single human-readable Markdown report.

Run after `make bench` (or after individual benches) to get a comparison
table. Defaults to picking the newest file of each kind in bench/results/.

Usage:
    python3 -m bench.aggregate                # newest of each kind
    python3 -m bench.aggregate --out report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.common import RESULTS_DIR


def _newest(pattern: str) -> Path | None:
    paths = sorted(RESULTS_DIR.glob(pattern))
    return paths[-1] if paths else None


def _load(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(path.read_text())


def _fmt_mib(b: int) -> str:
    return f"{b / 1024 / 1024:,.0f} MiB"


def _by_backend(runs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in runs:
        out.setdefault(r["backend"], []).append(r)
    return out


def _md_throughput(payload: dict) -> str:
    if not payload:
        return "_(no throughput run found)_\n"
    runs = payload["runs"]
    by_be = _by_backend(runs)
    backends = sorted(by_be.keys())
    prompt_ids = []
    for be in backends:
        for r in by_be[be]:
            if r["prompt_id"] not in prompt_ids:
                prompt_ids.append(r["prompt_id"])

    lines = ["## Throughput\n"]
    host = payload["host"]
    lines.append(
        f"_Host: {host['chip']}, {host['ram_gib']} GiB RAM, "
        f"{host['pcores']}P+{host['ecores']}E cores, macOS {host['macos']}_\n"
    )
    settings = payload["settings"]
    lines.append(
        f"_Settings: max_new_tokens={settings['decode_tokens']}, "
        f"temperature={settings['temperature']}, seed={settings['seed']}_\n\n"
    )

    header = ["prompt", "backend", "input_t", "output_t", "ttft (ms)",
              "prefill t/s", "decode t/s", "peak RSS"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for pid in prompt_ids:
        for be in backends:
            row = next((r for r in by_be[be] if r["prompt_id"] == pid), None)
            if not row:
                continue
            lines.append("| " + " | ".join([
                pid, be, str(row["input_tokens"]), str(row["output_tokens"]),
                f"{row['ttft_s'] * 1000:.0f}",
                f"{row['prefill_tps']:.1f}",
                f"{row['decode_tps']:.1f}",
                _fmt_mib(row["peak_rss_bytes"]),
            ]) + " |")

    return "\n".join(lines) + "\n\n"


def _md_perplexity(payload: dict) -> str:
    if not payload:
        return "_(no perplexity run found)_\n"
    lines = ["## Perplexity (WikiText-2-raw test)\n"]
    settings = payload["settings"]
    lines.append(
        f"_Chunk size: {settings['ctx']} tokens; "
        f"corpus cap: {settings['max_chars']:,} chars._\n\n"
    )
    lines.append("| backend | n_tokens | perplexity | wall (s) |")
    lines.append("| --- | --- | --- | --- |")
    for r in payload["runs"]:
        lines.append("| " + " | ".join([
            r["backend"], f"{r['n_tokens']:,}",
            f"{r['perplexity']:.4f}", f"{r['wall_s']:.1f}",
        ]) + " |")
    lines.append("\n_Lower perplexity is better. A delta < 0.1 between backends "
                 "is within tokenizer-noise; > 0.5 is a real quality regression._\n\n")
    return "\n".join(lines)


def _md_power(power_files: list[Path]) -> str:
    if not power_files:
        return "_(no power runs found — `sudo bench/bench_power.sh ...` to add)_\n"
    rows = []
    for p in power_files:
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            continue
    if not rows:
        return "_(no power runs found)_\n"

    lines = ["## Power (powermetrics integration)\n"]
    lines.append("| backend | prompt | output_t | total energy (J) | mWh | J / 1k tokens |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append("| " + " | ".join([
            r["backend"], r["prompt_id"], str(r["output_tokens"]),
            f"{r['energy_total_j']:.1f}",
            f"{r['energy_total_wh'] * 1000:.1f}",
            f"{r['joules_per_1k_output_tokens']:.2f}",
        ]) + " |")
    lines.append(
        "\n_J/1k-tokens is the backend-agnostic battery cost: lower is better. "
        "Wh ≈ mAh × V / 1000; an MBP battery is ~70 Wh._\n\n"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS_DIR / "REPORT.md"))
    ap.add_argument("--throughput", default=None)
    ap.add_argument("--perplexity", default=None)
    args = ap.parse_args()

    thr = _load(Path(args.throughput) if args.throughput else _newest("throughput-*.json"))
    ppl = _load(Path(args.perplexity) if args.perplexity else _newest("perplexity-*.json"))
    power_files = sorted(RESULTS_DIR.glob("power-*.json"))

    parts = [
        "# hermes-metal benchmark report\n",
        "Generated by `bench/aggregate.py`. Same Hermes-3-Llama-3.1-8B at 4 bpw "
        "on both backends; llama.cpp Q4_K_M GGUF vs MLX 4-bit affine.\n\n",
        _md_throughput(thr),
        _md_perplexity(ppl),
        _md_power(power_files),
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts))
    print(f"  wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
