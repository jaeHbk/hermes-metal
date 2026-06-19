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
