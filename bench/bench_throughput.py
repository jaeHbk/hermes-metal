"""
bench/bench_throughput.py — head-to-head throughput + memory benchmark.

Compares the two backends on identical prompts with identical sampling
settings (temperature=0, top_p=1, fixed seed, fixed max_new_tokens) and
records: TTFT, prefill tok/s, decode tok/s, peak RSS, wall time.

Backends:
  * llama_cpp  — talks to the locally-running llama-server (chat agent
                 installed by `make install-daemon`) over /completion,
                 which exposes per-request timing in the response body.
  * mlx        — uses mlx-lm in-process inside the bench/.venv-mlx venv.

Usage (from REPO_ROOT):
    .venv/bin/python -m bench.bench_throughput --backend llama_cpp
    bench/.venv-mlx/bin/python -m bench.bench_throughput --backend mlx
    bench/.venv-mlx/bin/python -m bench.bench_throughput --backend both

For `--backend both`, the script must run under .venv-mlx (mlx-lm import)
AND requires the llama-server daemon to be running on $HERMES_CHAT_URL.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from bench.common import (
    HostInfo, LLAMA_BASE_URL, LLAMA_CPP, MLX, MLX_MODEL_ID, ThroughputRun,
    load_prompts, write_result,
)

# ---- shared sampling params --------------------------------------------------

DECODE_DEFAULT = 256
WARMUP_TOKENS = 16
TIMEOUT_S = 600


def _peak_rss_bytes() -> int:
    """ru_maxrss on macOS is in BYTES (Linux returns KB). Documented quirk."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


# ---- llama.cpp backend (HTTP) ------------------------------------------------

def run_llama_cpp(prompt: str, decode_tokens: int, seed: int) -> ThroughputRun:
    """
    Hits llama-server's native /completion endpoint with stream=true so we can
    capture wall-clock TTFT ourselves. Llama-server also reports timings in the
    final stream chunk under the `timings` key (prompt_n, prompt_ms, predicted_n,
    predicted_ms) which we cross-check against our wall measurement.
    """
    import httpx  # daemon venv has it

    url = f"{LLAMA_BASE_URL}/completion"
    body = {
        "prompt": prompt,
        "n_predict": decode_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "stream": True,
        "cache_prompt": False,  # fair comparison: no prompt-cache reuse on the bench
    }

    proc = psutil.Process(os.getpid())
    rss_baseline = proc.memory_info().rss

    t0 = time.perf_counter()
    ttft = None
    timings: dict[str, Any] = {}
    n_decoded = 0

    with httpx.Client(timeout=TIMEOUT_S) as client:
        with client.stream("POST", url, json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if ttft is None and chunk.get("content"):
                    ttft = time.perf_counter() - t0
                if chunk.get("stop") and "timings" in chunk:
                    timings = chunk["timings"]
                if chunk.get("content"):
                    n_decoded += 1  # rough; refined by server timings below

    wall = time.perf_counter() - t0

    # llama-server's reported timings are authoritative for tok/s
    # (it counts BPE tokens; our streaming chunk count is per-text-piece).
    prompt_n = int(timings.get("prompt_n", 0))
    prompt_ms = float(timings.get("prompt_ms", 0.0))
    predicted_n = int(timings.get("predicted_n", n_decoded))
    predicted_ms = float(timings.get("predicted_ms", 0.0))

    prefill_s = prompt_ms / 1000.0 if prompt_ms else 0.0
    decode_s = predicted_ms / 1000.0 if predicted_ms else 0.0
    prefill_tps = (prompt_n / prefill_s) if prefill_s > 0 else 0.0
    decode_tps = (predicted_n / decode_s) if decode_s > 0 else 0.0

    rss_peak = max(proc.memory_info().rss, rss_baseline)

    return ThroughputRun(
        backend=LLAMA_CPP,
        prompt_id="",
        input_tokens=prompt_n,
        output_tokens=predicted_n,
        ttft_s=ttft if ttft is not None else wall,
        prefill_s=prefill_s,
        decode_s=decode_s,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        peak_rss_bytes=rss_peak,
        wall_s=wall,
    )


# ---- MLX backend (in-process) ------------------------------------------------

_mlx_state: dict[str, Any] = {"model": None, "tokenizer": None}


def _ensure_mlx_loaded() -> None:
    """Lazy-load so importing this module doesn't pull in MLX on the daemon side."""
    if _mlx_state["model"] is not None:
        return
    from mlx_lm import load
    print(f"  [mlx] loading {MLX_MODEL_ID} ...", flush=True)
    t = time.perf_counter()
    model, tokenizer = load(MLX_MODEL_ID)
    print(f"  [mlx] loaded in {time.perf_counter() - t:.1f}s", flush=True)
    _mlx_state["model"] = model
    _mlx_state["tokenizer"] = tokenizer


def run_mlx(prompt: str, decode_tokens: int, seed: int) -> ThroughputRun:
    """
    Uses mlx_lm.stream_generate so we can capture TTFT and token-by-token
    decode wall-time. mlx-lm 0.21+ exposes per-token GenerationResponse with
    prompt_tps and generation_tps fields, but we still measure wall externally
    so the comparison is backend-agnostic.
    """
    _ensure_mlx_loaded()
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    model = _mlx_state["model"]
    tokenizer = _mlx_state["tokenizer"]

    proc = psutil.Process(os.getpid())
    rss_baseline = proc.memory_info().rss

    mx.random.seed(seed)
    sampler = make_sampler(temp=0.0, top_p=1.0)

    prompt_tokens = tokenizer.encode(prompt)
    input_n = len(prompt_tokens)

    t0 = time.perf_counter()
    ttft = None
    last_tps_prefill = 0.0
    last_tps_decode = 0.0
    n_out = 0
    rss_peak = rss_baseline

    for resp in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=decode_tokens, sampler=sampler
    ):
        if ttft is None:
            ttft = time.perf_counter() - t0
        n_out += 1
        # mlx-lm exposes prompt_tps and generation_tps on GenerationResponse
        last_tps_prefill = float(getattr(resp, "prompt_tps", last_tps_prefill))
        last_tps_decode = float(getattr(resp, "generation_tps", last_tps_decode))
        # cheap periodic RSS poll (every 32 tokens)
        if n_out % 32 == 0:
            rss_peak = max(rss_peak, proc.memory_info().rss)

    wall = time.perf_counter() - t0
    rss_peak = max(rss_peak, proc.memory_info().rss, _peak_rss_bytes())

    prefill_s = (input_n / last_tps_prefill) if last_tps_prefill > 0 else 0.0
    decode_s = (n_out / last_tps_decode) if last_tps_decode > 0 else 0.0

    return ThroughputRun(
        backend=MLX,
        prompt_id="",
        input_tokens=input_n,
        output_tokens=n_out,
        ttft_s=ttft if ttft is not None else wall,
        prefill_s=prefill_s,
        decode_s=decode_s,
        prefill_tps=last_tps_prefill,
        decode_tps=last_tps_decode,
        peak_rss_bytes=rss_peak,
        wall_s=wall,
    )


# ---- driver ------------------------------------------------------------------

BACKENDS = {
    LLAMA_CPP: run_llama_cpp,
    MLX: run_mlx,
}


def warmup(backend: str, decode_tokens: int = WARMUP_TOKENS) -> None:
    """Discard the first run on each backend to absorb model-load + JIT costs."""
    print(f"  [warmup:{backend}]", flush=True)
    BACKENDS[backend]("Hello.", decode_tokens, seed=0)


def bench_one(
    backend: str, prompt_id: str, prompt: str, decode_tokens: int, seed: int,
) -> ThroughputRun:
    run = BACKENDS[backend](prompt, decode_tokens, seed)
    run.prompt_id = prompt_id
    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["llama_cpp", "mlx", "both"], default="both")
    ap.add_argument("--decode-tokens", type=int, default=DECODE_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-id", default=None,
                    help="Run only this prompt id (default: all).")
    args = ap.parse_args()

    backends = ["llama_cpp", "mlx"] if args.backend == "both" else [args.backend]

    cfg = load_prompts()
    prompts = cfg["prompts"]
    if args.prompt_id:
        prompts = [p for p in prompts if p["id"] == args.prompt_id]
        if not prompts:
            print(f"ERROR: no prompt with id {args.prompt_id!r}", file=sys.stderr)
            return 2

    seed = args.seed
    decode_tokens = args.decode_tokens

    runs: list[ThroughputRun] = []

    for be in backends:
        warmup(be)
        for p in prompts:
            print(f"  [run:{be}] {p['id']}  (input≈{p['approx_input_tokens']}t)",
                  flush=True)
            r = bench_one(be, p["id"], p["text"], decode_tokens, seed)
            print(
                f"    ttft={r.ttft_s*1000:.0f}ms  "
                f"prefill={r.prefill_tps:.1f}t/s  decode={r.decode_tps:.1f}t/s  "
                f"rss={r.peak_rss_bytes/1024/1024:.0f}MiB  "
                f"wall={r.wall_s:.1f}s",
                flush=True,
            )
            runs.append(r)

    payload = {
        "kind": "throughput",
        "host": dataclasses.asdict(HostInfo()),
        "settings": {
            "decode_tokens": decode_tokens,
            "seed": seed,
            "temperature": 0.0,
            "top_p": 1.0,
            "llama_url": LLAMA_BASE_URL,
            "mlx_model": MLX_MODEL_ID,
        },
        "runs": [dataclasses.asdict(r) for r in runs],
    }
    out = write_result("throughput", payload)
    print(f"  wrote: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
