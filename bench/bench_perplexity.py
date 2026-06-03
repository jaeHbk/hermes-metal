"""
bench/bench_perplexity.py — accuracy guardrail for the throughput numbers.

A perf optimization that silently degrades quality is a regression. We measure
WikiText-2-raw test perplexity (the de-facto standard for LLM perplexity) on
a fixed token budget so the result is stable across runs.

Two paths:
  * llama_cpp  — invokes the upstream `llama-perplexity` binary built alongside
                 llama-server (third_party/llama.cpp/build/bin/llama-perplexity).
                 This is the canonical implementation and matches what the
                 GGUF community uses for cross-quant comparison.
  * mlx        — computes log-probability of every token under the MLX model
                 in-process via mlx-lm.

NOTE: To compare apples-to-apples, both backends must process the SAME tokens.
We feed the raw text to each backend and let each backend tokenize with its
own (different) tokenizer — this is the convention used by published
quantization perplexity tables. Differences from tokenizer drift are real and
small (<0.05 ppl on standard corpora).

Usage (from REPO_ROOT):
    .venv/bin/python -m bench.bench_perplexity --backend llama_cpp
    bench/.venv-mlx/bin/python -m bench.bench_perplexity --backend mlx
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from bench.common import (
    HostInfo, LLAMA_CPP, MLX, MLX_MODEL_ID, PerplexityRun, REPO_ROOT,
    write_result,
)

WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_SPLIT = "test"

LLAMA_PPL_BIN = REPO_ROOT / "third_party/llama.cpp/build/bin/llama-perplexity"
LLAMA_MODEL = REPO_ROOT / "models/hermes-8b-q4_k_m.gguf"


def _load_wikitext_text(max_chars: int) -> str:
    """Concatenate the public WikiText-2-raw test split. Cached by HF hub."""
    from datasets import load_dataset
    ds = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split=WIKITEXT_SPLIT)
    buf = []
    total = 0
    for row in ds:
        line = row["text"]
        if not line:
            continue
        buf.append(line)
        total += len(line)
        if total >= max_chars:
            break
    return "\n".join(buf)


# ---- llama.cpp via the upstream perplexity binary ---------------------------

def run_llama_cpp(text: str, ctx: int) -> PerplexityRun:
    if not LLAMA_PPL_BIN.exists():
        raise SystemExit(
            f"ERROR: {LLAMA_PPL_BIN} not built. "
            f"Run 'cmake --build third_party/llama.cpp/build --target llama-perplexity' "
            f"or `make build-engine` then retry."
        )
    if not LLAMA_MODEL.exists():
        raise SystemExit(f"ERROR: {LLAMA_MODEL} missing. Run 'make fetch-model'.")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
        corpus_path = f.name

    cmd = [
        str(LLAMA_PPL_BIN),
        "-m", str(LLAMA_MODEL),
        "-f", corpus_path,
        "-c", str(ctx),
        "--ppl-stride", "0",
        "-ngl", "99",        # offload everything we can to Metal
        "-fa", "on",
        "-ctk", "q8_0",
        "-ctv", "q8_0",
    ]
    print(f"  [llama.cpp] {' '.join(cmd)}", flush=True)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    wall = time.perf_counter() - t0
    out = proc.stdout + "\n" + proc.stderr

    # llama-perplexity prints lines like:
    #   [1]4.1234,[2]4.0987,...
    # and a final summary:  Final estimate: PPL = 6.1234 +/- 0.04567
    m = re.search(r"PPL\s*=\s*([0-9]+\.[0-9]+)", out)
    if not m:
        sys.stderr.write(out)
        raise SystemExit("ERROR: could not parse perplexity from llama-perplexity output")

    ppl = float(m.group(1))

    # Token count = number of [N] markers (chunks of `ctx` tokens each)
    chunks = len(re.findall(r"\[\d+\]", out))
    n_tokens = chunks * ctx

    return PerplexityRun(
        backend=LLAMA_CPP, dataset=f"{WIKITEXT_CONFIG}/{WIKITEXT_SPLIT}",
        n_tokens=n_tokens, perplexity=ppl, wall_s=wall,
    )


# ---- MLX via in-process forward pass ----------------------------------------

def run_mlx(text: str, ctx: int) -> PerplexityRun:
    """
    Sliding-window perplexity to mirror llama-perplexity's chunked computation.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    print(f"  [mlx] loading {MLX_MODEL_ID} ...", flush=True)
    model, tokenizer = load(MLX_MODEL_ID)

    ids = tokenizer.encode(text)
    n = len(ids)
    if n < ctx + 1:
        raise SystemExit(
            f"ERROR: corpus only {n} tokens; need at least {ctx + 1} for one chunk. "
            f"Increase --max-chars."
        )

    n_chunks = (n - 1) // ctx
    total_logp = 0.0
    total_t = 0

    t0 = time.perf_counter()
    for i in range(n_chunks):
        chunk = mx.array(ids[i * ctx : i * ctx + ctx + 1])[None, :]
        logits = model(chunk[:, :-1])
        targets = chunk[:, 1:]
        # Stable per-token log-prob via log_softmax(logits)
        logp = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
            reduction="sum",
        )
        total_logp += float(logp.item())
        total_t += ctx
        if (i + 1) % 4 == 0:
            print(f"    chunk {i + 1}/{n_chunks}", flush=True)

    wall = time.perf_counter() - t0
    avg_nll = total_logp / total_t
    ppl = math.exp(avg_nll)

    return PerplexityRun(
        backend=MLX, dataset=f"{WIKITEXT_CONFIG}/{WIKITEXT_SPLIT}",
        n_tokens=total_t, perplexity=ppl, wall_s=wall,
    )


# ---- driver ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["llama_cpp", "mlx", "both"], default="both")
    ap.add_argument("--ctx", type=int, default=512,
                    help="Perplexity chunk size in tokens (default 512, llama.cpp convention).")
    ap.add_argument("--max-chars", type=int, default=300_000,
                    help="Cap raw corpus chars to bound runtime (default 300k ≈ 60k–80k tokens).")
    args = ap.parse_args()

    text = _load_wikitext_text(args.max_chars)
    print(f"  corpus: {len(text):,} chars from {WIKITEXT_CONFIG}/{WIKITEXT_SPLIT}",
          flush=True)

    backends = ["llama_cpp", "mlx"] if args.backend == "both" else [args.backend]
    runs: list[PerplexityRun] = []

    for be in backends:
        if be == "llama_cpp":
            r = run_llama_cpp(text, args.ctx)
        else:
            r = run_mlx(text, args.ctx)
        print(f"  [{be}] PPL={r.perplexity:.4f}  n_tokens={r.n_tokens:,}  "
              f"wall={r.wall_s:.1f}s", flush=True)
        runs.append(r)

    payload = {
        "kind": "perplexity",
        "host": dataclasses.asdict(HostInfo()),
        "settings": {
            "ctx": args.ctx,
            "max_chars": args.max_chars,
            "dataset": f"{WIKITEXT_CONFIG}/{WIKITEXT_SPLIT}",
            "mlx_model": MLX_MODEL_ID,
            "llama_model": str(LLAMA_MODEL.name),
        },
        "runs": [dataclasses.asdict(r) for r in runs],
    }
    out = write_result("perplexity", payload)
    print(f"  wrote: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
