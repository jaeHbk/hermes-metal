# bench/

Head-to-head benchmark of `hermes-metal`'s **llama.cpp + Metal + Q4_K_M**
stack against **MLX 4-bit** running the same Hermes-3-Llama-3.1-8B model.

The project's pitch is "minimize memory and battery while preserving
accuracy on Apple Silicon," so we measure all three:

| What                  | How                                          |
| --------------------- | -------------------------------------------- |
| Throughput            | `bench_throughput.py` — TTFT, prefill t/s, decode t/s, peak RSS |
| Accuracy guardrail    | `bench_perplexity.py` — WikiText-2-raw test perplexity |
| Battery / energy cost | `bench_power.sh` — `powermetrics` + J / 1k tokens |

Both backends run the same prompts (`prompts.json`) at `temperature=0`,
`seed=42`, fixed `max_new_tokens` so deltas are reproducible.

## One-time setup

```sh
make setup-venv         # daemon's runtime venv (if not already built)
bench/setup.sh          # creates bench/.venv-mlx, downloads MLX model (~5 GB)
```

The MLX side lives in a **separate** `bench/.venv-mlx` so its `mlx` and
`transformers` deps cannot drift the daemon's `.venv`.

## Run

The llama.cpp side requires the chat daemon to be up
(`make install-daemon` once, then `make start-daemon`).

```sh
# Throughput, both backends:
bench/.venv-mlx/bin/python -m bench.bench_throughput --backend both

# Or run separately if you want to keep MLX out of one process:
.venv/bin/python              -m bench.bench_throughput --backend llama_cpp
bench/.venv-mlx/bin/python    -m bench.bench_throughput --backend mlx

# Accuracy:
.venv/bin/python              -m bench.bench_perplexity --backend llama_cpp
bench/.venv-mlx/bin/python    -m bench.bench_perplexity --backend mlx

# Power (each backend, sudo):
sudo bench/bench_power.sh llama_cpp medium_summary
sudo bench/bench_power.sh mlx       medium_summary

# Aggregate the latest of each into one Markdown report:
.venv/bin/python -m bench.aggregate
# -> bench/results/REPORT.md
```

Or via the Makefile:

```sh
make bench              # throughput + perplexity (no sudo, no power)
make bench-power        # asks for sudo, runs the power suite
make bench-report       # regenerate REPORT.md from existing JSON
```

## What the numbers mean

* **TTFT** (time-to-first-token) — latency the user actually feels. Dominated
  by prefill on long prompts.
* **prefill t/s** — how fast the model ingests the input. MLX's matmul kernels
  are excellent here; llama.cpp Metal is competitive with Flash Attention on.
* **decode t/s** — generation speed. Largely bandwidth-bound for a 4-bit 8B
  model on UMA; expect ~30–60 t/s on M-series Pro/Max.
* **peak RSS** — process resident set. The KV cache is the bulk of this on
  long contexts; llama.cpp's q8_0 KV cache is the lever for the memory claim.
* **perplexity** — lower is better. **A delta > 0.5 between backends is a
  real quality regression** (i.e., the perf win came at a cost). Within ~0.1
  is normal tokenizer/quant-scheme noise.
* **J / 1k tokens** — backend-agnostic battery cost. Whether you're on AC or
  battery, this is the number that determines how much your daemon drains
  per useful answer.

## Caveats

* Q4_K_M ≠ MLX 4-bit affine. Both are ~4 bits-per-weight but use different
  schemes (K-quants super-blocks vs group-wise affine), so a small
  perplexity delta is expected.
* `powermetrics` requires sudo and must be the only consumer of the SMC
  power-rail counters during the run.
* Don't run benchmarks while another GPU-heavy app (Xcode, Final Cut, a
  game) is active — Metal contention will skew prefill numbers more than
  decode.
* First run on each backend is a warmup (discarded) to absorb JIT / Metal
  shader compile / model load.
