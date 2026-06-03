"""
bench/common.py — shared utilities for the hermes-metal benchmark harness.

Both throughput and perplexity scripts import from here so that the two
backends (llama.cpp via the running daemon, MLX via mlx-lm) are exercised
through identical I/O paths and parameters.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---- paths -------------------------------------------------------------------

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"
PROMPTS_PATH = BENCH_DIR / "prompts.json"

# ---- backend identifiers -----------------------------------------------------

LLAMA_CPP = "llama_cpp"
MLX = "mlx"

# Endpoint of the locally-running chat server installed by `make install-daemon`.
# Overridable so the harness works against an ad-hoc dev server too.
LLAMA_BASE_URL = os.environ.get("HERMES_CHAT_URL", "http://127.0.0.1:8080")

# Match the GGUF on the llama.cpp side. Both 4-bit, both Hermes-3-8B; not bit-
# identical (Q4_K_M ≠ MLX 4-bit affine) but the closest apples-to-apples pair.
MLX_MODEL_ID = os.environ.get(
    "HERMES_MLX_MODEL", "mlx-community/Hermes-3-Llama-3.1-8B-4bit"
)

# ---- result records ----------------------------------------------------------

@dataclass
class ThroughputRun:
    backend: str
    prompt_id: str
    input_tokens: int
    output_tokens: int
    ttft_s: float            # time to first token
    prefill_s: float         # input tokens / prefill_tps
    decode_s: float          # output tokens / decode_tps
    prefill_tps: float
    decode_tps: float
    peak_rss_bytes: int      # process-level peak RSS observed during the run
    wall_s: float


@dataclass
class PerplexityRun:
    backend: str
    dataset: str
    n_tokens: int
    perplexity: float
    wall_s: float


@dataclass
class HostInfo:
    arch: str = field(default_factory=lambda: platform.machine())
    macos: str = field(default_factory=lambda: platform.mac_ver()[0])
    chip: str = field(default_factory=lambda: _sysctl_chip())
    ram_gib: int = field(default_factory=lambda: _sysctl_ram_gib())
    pcores: int = field(default_factory=lambda: _sysctl_int("hw.perflevel0.physicalcpu"))
    ecores: int = field(default_factory=lambda: _sysctl_int("hw.perflevel1.physicalcpu", default=0))


def _sysctl_chip() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _sysctl_ram_gib() -> int:
    try:
        b = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        return b // (1024 ** 3)
    except Exception:
        return 0


def _sysctl_int(key: str, default: int = 0) -> int:
    try:
        return int(subprocess.check_output(["sysctl", "-n", key], text=True))
    except Exception:
        return default


# ---- io helpers --------------------------------------------------------------

def load_prompts() -> dict[str, Any]:
    with PROMPTS_PATH.open() as f:
        return json.load(f)


def write_result(name: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = RESULTS_DIR / f"{name}-{stamp}.json"
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return path


def _json_default(o: Any) -> Any:
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    raise TypeError(f"unserializable: {type(o).__name__}")
