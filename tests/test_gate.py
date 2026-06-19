"""Tests for bench/gate.py — the bench-gate result loaders and verdict logic."""
from __future__ import annotations

import json
from pathlib import Path

from bench import gate


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def test_load_throughput_indexes_by_prompt(tmp_path):
    payload = {
        "kind": "throughput",
        "runs": [
            {"backend": "llama_cpp", "prompt_id": "short_qa",
             "decode_tps": 23.8, "prefill_tps": 199.8, "peak_rss_bytes": 190_840_832},
            {"backend": "mlx", "prompt_id": "short_qa",
             "decode_tps": 29.9, "prefill_tps": 142.6, "peak_rss_bytes": 1_900_000_000},
        ],
    }
    path = _write(tmp_path, "throughput-x.json", payload)
    got = gate.load_throughput(path)            # llama_cpp only
    assert set(got) == {"short_qa"}
    assert got["short_qa"]["decode_tps"] == 23.8
    assert got["short_qa"]["peak_rss_bytes"] == 190_840_832


def test_load_perplexity_picks_llama_cpp(tmp_path):
    payload = {"kind": "perplexity", "runs": [
        {"backend": "llama_cpp", "perplexity": 8.4047},
        {"backend": "mlx", "perplexity": 11.7232},
    ]}
    path = _write(tmp_path, "perplexity-x.json", payload)
    assert gate.load_perplexity(path) == 8.4047


def test_make_baseline_shape(tmp_path):
    tput = _write(tmp_path, "throughput-x.json", {
        "host": {"chip": "Apple M3 Pro"},
        "runs": [{"backend": "llama_cpp", "prompt_id": "short_qa",
                  "decode_tps": 23.8, "prefill_tps": 199.8, "peak_rss_bytes": 190_840_832}],
    })
    ppl = _write(tmp_path, "perplexity-x.json", {
        "runs": [{"backend": "llama_cpp", "perplexity": 8.4047}],
    })
    base = gate.make_baseline(tput, ppl)
    assert base["perplexity"] == 8.4047
    assert base["prompts"]["short_qa"]["decode_tps"] == 23.8
    assert base["host"]["chip"] == "Apple M3 Pro"
