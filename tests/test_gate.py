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


BASE = {
    "prompts": {
        "short_qa":       {"decode_tps": 23.8, "prefill_tps": 199.8, "peak_rss_bytes": 190_840_832},
        "medium_summary": {"decode_tps": 19.3, "prefill_tps": 321.9, "peak_rss_bytes": 263_192_576},
    },
    "perplexity": 8.40,
}


def _cur(decode=24.0, rss=190_840_832, ppl=8.40):
    return (
        {"short_qa": {"decode_tps": decode, "prefill_tps": 199.8, "peak_rss_bytes": rss},
         "medium_summary": {"decode_tps": 19.3, "prefill_tps": 321.9, "peak_rss_bytes": 263_192_576}},
        ppl,
    )


def test_compare_passes_on_improvement():
    cur_t, cur_p = _cur(decode=26.0)
    v = gate.compare(cur_t, cur_p, BASE)
    assert v.ok and v.improved
    assert v.deltas["short_qa"]["decode_tps"] > 0


def test_compare_fails_on_perplexity_regression():
    cur_t, cur_p = _cur(ppl=8.55)            # +0.15 > tol 0.10
    v = gate.compare(cur_t, cur_p, BASE)
    assert not v.ok and "perplexity" in v.reason


def test_compare_passes_within_perplexity_noise():
    cur_t, cur_p = _cur(ppl=8.49)            # +0.09 < tol 0.10
    v = gate.compare(cur_t, cur_p, BASE)
    assert v.ok


def test_compare_fails_on_rss_regression_default():
    cur_t, cur_p = _cur(rss=int(190_840_832 * 1.2))   # +20% > factor 1.10
    v = gate.compare(cur_t, cur_p, BASE)
    assert not v.ok and "rss" in v.reason.lower()


def test_compare_rss_ceiling_override_allows_memory_trade():
    # Speculative decoding: 600 MiB resident, way past x1.10, but under the
    # absolute ceiling — must PASS so the human sees the speed win.
    cur_t, cur_p = _cur(decode=40.0, rss=600 * 1024 * 1024)
    v = gate.compare(cur_t, cur_p, BASE, rss_ceiling_mib=700)
    assert v.ok and v.improved


def test_compare_rss_ceiling_still_a_hard_floor():
    cur_t, cur_p = _cur(decode=40.0, rss=900 * 1024 * 1024)   # over 700 ceiling
    v = gate.compare(cur_t, cur_p, BASE, rss_ceiling_mib=700)
    assert not v.ok and "ceiling" in v.reason.lower()


def test_compare_no_gain_passes_but_flags_not_improved():
    cur_t, cur_p = _cur(decode=23.8)         # identical
    v = gate.compare(cur_t, cur_p, BASE)
    assert v.ok and not v.improved
