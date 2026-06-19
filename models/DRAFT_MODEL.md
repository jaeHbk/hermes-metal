# Draft model for speculative decoding (E5)

`llama-3.2-1b-instruct-q4_k_m.gguf` (~0.75 GB) — fetched 2026-06-19 from
`bartowski/Llama-3.2-1B-Instruct-GGUF` on HuggingFace.

Used as the `--model-draft` (`-md`) for speculative decoding against the
Hermes-3-Llama-3.1-8B target. Llama-3.x family → vocab-compatible with the
target (required for speculative decoding). The .gguf itself is gitignored
(models/*.gguf); this note records provenance.
