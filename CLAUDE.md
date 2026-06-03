# CLAUDE.md — Project Blueprint: hermes-metal

This file serves as the definitive architecture specification and onboarding guide for the `hermes-metal` project. Any AI agent or developer reading this document must use it as a strict structural guideline to initialize, construct, and maintain the repository.

---

## 1. Project Vision & Core Objective

`hermes-metal` is a hyper-optimized, zero-lag background AI agent designed to function as an omnipresent "second brain" exclusively on Apple Silicon (M-series) architectures. It maintains a continuous semantic understanding of user knowledge assets (class materials, daily notes, web fragments) to handle low-latency retrieval learning tasks, flashcard generation, and reminders without degrading system performance or compromising laptop battery lifespan.

### Core Architecture Axioms
*   **Zero Resource Contention:** Must run entirely in the background, consuming minimal RAM and relying on system prioritization fences (`renice`) so primary UI applications (browsers, IDEs) suffer zero stutter.
*   **Hardware-Native Execution:** Rejects cross-platform virtualization layers (e.g., Docker, Triton) in favor of metal-level compilation against Apple's Unified Memory Architecture (UMA) via `llama.cpp` and native ARM64 runtimes.
*   **Hybrid Memory Pipeline:** Splits human-facing editing (Markdown in Obsidian) from AI-facing token parsing (embedded vector indexing via LanceDB), maintaining an instantaneous file-watcher ingestion layer.

---

## 2. Technical Specification & Hardware Optimization Matrix

The installation pipeline must dynamically inspect host hardware parameters via standard macOS shell utilities (`sysctl`) and configure execution limits based on the following profile definitions:

| Hardware Tier | Memory Floor | Target Context Window | KV Cache Allocation | Engine Compute Threads |
| :--- | :--- | :--- | :--- | :--- |
| **Base Chips** (M1–M4) | < 32 GB RAM | 8,192 tokens | 8-bit Quantized (`q8_0`) | 4 P-Cores (Burst) / 2 E-Cores (Idle) |
| **Pro / Max / Ultra** | ≥ 32 GB RAM | 32,768 tokens | 8-bit Quantized (`q8_0`) | 8 P-Cores (Burst) / 4 E-Cores (Idle) |

### Engine Optimization Parameters
*   **Model Core:** Hermes 8B configured strictly in GGUF format using the `Q4_K_M` quantization algorithm (~4.7 GB memory footprint).
*   **Flash Attention:** Enabled via the `-fa` compiler configuration flag to accelerate matrix operations during context pre-filling.
*   **Prompt Caching:** Persistent prompt states (`--prompt-cache`) are mandated to bypass redundant evaluations of core system instructions during background invocations.

---

## 3. System Architecture & Data Flow

[ Human Interface ]                [ Background Pipeline ]             [ AI Execution Layer ]
Obsidian Vault                     Python File Watcher                 Local llama.cpp Server
(Flat Markdown Files) ------------> (Watchdog Daemon)                  (Unified Memory Port)
|                                     ^
v                                     |
[ Vector Backend ]                            |
LanceDB (ARM64) <----------------------------+
(Local Semantic Indices)


1.  **Ingestion:** The user edits notes or appends materials within a local Obsidian Vault folder.
2.  **Tracking:** A lightweight Python background daemon (`watchdog`) catches file modification events instantly.
3.  **Vectorization:** Modified documents are chunked and transformed into embeddings locally using an ARM-native implementation of `nomic-embed-text`.
4.  **Retrieval & Inference:** When requested by the agent interface, the daemon queries LanceDB, crafts a tightly pruned context window, and fires an API request to the local `llama.cpp` server sitting at `http://localhost:8080/v1`.

---

## 4. Repository Blueprint & Structural Schema

hermes-metal/
├── CLAUDE.md                # This specification file
├── Makefile                 # Infrastructure-as-Code build automation orchestrator
├── README.md                # Public user documentation
├── config/
│   ├── daemon.plist.template # macOS launchd service template configuration
│   └── engine_flags.env     # Hardware-specific execution environment flags
├── src/
│   ├── backend/             # Local database and RAG management logic
│   │   ├── init.py
│   │   ├── database.py      # LanceDB schema definition and storage handlers
│   │   └── indexer.py       # Text extraction, layout parsing, and chunking
│   ├── daemon/              # Persistent file system observation engine
│   │   ├── init.py
│   │   └── watcher.py       # Watchdog filesystem daemon script
│   └── server/              # Interface client wrappers connecting to llama.cpp
│       ├── init.py
│       └── client.py        # Local API controller targeting port 8080
└── third_party/             # Submodule anchor directories for source compilation
└── llama.cpp/           # Tracked git submodule pointer for target compiling


---

## 5. Automation Architecture: The Infrastructure Orchestrator

The project utilizes a native `Makefile` to handle machine validation, compilation from source, software provisioning, and daemon orchestration.

```makefile
# Partial Architecture Specification for Makefile Execution Steps
.PHONY: all check-env build-engine setup-venv install-daemon start-daemon

all: check-env build-engine setup-venv install-daemon

check-env:
	@echo "Checking system architecture..."
	@uname -m | grep -q "arm64" || (echo "Error: Apple Silicon (arm64) hardware target required." && exit 1)
	@xcode-select -p > /dev/null || (echo "Triggering Xcode Command Line Tools install..." && xcode-select --install)

build-engine:
	@echo "Initializing and building hardware-tailored llama.cpp binaries..."
	git submodule update --init --recursive
	cd third_party/llama.cpp && $(MAKE) GGML_METAL=1

setup-venv:
	@echo "Constructing native ARM64 virtual environment..."
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	# Force explicit native wheel mapping for LanceDB/PyTorch matrix backends
	.venv/bin/pip install --only-binary=:all: --platform macosx_11_0_arm64 lancedb watchdog minimal-llama-bindings

install-daemon:
	@echo "Configuring macOS launchd property list configurations..."
	# Automate dynamic absolute path mapping injection into target .plist files