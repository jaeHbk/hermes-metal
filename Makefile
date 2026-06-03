# hermes-metal — Makefile
#
# Infrastructure-as-Code orchestrator for the hermes-metal Apple Silicon
# second-brain daemon. Drives host validation, llama.cpp compilation, Python
# venv provisioning, model download, launchd agent install, and lifecycle
# control.
#
# Divergences from CLAUDE.md (kept here, not in CLAUDE.md, per instructions):
#   * build-engine uses CMake instead of `make GGML_METAL=1`. The top-level
#     Makefile in ggml-org/llama.cpp now only emits an $(error ...) directing
#     users to CMake; the legacy GNU-make path is dead.
#   * The compiled server binary lives at
#     third_party/llama.cpp/build/bin/llama-server (not at the source root).
#   * setup-venv does NOT use `--platform macosx_11_0_arm64`. On native arm64
#     Python, that flag is redundant and pip refuses it without --target /
#     --dry-run. `--prefer-binary` alone forces wheel-only installs.
#   * `minimal-llama-bindings` does not exist on PyPI; we talk to llama-server
#     over HTTP via httpx instead. (See requirements.txt.)
#   * The daemon is installed as a LaunchAgent (per-user, ~/Library/LaunchAgents)
#     rather than a LaunchDaemon (system-wide, /Library/LaunchDaemons), since
#     it runs in the user's session, has no root needs, and must access TCC-
#     protected user files like the Obsidian vault.

# ----- Variables ------------------------------------------------------------

WORKING_DIR     := $(shell pwd)
PYTHON          := python3
VENV_DIR        := $(WORKING_DIR)/.venv
VENV_PIP        := $(VENV_DIR)/bin/pip
VENV_PY         := $(VENV_DIR)/bin/python

LLAMA_DIR       := $(WORKING_DIR)/third_party/llama.cpp
LLAMA_BUILD_DIR := $(LLAMA_DIR)/build
LLAMA_SERVER    := $(LLAMA_BUILD_DIR)/bin/llama-server
LLAMA_REPO_URL  := https://github.com/ggerganov/llama.cpp

MODELS_DIR      := $(WORKING_DIR)/models
STORAGE_DIR     := $(WORKING_DIR)/storage
SLOTS_DIR       := $(STORAGE_DIR)/slots
LOGS_DIR        := $(WORKING_DIR)/logs

MODEL_FILE      := hermes-8b-q4_k_m.gguf
MODEL_PATH      := $(MODELS_DIR)/$(MODEL_FILE)
EMBED_MODEL_FILE:= nomic-embed-text-v1.5.f16.gguf
EMBED_MODEL_PATH:= $(MODELS_DIR)/$(EMBED_MODEL_FILE)

CONFIG_DIR      := $(WORKING_DIR)/config
PLIST_TEMPLATE  := $(CONFIG_DIR)/daemon.plist.template
EMBED_TEMPLATE  := $(CONFIG_DIR)/embed.plist.template
WATCH_TEMPLATE  := $(CONFIG_DIR)/watcher.plist.template
HOST_TOPOLOGY   := $(CONFIG_DIR)/host_topology.env
ENGINE_FLAGS    := $(CONFIG_DIR)/engine_flags.env

USER_UID        := $(shell id -u)
GUI_DOMAIN      := gui/$(USER_UID)

ENGINE_LABEL    := com.hermes.metal.engine
ENGINE_PLIST    := $(HOME)/Library/LaunchAgents/$(ENGINE_LABEL).plist
ENGINE_TARGET   := $(GUI_DOMAIN)/$(ENGINE_LABEL)

EMBED_LABEL     := com.hermes.metal.embed
EMBED_PLIST     := $(HOME)/Library/LaunchAgents/$(EMBED_LABEL).plist
EMBED_TARGET    := $(GUI_DOMAIN)/$(EMBED_LABEL)

WATCH_LABEL     := com.hermes.metal.watcher
WATCH_PLIST     := $(HOME)/Library/LaunchAgents/$(WATCH_LABEL).plist
WATCH_TARGET    := $(GUI_DOMAIN)/$(WATCH_LABEL)

# HERMES_VAULT_PATH is consulted at install time so the watcher LaunchAgent
# has a concrete path baked into its plist EnvironmentVariables.
VAULT_PATH      := $(if $(HERMES_VAULT_PATH),$(HERMES_VAULT_PATH),$(HOME)/Documents/Obsidian)

# ----- Phony targets --------------------------------------------------------

.PHONY: all init check-env build-engine setup-venv fetch-model fetch-embed-model \
        install-daemon install-engine-daemon install-embed-daemon install-watcher-daemon \
        install-cli uninstall-cli \
        start-daemon stop-daemon clean-cache uninstall doctor test help \
        bench bench-setup bench-throughput bench-perplexity bench-power bench-report

# Default: full installation pipeline end-to-end. install-daemon installs
# all three LaunchAgents (engine, embed, watcher); see that target for the
# rendering recipes.
all: check-env init build-engine setup-venv fetch-model fetch-embed-model install-daemon

# ----- help -----------------------------------------------------------------

help:
	@echo "hermes-metal Makefile — targets:"
	@echo ""
	@echo "  make all                 Full pipeline: check-env -> init -> build-engine ->"
	@echo "                           setup-venv -> fetch-model -> fetch-embed-model ->"
	@echo "                           install-daemon."
	@echo ""
	@echo "  make check-env           Verify Apple Silicon, Xcode CLT, and write"
	@echo "                           config/host_topology.env (TIER, CONTEXT_TOKENS,"
	@echo "                           THREAD_COUNT) from sysctl probes."
	@echo "  make init                git init (if needed), add llama.cpp submodule,"
	@echo "                           recursive submodule update, mkdir runtime dirs."
	@echo "  make build-engine        CMake-build llama.cpp with Metal + embedded"
	@echo "                           metallib; produces build/bin/llama-server."
	@echo "  make setup-venv          Create .venv and pip-install requirements.txt"
	@echo "                           (wheel-only, native arm64)."
	@echo "  make fetch-model         Download Hermes-3-Llama-3.1-8B Q4_K_M GGUF."
	@echo "  make fetch-embed-model   Download nomic-embed-text-v1.5 f16 GGUF."
	@echo "  make install-daemon      Render and bootstrap all three LaunchAgents:"
	@echo "                           engine (chat), embed, watcher."
	@echo "  make install-engine-daemon   Render only com.hermes.metal.engine."
	@echo "  make install-embed-daemon    Render only com.hermes.metal.embed."
	@echo "  make install-watcher-daemon  Render only com.hermes.metal.watcher."
	@echo ""
	@echo "  make install-cli         Symlink bin/hermes into a directory on PATH"
	@echo "                           (default ~/.local/bin; override with PREFIX=...)"
	@echo "  make uninstall-cli       Remove the hermes symlink."
	@echo ""
	@echo "  make start-daemon        launchctl kickstart all three agents."
	@echo "  make stop-daemon         launchctl stop all three agents (KeepAlive may"
	@echo "                           respawn on crash; use uninstall to remove)."
	@echo "  make clean-cache         Wipe storage/slots/* (KV slot caches)."
	@echo "  make uninstall           launchctl bootout + remove all three plists."
	@echo "                           Models and storage/ are kept on disk."
	@echo "  make doctor              End-to-end self-diagnostic with remediation."
	@echo "  make test                Run the pytest suite (pure-Python; no daemons)."
	@echo ""
	@echo "  make bench               Throughput + perplexity vs MLX 4-bit (no sudo)."
	@echo "  make bench-power         powermetrics power suite (asks for sudo)."
	@echo "  make bench-report        Regenerate bench/results/REPORT.md."
	@echo ""
	@echo "  make help                This message."

# ----- check-env ------------------------------------------------------------

check-env:
	@echo "==> check-env: validating host"
	@if [ "$$(uname -m)" != "arm64" ]; then \
		echo "ERROR: hermes-metal requires Apple Silicon (arm64). Detected: $$(uname -m)"; \
		exit 1; \
	fi
	@echo "    arch:        arm64 OK"
	@if ! xcode-select -p >/dev/null 2>&1; then \
		echo "    Xcode Command Line Tools missing — triggering install GUI..."; \
		xcode-select --install || true; \
		echo "ERROR: re-run 'make check-env' after the CLT install completes."; \
		exit 1; \
	fi
	@echo "    xcode CLT:   $$(xcode-select -p)"
	@mkdir -p "$(CONFIG_DIR)"
	@RAM_BYTES=$$(sysctl -n hw.memsize); \
	RAM_GIB=$$(( RAM_BYTES / 1024 / 1024 / 1024 )); \
	NCPU=$$(sysctl -n hw.ncpu); \
	PCORES=$$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo "$$NCPU"); \
	ECORES=$$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo "0"); \
	if [ "$$RAM_GIB" -lt 32 ]; then \
		TIER=BASE; CONTEXT_TOKENS=8192; THREAD_COUNT=4; \
	else \
		TIER=PRO; CONTEXT_TOKENS=32768; THREAD_COUNT=8; \
	fi; \
	echo "    ram:         $${RAM_GIB} GiB"; \
	echo "    cpu cores:   total=$${NCPU}  P=$${PCORES}  E=$${ECORES}"; \
	echo "    tier:        $${TIER}"; \
	echo "    ctx tokens:  $${CONTEXT_TOKENS}"; \
	echo "    threads:     $${THREAD_COUNT}"; \
	{ \
		echo "# host_topology.env — auto-generated by 'make check-env'."; \
		echo "# Do not edit by hand; rerun check-env to refresh."; \
		echo "TIER=$${TIER}"; \
		echo "RAM_GIB=$${RAM_GIB}"; \
		echo "NCPU=$${NCPU}"; \
		echo "PCORES=$${PCORES}"; \
		echo "ECORES=$${ECORES}"; \
		echo "CONTEXT_TOKENS=$${CONTEXT_TOKENS}"; \
		echo "THREAD_COUNT=$${THREAD_COUNT}"; \
	} > "$(HOST_TOPOLOGY)"
	@echo "    wrote:       $(HOST_TOPOLOGY)"

# ----- init -----------------------------------------------------------------

init:
	@echo "==> init: repo + submodules + runtime dirs"
	@if [ ! -d "$(WORKING_DIR)/.git" ]; then \
		echo "    git init"; \
		git init >/dev/null; \
	else \
		echo "    git repo: present"; \
	fi
	@if [ ! -d "$(LLAMA_DIR)/.git" ] && [ ! -f "$(LLAMA_DIR)/.git" ]; then \
		echo "    adding llama.cpp submodule"; \
		mkdir -p "$(WORKING_DIR)/third_party"; \
		git submodule add "$(LLAMA_REPO_URL)" "third_party/llama.cpp" || \
			(echo "    submodule add failed — attempting plain clone fallback" && \
			 git clone --recursive "$(LLAMA_REPO_URL)" "$(LLAMA_DIR)"); \
	else \
		echo "    llama.cpp submodule: present"; \
	fi
	@echo "    submodule update --init --recursive"
	@git submodule update --init --recursive 2>/dev/null || true
	@mkdir -p "$(MODELS_DIR)" "$(STORAGE_DIR)" "$(SLOTS_DIR)" "$(LOGS_DIR)"
	@echo "    mkdir:       models/  storage/  storage/slots/  logs/"

# ----- build-engine ---------------------------------------------------------

build-engine: $(LLAMA_SERVER)

$(LLAMA_SERVER):
	@echo "==> build-engine: cmake llama.cpp with Metal (embedded metallib)"
	@if [ ! -d "$(LLAMA_DIR)" ]; then \
		echo "ERROR: $(LLAMA_DIR) missing. Run 'make init' first."; \
		exit 1; \
	fi
	cd "$(LLAMA_DIR)" && cmake -B build \
		-DCMAKE_BUILD_TYPE=Release \
		-DGGML_METAL=ON \
		-DGGML_METAL_EMBED_LIBRARY=ON \
		-DGGML_NATIVE=ON \
		-DLLAMA_CURL=OFF \
		-DLLAMA_BUILD_TESTS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_SERVER=ON
	cd "$(LLAMA_DIR)" && cmake --build build --config Release --target llama-server -j
	@test -x "$(LLAMA_SERVER)" || (echo "ERROR: expected $(LLAMA_SERVER) after build" && exit 1)
	@echo "    built:       $(LLAMA_SERVER)"

# ----- setup-venv -----------------------------------------------------------

setup-venv:
	@echo "==> setup-venv: native arm64 .venv"
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv "$(VENV_DIR)"; \
	fi
	@$(VENV_PIP) install --upgrade pip >/dev/null
	@if [ -f "$(WORKING_DIR)/requirements.txt" ]; then \
		$(VENV_PIP) install --prefer-binary -r "$(WORKING_DIR)/requirements.txt"; \
	else \
		echo "    requirements.txt not found — installing canonical set inline"; \
		$(VENV_PIP) install --prefer-binary --upgrade \
			lancedb watchdog httpx pyyaml numpy pyarrow; \
	fi
	@$(VENV_PY) -c "import platform, sys; \
assert platform.machine() == 'arm64', f'expected arm64, got {platform.machine()}'; \
print(f'    venv ready:  Python {sys.version.split()[0]} on {platform.machine()}')"

# ----- fetch-model / fetch-embed-model --------------------------------------

fetch-model:
	@echo "==> fetch-model: $(MODEL_FILE)"
	@mkdir -p "$(MODELS_DIR)"
	bash "$(WORKING_DIR)/scripts/fetch_model.sh"

fetch-embed-model:
	@echo "==> fetch-embed-model: $(EMBED_MODEL_FILE)"
	@mkdir -p "$(MODELS_DIR)"
	bash "$(WORKING_DIR)/scripts/fetch_embed_model.sh"

# ----- install-daemon -------------------------------------------------------
#
# The hermes-metal daemon is actually three cooperating LaunchAgents:
#   * com.hermes.metal.engine   — chat llama-server on $(ENGINE_PORT)
#   * com.hermes.metal.embed    — embedding llama-server on $(EMBED_PORT)
#   * com.hermes.metal.watcher  — Python watchdog that re-indexes the vault
# install-daemon is a meta-target that renders + bootstraps all three.

install-daemon: install-engine-daemon install-embed-daemon install-watcher-daemon
	@echo "==> install-daemon: all three LaunchAgents bootstrapped"
	@echo "    engine:  $(ENGINE_LABEL)  -> http://127.0.0.1:$$( . $(ENGINE_FLAGS) && echo $$ENGINE_PORT )"
	@echo "    embed:   $(EMBED_LABEL)   -> http://127.0.0.1:$$( . $(ENGINE_FLAGS) && echo $$EMBED_PORT )"
	@echo "    watcher: $(WATCH_LABEL)   (vault: $(VAULT_PATH))"

install-engine-daemon:
	@echo "==> install-engine-daemon: render + bootstrap $(ENGINE_LABEL)"
	@if [ ! -f "$(HOST_TOPOLOGY)" ]; then \
		echo "ERROR: $(HOST_TOPOLOGY) missing. Run 'make check-env' first."; \
		exit 1; \
	fi
	@if [ ! -f "$(ENGINE_FLAGS)" ]; then \
		echo "ERROR: $(ENGINE_FLAGS) missing."; \
		exit 1; \
	fi
	@if [ ! -f "$(PLIST_TEMPLATE)" ]; then \
		echo "ERROR: $(PLIST_TEMPLATE) missing."; \
		exit 1; \
	fi
	@if [ ! -x "$(LLAMA_SERVER)" ]; then \
		echo "ERROR: $(LLAMA_SERVER) missing. Run 'make build-engine' first."; \
		exit 1; \
	fi
	@if [ ! -f "$(MODEL_PATH)" ]; then \
		echo "ERROR: $(MODEL_PATH) missing. Run 'make fetch-model' first."; \
		exit 1; \
	fi
	@mkdir -p "$(HOME)/Library/LaunchAgents" "$(SLOTS_DIR)" "$(LOGS_DIR)"
	@. "$(HOST_TOPOLOGY)"; . "$(ENGINE_FLAGS)"; \
	echo "    tier=$$TIER  ctx=$$CONTEXT_TOKENS  threads=$$THREAD_COUNT  port=$$ENGINE_PORT"; \
	sed \
		-e "s|{WORKING_DIR}|$(WORKING_DIR)|g" \
		-e "s|{MODEL_PATH}|$(MODEL_PATH)|g" \
		-e "s|{CONTEXT_TOKENS}|$$CONTEXT_TOKENS|g" \
		-e "s|{THREAD_COUNT}|$$THREAD_COUNT|g" \
		-e "s|{ENGINE_PORT}|$$ENGINE_PORT|g" \
		-e "s|{CACHE_TYPE_K}|$$CACHE_TYPE_K|g" \
		-e "s|{CACHE_TYPE_V}|$$CACHE_TYPE_V|g" \
		-e "s|{SLOT_SAVE_PATH}|$$SLOT_SAVE_PATH|g" \
		"$(PLIST_TEMPLATE)" > "$(ENGINE_PLIST)"
	@echo "    rendered:    $(ENGINE_PLIST)"
	@$(MAKE) --no-print-directory _bootstrap PLIST=$(ENGINE_PLIST) TARGET=$(ENGINE_TARGET)

install-embed-daemon:
	@echo "==> install-embed-daemon: render + bootstrap $(EMBED_LABEL)"
	@if [ ! -f "$(HOST_TOPOLOGY)" ]; then \
		echo "ERROR: $(HOST_TOPOLOGY) missing. Run 'make check-env' first."; \
		exit 1; \
	fi
	@if [ ! -f "$(ENGINE_FLAGS)" ]; then \
		echo "ERROR: $(ENGINE_FLAGS) missing."; \
		exit 1; \
	fi
	@if [ ! -f "$(EMBED_TEMPLATE)" ]; then \
		echo "ERROR: $(EMBED_TEMPLATE) missing."; \
		exit 1; \
	fi
	@if [ ! -x "$(LLAMA_SERVER)" ]; then \
		echo "ERROR: $(LLAMA_SERVER) missing. Run 'make build-engine' first."; \
		exit 1; \
	fi
	@if [ ! -f "$(EMBED_MODEL_PATH)" ]; then \
		echo "ERROR: $(EMBED_MODEL_PATH) missing. Run 'make fetch-embed-model' first."; \
		exit 1; \
	fi
	@mkdir -p "$(HOME)/Library/LaunchAgents" "$(LOGS_DIR)"
	@. "$(HOST_TOPOLOGY)"; . "$(ENGINE_FLAGS)"; \
	echo "    threads=$$THREAD_COUNT  port=$$EMBED_PORT"; \
	sed \
		-e "s|{WORKING_DIR}|$(WORKING_DIR)|g" \
		-e "s|{EMBED_MODEL_PATH}|$(EMBED_MODEL_PATH)|g" \
		-e "s|{THREAD_COUNT}|$$THREAD_COUNT|g" \
		-e "s|{EMBED_PORT}|$$EMBED_PORT|g" \
		"$(EMBED_TEMPLATE)" > "$(EMBED_PLIST)"
	@echo "    rendered:    $(EMBED_PLIST)"
	@$(MAKE) --no-print-directory _bootstrap PLIST=$(EMBED_PLIST) TARGET=$(EMBED_TARGET)

install-watcher-daemon:
	@echo "==> install-watcher-daemon: render + bootstrap $(WATCH_LABEL)"
	@if [ ! -f "$(WATCH_TEMPLATE)" ]; then \
		echo "ERROR: $(WATCH_TEMPLATE) missing."; \
		exit 1; \
	fi
	@if [ ! -f "$(ENGINE_FLAGS)" ]; then \
		echo "ERROR: $(ENGINE_FLAGS) missing."; \
		exit 1; \
	fi
	@if [ ! -x "$(VENV_PY)" ]; then \
		echo "ERROR: $(VENV_PY) missing. Run 'make setup-venv' first."; \
		exit 1; \
	fi
	@if [ ! -d "$(VAULT_PATH)" ]; then \
		echo "WARNING: vault path $(VAULT_PATH) does not exist."; \
		echo "         The watcher will refuse to start until it does."; \
		echo "         Set HERMES_VAULT_PATH=/your/vault and re-run install-watcher-daemon."; \
	fi
	@mkdir -p "$(HOME)/Library/LaunchAgents" "$(LOGS_DIR)" "$(STORAGE_DIR)/lancedb"
	@. "$(ENGINE_FLAGS)"; \
	echo "    vault=$(VAULT_PATH)  embed_port=$$EMBED_PORT"; \
	sed \
		-e "s|{WORKING_DIR}|$(WORKING_DIR)|g" \
		-e "s|{VAULT_PATH}|$(VAULT_PATH)|g" \
		-e "s|{EMBED_PORT}|$$EMBED_PORT|g" \
		"$(WATCH_TEMPLATE)" > "$(WATCH_PLIST)"
	@echo "    rendered:    $(WATCH_PLIST)"
	@$(MAKE) --no-print-directory _bootstrap PLIST=$(WATCH_PLIST) TARGET=$(WATCH_TARGET)

# Internal helper: lint + bootstrap (or legacy load) a rendered plist.
# Invoked by install-{engine,embed,watcher}-daemon. Not in .PHONY because it's
# a private hook; underscore prefix signals "do not call directly."
_bootstrap:
	@if command -v plutil >/dev/null 2>&1; then \
		plutil -lint "$(PLIST)" >/dev/null || \
			(echo "ERROR: $(PLIST) failed plutil -lint" && exit 1); \
		echo "    plutil:      lint OK"; \
	fi
	@launchctl bootout "$(TARGET)" 2>/dev/null || true
	@if launchctl bootstrap "$(GUI_DOMAIN)" "$(PLIST)" 2>/dev/null; then \
		echo "    launchctl:   bootstrapped $(TARGET)"; \
	else \
		echo "    launchctl bootstrap unsupported — falling back to legacy load"; \
		launchctl unload "$(PLIST)" 2>/dev/null || true; \
		launchctl load -w "$(PLIST)"; \
		echo "    launchctl:   loaded (legacy)"; \
	fi

# ----- install-cli / uninstall-cli ------------------------------------------
# Symlink bin/hermes into a PATH directory. Defaults to ~/.local/bin (no
# sudo required); pass PREFIX=/usr/local/bin to install system-wide.

PREFIX ?= $(HOME)/.local/bin
HERMES_BIN := $(WORKING_DIR)/bin/hermes
HERMES_LINK := $(PREFIX)/hermes

install-cli:
	@echo "==> install-cli: link $(HERMES_BIN) -> $(HERMES_LINK)"
	@if [ ! -x "$(HERMES_BIN)" ]; then \
		echo "ERROR: $(HERMES_BIN) missing or not executable."; \
		exit 1; \
	fi
	@mkdir -p "$(PREFIX)"
	@ln -sfn "$(HERMES_BIN)" "$(HERMES_LINK)"
	@echo "    linked: $(HERMES_LINK)"
	@case ":$$PATH:" in \
		*":$(PREFIX):"*) echo "    PATH:   OK ($(PREFIX) is on PATH)" ;; \
		*) echo "    PATH:   WARNING — $(PREFIX) is NOT on PATH."; \
		   echo "             Add to your shell rc:  export PATH=\"$(PREFIX):\$$PATH\"" ;; \
	esac
	@echo "    try:    hermes status"

uninstall-cli:
	@echo "==> uninstall-cli: remove $(HERMES_LINK)"
	@rm -f "$(HERMES_LINK)"
	@echo "    removed (if present)."

# ----- start / stop ---------------------------------------------------------

start-daemon:
	@echo "==> start-daemon: kickstart all three agents"
	@for tgt in $(ENGINE_TARGET) $(EMBED_TARGET) $(WATCH_TARGET); do \
		echo "    kickstart $$tgt"; \
		launchctl kickstart -k "$$tgt" 2>/dev/null || \
			(echo "      kickstart unsupported — fallback launchctl start"; \
			 lbl=$$(echo $$tgt | sed 's|.*/||'); launchctl start "$$lbl" 2>/dev/null || true); \
	done

stop-daemon:
	@echo "==> stop-daemon: stop all three agents"
	@echo "    note: KeepAlive will respawn on non-zero exit."
	@echo "    use 'make uninstall' to remove the agents entirely."
	@for lbl in $(ENGINE_LABEL) $(EMBED_LABEL) $(WATCH_LABEL); do \
		launchctl stop "$$lbl" 2>/dev/null || true; \
	done

# ----- clean-cache ----------------------------------------------------------

clean-cache:
	@echo "==> clean-cache: wiping storage/slots/*"
	@rm -rf "$(SLOTS_DIR)"
	@mkdir -p "$(SLOTS_DIR)"
	@echo "    slots dir reset: $(SLOTS_DIR)"

# ----- doctor ---------------------------------------------------------------
# Run the end-to-end self-diagnostic. Routes through bin/hermes so it works
# whether the venv exists or not (doctor is stdlib-only, see src/doctor.py).
# Exits non-zero on FAIL so `make doctor && make start-daemon` is safe.

doctor:
	@"$(WORKING_DIR)/bin/hermes" doctor

# ----- test -----------------------------------------------------------------
# Run the pytest suite. Pure-Python tests; no daemons or network needed.

test:
	@"$(VENV_PY)" -m pytest tests/ -v

# ----- uninstall ------------------------------------------------------------

uninstall:
	@echo "==> uninstall: bootout + remove all three agent plists"
	@for tgt in $(ENGINE_TARGET) $(EMBED_TARGET) $(WATCH_TARGET); do \
		launchctl bootout "$$tgt" 2>/dev/null || true; \
	done
	@for plist in $(ENGINE_PLIST) $(EMBED_PLIST) $(WATCH_PLIST); do \
		[ -f "$$plist" ] && (launchctl unload "$$plist" 2>/dev/null || true; rm -f "$$plist"; echo "    removed:     $$plist") || true; \
	done
	@echo ""
	@echo "    NOTE: models/ and storage/ are intentionally KEPT."
	@echo "    To fully reset: rm -rf $(MODELS_DIR) $(STORAGE_DIR) $(LOGS_DIR) $(VENV_DIR)"

# ----- bench ----------------------------------------------------------------
#
# Head-to-head benchmark vs MLX 4-bit on the same Hermes-3-8B. The MLX side
# uses a SEPARATE venv (bench/.venv-mlx) so its torch / transformers chain
# cannot drift the daemon's runtime stack. See bench/README.md.

BENCH_DIR        := $(WORKING_DIR)/bench
BENCH_MLX_VENV   := $(BENCH_DIR)/.venv-mlx
BENCH_MLX_PY     := $(BENCH_MLX_VENV)/bin/python

bench-setup:
	@echo "==> bench-setup: provisioning $(BENCH_MLX_VENV)"
	@bash "$(BENCH_DIR)/setup.sh"

bench-throughput: bench-setup
	@echo "==> bench-throughput: llama_cpp + mlx (both backends)"
	@if ! curl -sf "$$( . $(ENGINE_FLAGS) && echo http://127.0.0.1:$$ENGINE_PORT )/health" >/dev/null 2>&1; then \
		echo "ERROR: chat server not reachable. Run 'make start-daemon' first."; \
		exit 1; \
	fi
	@cd "$(WORKING_DIR)" && "$(BENCH_MLX_PY)" -m bench.bench_throughput --backend both

bench-perplexity: bench-setup
	@echo "==> bench-perplexity: llama_cpp via llama-perplexity, mlx in-process"
	@if [ ! -x "$(LLAMA_DIR)/build/bin/llama-perplexity" ]; then \
		echo "    building llama-perplexity (one-time)"; \
		cd "$(LLAMA_DIR)" && cmake --build build --config Release --target llama-perplexity -j; \
	fi
	@cd "$(WORKING_DIR)" && "$(VENV_PY)"      -m bench.bench_perplexity --backend llama_cpp
	@cd "$(WORKING_DIR)" && "$(BENCH_MLX_PY)" -m bench.bench_perplexity --backend mlx

bench-power:
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "ERROR: bench-power needs root. Run: sudo make bench-power"; \
		exit 1; \
	fi
	@echo "==> bench-power: powermetrics suite (running as root)"
	"$(BENCH_DIR)/bench_power.sh" llama_cpp medium_summary
	"$(BENCH_DIR)/bench_power.sh" mlx       medium_summary

bench-report:
	@cd "$(WORKING_DIR)" && "$(VENV_PY)" -m bench.aggregate
	@echo "    open: bench/results/REPORT.md"

# Default `make bench` skips power (sudo) — explicit gate via bench-power.
bench: bench-throughput bench-perplexity bench-report
	@echo "==> bench: done. For battery numbers run 'make bench-power' (asks sudo)."
