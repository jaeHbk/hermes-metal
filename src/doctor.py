"""hermes-metal self-diagnostic.

`hermes doctor` walks the install end-to-end and reports the first thing the
user can fix. It is intentionally stdlib-only: the most common reason to reach
for `doctor` is that something in the venv or the LaunchAgents is wrong, and
the diagnostic must keep working when those layers are broken.

Each check returns a `Result` with one of four levels:

* OK    — passing; user does nothing.
* WARN  — degraded but the system can still serve queries (e.g. agent loaded
          but no chat traffic seen yet).
* FAIL  — the user-facing path is broken (no chat server, no embed server,
          empty index, missing model).
* SKIP  — a prerequisite for this check is missing, so we did not run it.

Every non-OK result carries a one-line `fix:` remediation. The exit code from
`run()` is 0 if no FAIL, otherwise 1, so `make doctor || echo "broken"` works
in scripts.

Doctor purposely never imports the project's own packages (lancedb, watchdog,
httpx, src.backend.*). Those imports are the ones most likely to fail when
the user reaches for the doctor.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import plistlib
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- types


OK = "OK"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"

# Order matters: severity rank for "did anything fail?" / overall status.
_RANK = {OK: 0, SKIP: 1, WARN: 2, FAIL: 3}


@dataclasses.dataclass
class Result:
    name: str
    level: str
    detail: str = ""
    fix: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Section:
    title: str
    results: list[Result] = dataclasses.field(default_factory=list)

    def add(self, r: Result) -> None:
        self.results.append(r)

    @property
    def worst(self) -> str:
        return max((r.level for r in self.results), key=lambda lv: _RANK[lv], default=OK)


# ----------------------------------------------------------- env loading


def _load_env_file(path: Path) -> dict[str, str]:
    """Parse a KEY=VALUE .env file. Tolerant of blank lines and `#` comments.

    No shell substitution: this matches the contract documented in
    `config/engine_flags.env` ("Sourced; do NOT include shell logic").
    """
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _engine_flags() -> dict[str, str]:
    return _load_env_file(REPO_ROOT / "config" / "engine_flags.env")


def _host_topology() -> dict[str, str]:
    return _load_env_file(REPO_ROOT / "config" / "host_topology.env")


def _engine_port() -> int:
    return int(_engine_flags().get("ENGINE_PORT", "8080"))


def _embed_port() -> int:
    return int(_engine_flags().get("EMBED_PORT", "8081"))


def _vault_path() -> Path | None:
    raw = os.environ.get("HERMES_VAULT_PATH")
    if not raw:
        # Watcher plist bakes the value at install time; recover it from there
        # so `hermes doctor` works in shells that don't export the var.
        plist = Path.home() / "Library" / "LaunchAgents" / "com.hermes.metal.watcher.plist"
        if plist.is_file():
            try:
                with plist.open("rb") as fh:
                    data = plistlib.load(fh)
                env = data.get("EnvironmentVariables") or {}
                raw = env.get("HERMES_VAULT_PATH")
            except (plistlib.InvalidFileException, OSError):
                raw = None
    if not raw:
        return None
    return Path(raw).expanduser()


def _lancedb_path() -> Path:
    raw = os.environ.get("HERMES_LANCEDB_PATH") or str(REPO_ROOT / "storage" / "lancedb")
    return Path(raw).expanduser().resolve()


# ---------------------------------------------------------------- helpers


def _run(cmd: list[str], timeout: float = 5.0) -> tuple[int, str, str]:
    """Run a process, return (rc, stdout, stderr). Never raises."""
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return p.returncode, p.stdout, p.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return 127, "", str(exc)


def _http_get_json(url: str, timeout: float = 3.0) -> tuple[int | None, dict[str, Any] | None, str]:
    """GET `url` with stdlib urllib. Returns (status, body_or_None, error)."""
    try:
        # Request() raises ValueError on empty/scheme-less URLs (e.g. user
        # exported HERMES_CHAT_URL=""). Build it inside the try so the
        # except below catches that just like a transport failure.
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw), ""
            except json.JSONDecodeError:
                # /health on llama-server returns JSON; if we got HTML or text,
                # surface the prefix so the user can identify the wrong service.
                return resp.status, None, raw[:120]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return exc.code, None, body[:200]
    except (
        urllib.error.URLError,
        TimeoutError,
        socket.timeout,
        ConnectionError,
        OSError,
        # urlopen raises ValueError on empty / unparseable URLs (e.g. the user
        # exported HERMES_CHAT_URL="" to clear it). Treat as transport failure
        # so the doctor surfaces a clean FAIL instead of crashing.
        ValueError,
    ) as exc:
        return None, None, str(exc)


def _port_listening(port: int) -> bool:
    """True if a process on this host is bound to 127.0.0.1:port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        # Connect to loopback; success means *something* is accepting on that port.
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


# ----------------------------------------------------------------- checks


def check_host() -> Section:
    s = Section("Host")
    arch = platform.machine()
    if arch == "arm64":
        s.add(Result("architecture", OK, f"arm64 (Apple Silicon) — {platform.platform()}"))
    else:
        s.add(Result(
            "architecture",
            FAIL,
            f"hermes-metal requires Apple Silicon; detected {arch}",
            fix="run on an M-series Mac; cross-arch is not supported.",
        ))

    rc, out, _ = _run(["xcode-select", "-p"], timeout=2.0)
    if rc == 0 and out.strip():
        s.add(Result("xcode_clt", OK, out.strip()))
    else:
        s.add(Result(
            "xcode_clt",
            FAIL,
            "Xcode Command Line Tools not installed (required to build llama.cpp)",
            fix="xcode-select --install",
        ))

    # RAM tier sanity. We don't fail on <32 GB — base tier is supported — but
    # surface it so users with 8 GB don't wonder why context is 8K.
    rc, out, _ = _run(["sysctl", "-n", "hw.memsize"], timeout=2.0)
    if rc == 0 and out.strip().isdigit():
        ram_gib = int(out.strip()) // (1024 ** 3)
        tier = "BASE" if ram_gib < 32 else "PRO"
        s.add(Result("ram", OK, f"{ram_gib} GiB → tier {tier}"))
    return s


def check_repo() -> Section:
    s = Section("Repository")
    must_exist: list[tuple[str, Path, str]] = [
        ("Makefile", REPO_ROOT / "Makefile", FAIL),
        ("config/engine_flags.env", REPO_ROOT / "config" / "engine_flags.env", FAIL),
        ("config/host_topology.env", REPO_ROOT / "config" / "host_topology.env", WARN),
        ("config/daemon.plist.template", REPO_ROOT / "config" / "daemon.plist.template", FAIL),
        ("config/embed.plist.template", REPO_ROOT / "config" / "embed.plist.template", FAIL),
        ("config/watcher.plist.template", REPO_ROOT / "config" / "watcher.plist.template", FAIL),
    ]
    for label, path, severity in must_exist:
        if path.is_file():
            s.add(Result(label, OK, str(path.relative_to(REPO_ROOT))))
        else:
            fix = "make check-env" if "host_topology" in label else f"missing file: {path}"
            s.add(Result(label, severity, "missing", fix=fix))

    submodule = REPO_ROOT / "third_party" / "llama.cpp"
    if (submodule / "CMakeLists.txt").is_file():
        s.add(Result("submodule:llama.cpp", OK, "third_party/llama.cpp present"))
    else:
        s.add(Result(
            "submodule:llama.cpp",
            FAIL,
            "third_party/llama.cpp is empty or missing",
            fix="git submodule update --init --recursive",
        ))
    return s


def check_build() -> Section:
    s = Section("Build")
    server = REPO_ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-server"
    if server.is_file() and os.access(server, os.X_OK):
        s.add(Result("llama-server", OK, str(server.relative_to(REPO_ROOT))))
    elif server.is_file():
        s.add(Result(
            "llama-server",
            FAIL,
            f"{server} exists but is not executable",
            fix=f"chmod +x {server}",
        ))
    else:
        s.add(Result(
            "llama-server",
            FAIL,
            "llama-server binary missing",
            fix="make build-engine",
        ))
    return s


def check_python() -> Section:
    s = Section("Python venv")
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_py.is_file():
        s.add(Result(".venv/bin/python", FAIL, "virtualenv missing", fix="make setup-venv"))
        return s

    rc, out, err = _run([str(venv_py), "-c",
        "import platform,sys;print(platform.machine());print(sys.version.split()[0])"],
        timeout=5.0,
    )
    if rc != 0:
        s.add(Result(
            "venv python", FAIL,
            f"venv python failed to start: {err.strip() or out.strip()}",
            fix="rm -rf .venv && make setup-venv",
        ))
        return s
    lines = out.strip().splitlines()
    arch = lines[0] if lines else "?"
    pyver = lines[1] if len(lines) > 1 else "?"
    if arch != "arm64":
        s.add(Result(
            "venv arch", FAIL,
            f"venv is {arch}, not arm64 (likely created with x86_64 Python under Rosetta)",
            fix="rm -rf .venv && /opt/homebrew/bin/python3 -m venv .venv && make setup-venv",
        ))
    else:
        s.add(Result("venv arch", OK, f"arm64, Python {pyver}"))

    # Module import probe: catch the case where `pip install` succeeded but a
    # wheel is broken (rare but real on bleeding-edge Python).
    rc, _out, err = _run(
        [str(venv_py), "-c", "import lancedb, watchdog, httpx, pyarrow, numpy"],
        timeout=10.0,
    )
    if rc == 0:
        s.add(Result("venv packages", OK, "lancedb, watchdog, httpx, pyarrow, numpy import OK"))
    else:
        s.add(Result(
            "venv packages", FAIL,
            f"import error: {err.strip().splitlines()[-1] if err else 'unknown'}",
            fix="rm -rf .venv && make setup-venv",
        ))
    return s


def check_models() -> Section:
    s = Section("Models")
    flags = _engine_flags()
    chat_file = flags.get("MODEL_FILE", "hermes-8b-q4_k_m.gguf")
    embed_file = flags.get("EMBED_MODEL_FILE", "nomic-embed-text-v1.5.f16.gguf")
    for label, fname, fix_target, expected_min_mb in [
        ("chat model", chat_file, "fetch-model", 3 * 1024),
        ("embed model", embed_file, "fetch-embed-model", 100),
    ]:
        path = REPO_ROOT / "models" / fname
        if not path.is_file():
            s.add(Result(label, FAIL, f"{path.relative_to(REPO_ROOT)} missing",
                         fix=f"make {fix_target}"))
            continue
        size_mb = path.stat().st_size // (1024 * 1024)
        if size_mb < expected_min_mb:
            s.add(Result(
                label, FAIL,
                f"{path.relative_to(REPO_ROOT)} is only {size_mb} MiB (probably truncated)",
                fix=f"rm {path} && make {fix_target}",
            ))
        else:
            s.add(Result(label, OK, f"{path.relative_to(REPO_ROOT)} ({size_mb} MiB)"))
    return s


def check_config() -> Section:
    s = Section("Configuration")
    flags = _engine_flags()
    topo = _host_topology()

    if topo:
        s.add(Result(
            "host_topology",
            OK,
            f"tier={topo.get('TIER','?')} ctx={topo.get('CONTEXT_TOKENS','?')} "
            f"threads={topo.get('THREAD_COUNT','?')}",
        ))
    else:
        s.add(Result(
            "host_topology", WARN,
            "config/host_topology.env not generated yet",
            fix="make check-env",
        ))

    vault = _vault_path()
    if vault is None:
        s.add(Result(
            "vault path", FAIL,
            "HERMES_VAULT_PATH not set and no watcher plist found",
            fix='export HERMES_VAULT_PATH=~/Documents/Obsidian   # then: make install-watcher-daemon',
        ))
    elif not vault.is_dir():
        s.add(Result(
            "vault path", FAIL,
            f"{vault} does not exist",
            fix=f"mkdir -p {vault}   # or point HERMES_VAULT_PATH at an existing vault",
        ))
    else:
        try:
            md_count = sum(1 for _ in vault.rglob("*.md"))
        except OSError as exc:
            md_count = -1
            s.add(Result("vault path", WARN, f"{vault} ({exc})"))
        else:
            if md_count == 0:
                s.add(Result(
                    "vault path", WARN,
                    f"{vault} exists but contains no .md files",
                    fix="add a Markdown note (the watcher only indexes .md / .markdown).",
                ))
            else:
                s.add(Result("vault path", OK, f"{vault} ({md_count} markdown files)"))

    # Port collision detection: launchd KeepAlive will respawn-loop if another
    # process owns 8080/8081. `lsof +c 0` disables the default 9-char COMMAND
    # truncation that turns "llama-server" into "llama-ser" and trips a naive
    # substring match.
    for label, port in [("engine port", _engine_port()), ("embed port", _embed_port())]:
        if not _port_listening(port):
            continue  # nothing on the port — handled by the Servers section.
        rc, out, _ = _run(
            ["lsof", "-nP", "+c", "0", "-iTCP:%d" % port, "-sTCP:LISTEN"],
            timeout=2.0,
        )
        if rc != 0 or not out.strip():
            continue  # lsof unavailable or no permission — skip silently.
        lines = out.strip().splitlines()
        if len(lines) < 2:
            continue
        owner = lines[1].split()[0]
        if owner != "llama-server":
            s.add(Result(
                label, FAIL,
                f"port {port} is held by {owner}, not llama-server",
                fix=f"stop the conflicting process or change "
                    f"{'ENGINE_PORT' if 'engine' in label else 'EMBED_PORT'} "
                    f"in config/engine_flags.env",
            ))
    return s


def check_agents() -> Section:
    s = Section("LaunchAgents")
    labels = [
        ("com.hermes.metal.engine", "install-engine-daemon"),
        ("com.hermes.metal.embed", "install-embed-daemon"),
        ("com.hermes.metal.watcher", "install-watcher-daemon"),
    ]
    home_la = Path.home() / "Library" / "LaunchAgents"
    uid = os.getuid()
    rc, listing, _ = _run(["launchctl", "print", f"gui/{uid}"], timeout=5.0)
    have_print = rc == 0
    for label, target in labels:
        plist = home_la / f"{label}.plist"
        if not plist.is_file():
            s.add(Result(label, FAIL, "plist not installed", fix=f"make {target}"))
            continue
        if have_print:
            # Anchor on word boundaries so `com.hermes.metal.engine` doesn't
            # match the prefix of an unrelated `com.hermes.metal.engine.helper`
            # that someone might load alongside ours.
            pattern = re.compile(r"(?<![\w.])" + re.escape(label) + r"(?![\w.])")
            if pattern.search(listing):
                s.add(Result(label, OK, f"loaded in gui/{uid}"))
            else:
                s.add(Result(
                    label, WARN,
                    "plist on disk but not loaded by launchd",
                    fix=f"launchctl bootstrap gui/{uid} {plist}",
                ))
        else:
            # `launchctl print` failed (older macOS, sandboxed shell). Fall
            # back to a plain "file exists" check.
            s.add(Result(label, OK, "plist on disk (launchctl print unavailable)"))
    return s


def check_servers() -> Section:
    s = Section("Servers")
    chat_url_base = os.environ.get("HERMES_CHAT_URL", f"http://127.0.0.1:{_engine_port()}").rstrip("/")
    embed_url_raw = os.environ.get("HERMES_EMBED_URL", f"http://127.0.0.1:{_embed_port()}/v1/embeddings")
    # HERMES_EMBED_URL is the full /v1/embeddings URL by convention; trim back
    # to the base for /health probing. rstrip("/") first so trailing slashes
    # ("http://.../v1/embeddings/") don't bypass the suffix match and produce
    # double-slash probes downstream.
    embed_url_base = embed_url_raw.rstrip("/")
    for suffix in ("/v1/embeddings", "/embedding", "/embeddings"):
        if embed_url_base.endswith(suffix):
            embed_url_base = embed_url_base[: -len(suffix)]
            break

    for label, base, port, fix in [
        ("chat", chat_url_base, _engine_port(),
         f"launchctl kickstart -k gui/{os.getuid()}/com.hermes.metal.engine"),
        ("embed", embed_url_base, _embed_port(),
         f"launchctl kickstart -k gui/{os.getuid()}/com.hermes.metal.embed"),
    ]:
        # Step 1: is anything bound to the port at all?
        if not _port_listening(port):
            s.add(Result(f"{label} server", FAIL, f"nothing listening on 127.0.0.1:{port}", fix=fix))
            continue

        # Step 2: probe /health and surface llama-server's documented states.
        status, body, err = _http_get_json(f"{base}/health", timeout=3.0)
        if status is None:
            s.add(Result(f"{label} server", FAIL,
                         f"{base}/health unreachable: {err}", fix=fix))
            continue
        if status == 503 or (isinstance(body, dict) and body.get("status") == "loading model"):
            progress = (body or {}).get("progress")
            extra = f" ({int(progress * 100)}%)" if isinstance(progress, (int, float)) else ""
            s.add(Result(f"{label} server", WARN,
                         f"{base} is still loading model{extra}",
                         fix="wait ~10–60s for the model to mmap, then re-run."))
            continue
        if status >= 400:
            s.add(Result(f"{label} server", FAIL,
                         f"{base}/health returned HTTP {status}: {err}", fix=fix))
            continue
        ok_body = (isinstance(body, dict) and body.get("status") == "ok")
        s.add(Result(f"{label} server", OK if ok_body else WARN,
                     f"{base}/health → {body if body else err[:60]!r}"))
    return s


def check_index() -> Section:
    """Probe LanceDB without importing lancedb.

    Doctor stays stdlib-only, so we infer index population from the on-disk
    layout: a non-empty index has fragment files under `<table>.lance/data/`.

    Caveat: `LanceVault.__init__` (src/backend/database.py) creates the table
    on every connect, so the on-disk presence of `vault_chunks.lance/` only
    proves "the watcher (or some client) reached the DB at least once" — not
    that any vault content has been indexed. We surface "no fragments" as
    WARN with the same remediation either way.
    """
    s = Section("Index")
    db = _lancedb_path()
    if not db.exists():
        s.add(Result(
            "lancedb dir", WARN,
            f"{db} does not exist yet",
            fix="edit a .md file in your vault, or run: launchctl kickstart -k "
                f"gui/{os.getuid()}/com.hermes.metal.watcher",
        ))
        return s

    table = db / "vault_chunks.lance"
    if not table.is_dir():
        s.add(Result(
            "lancedb table", WARN,
            f"{table.name} not created yet",
            fix="edit a .md note in your vault to trigger an index write.",
        ))
        return s

    # Lance stores fragments under data/. Counting fragments avoids the
    # lancedb dependency. Note: a freshly-created empty table has no data/
    # at all, so "missing or empty" both mean "nothing indexed yet."
    data_dir = table / "data"
    fragment_count = 0
    if data_dir.is_dir():
        try:
            fragment_count = sum(1 for p in data_dir.iterdir() if p.is_file())
        except OSError:
            fragment_count = -1

    if fragment_count == 0:
        s.add(Result(
            "index rows", WARN,
            "table exists but has no data fragments yet",
            fix="edit a .md note in your vault to trigger an index write.",
        ))
    else:
        try:
            shown = table.relative_to(REPO_ROOT)
        except ValueError:
            shown = table
        s.add(Result(
            "index rows", OK,
            f"{fragment_count} fragment file(s) under {shown}",
        ))
    return s


# ------------------------------------------------------------------ render


_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_GREEN = "\033[32m" if _COLOR else ""
_YELLOW = "\033[33m" if _COLOR else ""
_RED = "\033[31m" if _COLOR else ""
_DIM = "\033[2m" if _COLOR else ""
_BOLD = "\033[1m" if _COLOR else ""
_RESET = "\033[0m" if _COLOR else ""

_BADGE = {
    OK: f"{_GREEN}OK  {_RESET}",
    WARN: f"{_YELLOW}WARN{_RESET}",
    FAIL: f"{_RED}FAIL{_RESET}",
    SKIP: f"{_DIM}SKIP{_RESET}",
}


def _print_section(sec: Section) -> None:
    print(f"{_BOLD}{sec.title}{_RESET}")
    for r in sec.results:
        line = f"  [{_BADGE[r.level]}] {r.name}"
        if r.detail:
            line += f"  {_DIM}{r.detail}{_RESET}"
        print(line)
        if r.fix and r.level in (FAIL, WARN):
            print(f"           {_DIM}fix:{_RESET} {r.fix}")
    print()


def _summary_line(sections: list[Section]) -> tuple[str, str]:
    counts = {OK: 0, WARN: 0, FAIL: 0, SKIP: 0}
    for sec in sections:
        for r in sec.results:
            counts[r.level] += 1
    overall = max(counts, key=lambda lv: _RANK[lv] if counts[lv] else -1)
    parts = [f"{counts[OK]} ok", f"{counts[WARN]} warn", f"{counts[FAIL]} fail"]
    if counts[SKIP]:
        parts.append(f"{counts[SKIP]} skip")
    summary = ", ".join(parts)
    if counts[FAIL]:
        return FAIL, f"{_RED}{_BOLD}hermes-metal: NOT READY{_RESET} — {summary}"
    if counts[WARN]:
        return WARN, f"{_YELLOW}{_BOLD}hermes-metal: degraded{_RESET} — {summary}"
    return OK, f"{_GREEN}{_BOLD}hermes-metal: ready{_RESET} — {summary}"


# --------------------------------------------------------------------- run


def run_all() -> list[Section]:
    return [
        check_host(),
        check_repo(),
        check_build(),
        check_python(),
        check_models(),
        check_config(),
        check_agents(),
        check_servers(),
        check_index(),
    ]


def run(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="hermes doctor",
        description="Diagnose hermes-metal. Exit 0 if healthy, 1 on FAIL.",
    )
    p.add_argument("--json", action="store_true",
                   help="emit machine-readable JSON instead of the colored report.")
    args = p.parse_args(argv)

    started = time.monotonic()
    sections = run_all()
    elapsed_ms = int((time.monotonic() - started) * 1000)

    if args.json:
        out = {
            "elapsed_ms": elapsed_ms,
            "sections": [
                {
                    "title": s.title,
                    "worst": s.worst,
                    "results": [r.to_dict() for r in s.results],
                }
                for s in sections
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"{_BOLD}hermes-metal doctor{_RESET}  ({elapsed_ms} ms)\n")
        for sec in sections:
            _print_section(sec)
        overall, line = _summary_line(sections)
        print(line)

    # Exit non-zero only on hard FAIL — warnings are informational.
    overall = max(
        (r.level for sec in sections for r in sec.results),
        key=lambda lv: _RANK[lv],
        default=OK,
    )
    return 1 if overall == FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
