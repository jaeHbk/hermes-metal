"""Tests for doctor's strict-plist validation.

The LaunchAgents fail-to-launch investigation found that a literal ``--``
inside an XML comment makes a plist that ``plutil -lint`` accepts but launchd
rejects with exit 78. ``_strict_plist_ok`` (used by doctor and mirrored by the
Makefile install guard) catches that class of bug; these tests pin it down.
"""
from __future__ import annotations

from pathlib import Path

from src.doctor import _strict_plist_ok


_VALID = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.example.ok</string>
  <!-- a clean comment with no double hyphen -->
  <key>ProgramArguments</key><array><string>/bin/true</string></array>
</dict></plist>
"""

# A literal "--" inside a comment is illegal XML; expat (and launchd) reject it,
# even though `plutil -lint` is lenient enough to pass it.
_DOUBLE_HYPHEN_COMMENT = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.example.bad</string>
  <!-- pass the --flash-attn flag explicitly -->
  <key>ProgramArguments</key><array><string>/bin/true</string></array>
</dict></plist>
"""


def test_valid_plist_passes(tmp_path: Path):
    p = tmp_path / "ok.plist"
    p.write_text(_VALID)
    ok, err = _strict_plist_ok(p)
    assert ok is True
    assert err == ""


def test_double_hyphen_comment_rejected(tmp_path: Path):
    # This is the exact shape that caused the exit-78 LaunchAgent failure.
    p = tmp_path / "bad.plist"
    p.write_text(_DOUBLE_HYPHEN_COMMENT)
    ok, err = _strict_plist_ok(p)
    assert ok is False
    assert "well-formed" in err or "token" in err


def test_missing_file_is_not_ok(tmp_path: Path):
    ok, err = _strict_plist_ok(tmp_path / "nope.plist")
    assert ok is False
    assert err  # some error message


def test_shipped_templates_render_to_valid_xml(tmp_path: Path):
    # Every shipped plist template, with placeholders filled, must strict-parse
    # — a regression guard so a future comment edit can't reintroduce the bug.
    repo = Path(__file__).resolve().parents[1]
    subs = {
        "{WORKING_DIR}": "/w", "{MODEL_PATH}": "/m.gguf", "{EMBED_MODEL_PATH}": "/e.gguf",
        "{CONTEXT_TOKENS}": "32768", "{THREAD_COUNT}": "8", "{ENGINE_PORT}": "8080",
        "{EMBED_PORT}": "8081", "{CACHE_TYPE_K}": "q8_0", "{CACHE_TYPE_V}": "q8_0",
        "{SLOT_SAVE_PATH}": "storage/slots", "{VAULT_PATH}": "/v",
        "{DIGEST_HOUR}": "7", "{DIGEST_MINUTE}": "30", "{DIGEST_PUSH}": "0",
    }
    for name in ("daemon", "embed", "watcher", "digest"):
        tpl = repo / "config" / f"{name}.plist.template"
        if not tpl.is_file():
            continue
        text = tpl.read_text()
        for k, v in subs.items():
            text = text.replace(k, v)
        out = tmp_path / f"{name}.plist"
        out.write_text(text)
        ok, err = _strict_plist_ok(out)
        assert ok, f"{name}.plist.template renders to invalid XML: {err}"
