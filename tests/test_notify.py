"""Tests for src/notify.py — the Telegram transport.

We mock at the urllib layer so the suite stays offline. Each test sets up
a custom ``urlopen`` stand-in that asserts the request shape and returns
the shape Telegram would produce.
"""
from __future__ import annotations

import io
import json
import os
import urllib.error
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src import notify
from src.notify import (
    MAX_TEXT_CHARS,
    NotConfiguredError,
    NotifyError,
    TelegramConfig,
    _split_for_telegram,
    is_configured,
    load_config,
    send,
    validate_chat,
    validate_token,
    write_config,
)


# ----------------------------------------------------------- urlopen mock


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ok(payload: dict) -> _FakeResponse:
    return _FakeResponse(json.dumps({"ok": True, "result": payload}).encode("utf-8"))


def _http_error(code: int, body: dict) -> urllib.error.HTTPError:
    fp = io.BytesIO(json.dumps(body).encode("utf-8"))
    return urllib.error.HTTPError("https://api.telegram.org/", code, "err", {}, fp)


@contextmanager
def fake_urlopen(handler):
    """Patch urllib.request.urlopen with a callable that takes the Request
    and returns a response (or raises). Yields a list collecting Requests."""
    seen: list = []

    def _impl(req, *args, **kwargs):
        seen.append(req)
        result = handler(req)
        if isinstance(result, BaseException):
            raise result
        return result

    with patch.object(notify.urllib.request, "urlopen", _impl):
        yield seen


# --------------------------------------------------------------- config


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Redirect ~/.hermes/telegram.json into tmp_path so tests don't read
    or clobber the user's real config."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(notify, "CONFIG_PATH", fake_home / ".hermes" / "telegram.json")
    monkeypatch.delenv("HERMES_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("HERMES_TELEGRAM_CHAT_ID", raising=False)
    return fake_home


def test_load_config_unconfigured_raises(isolated_home):
    with pytest.raises(NotConfiguredError):
        load_config()


def test_load_config_from_env(isolated_home, monkeypatch):
    monkeypatch.setenv("HERMES_TELEGRAM_BOT_TOKEN", "TKN")
    monkeypatch.setenv("HERMES_TELEGRAM_CHAT_ID", "1234")
    cfg = load_config()
    assert cfg.bot_token == "TKN"
    assert cfg.chat_id == "1234"


def test_load_config_from_file(isolated_home):
    notify.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    notify.CONFIG_PATH.write_text(json.dumps({"bot_token": "FILE_TOK", "chat_id": "999"}))
    cfg = load_config()
    assert cfg.bot_token == "FILE_TOK"
    assert cfg.chat_id == "999"


def test_env_overrides_file(isolated_home, monkeypatch):
    notify.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    notify.CONFIG_PATH.write_text(json.dumps({"bot_token": "FILE_TOK", "chat_id": "999"}))
    monkeypatch.setenv("HERMES_TELEGRAM_BOT_TOKEN", "ENV_TOK")
    cfg = load_config()
    # Env wins for token; file fills in the chat_id.
    assert cfg.bot_token == "ENV_TOK"
    assert cfg.chat_id == "999"


def test_malformed_config_falls_through(isolated_home):
    notify.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    notify.CONFIG_PATH.write_text("{not valid json")
    with pytest.raises(NotConfiguredError):
        load_config()


def test_is_configured(isolated_home, monkeypatch):
    assert not is_configured()
    monkeypatch.setenv("HERMES_TELEGRAM_BOT_TOKEN", "TKN")
    monkeypatch.setenv("HERMES_TELEGRAM_CHAT_ID", "1")
    assert is_configured()


def test_write_config_mode_0600(isolated_home):
    path = write_config("ABC", "42")
    assert path.is_file()
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600
    parent_mode = path.parent.stat().st_mode & 0o777
    # Parent dir should be at most 0o700 (we created with that mode; user
    # umask doesn't loosen owner-only perms).
    assert parent_mode & 0o077 == 0  # no group/other access
    data = json.loads(path.read_text())
    assert data == {"bot_token": "ABC", "chat_id": "42"}


def test_write_config_overwrites(isolated_home):
    write_config("OLD", "1")
    write_config("NEW", "2")
    data = json.loads(notify.CONFIG_PATH.read_text())
    assert data == {"bot_token": "NEW", "chat_id": "2"}


# ---------------------------------------------------------- transport


def test_validate_token_success():
    def handler(req):
        assert "/botTKN/getMe" in req.full_url
        return _ok({"id": 1, "username": "hermes_bot"})

    with fake_urlopen(handler):
        bot = validate_token("TKN")
    assert bot["username"] == "hermes_bot"


def test_validate_token_401():
    def handler(req):
        return _http_error(401, {"ok": False, "error_code": 401, "description": "Unauthorized"})

    with fake_urlopen(handler):
        with pytest.raises(NotifyError, match="401"):
            validate_token("BAD")


def test_validate_chat_400():
    def handler(req):
        return _http_error(400, {
            "ok": False, "error_code": 400, "description": "Bad Request: chat not found"
        })

    with fake_urlopen(handler):
        with pytest.raises(NotifyError, match="chat not found"):
            validate_chat("TKN", "12345")


def test_send_success():
    cfg = TelegramConfig(bot_token="TKN", chat_id="42")

    def handler(req):
        body = req.data.decode("utf-8")
        # urlencoded chat_id and text
        assert "chat_id=42" in body
        assert "text=" in body
        return _ok({"message_id": 1})

    with fake_urlopen(handler) as seen:
        n = send("hello", config=cfg)
    assert n == 1
    assert len(seen) == 1


def test_send_429_rate_limit():
    cfg = TelegramConfig(bot_token="TKN", chat_id="42")

    def handler(req):
        return _http_error(429, {
            "ok": False, "error_code": 429,
            "description": "Too Many Requests",
            "parameters": {"retry_after": 7},
        })

    with fake_urlopen(handler):
        with pytest.raises(NotifyError, match="retry after 7s"):
            send("hello", config=cfg)


def test_send_long_message_chunks():
    cfg = TelegramConfig(bot_token="TKN", chat_id="42")
    big = ("paragraph " * 500 + "\n\n") * 3  # well over 4000 chars

    sent_texts: list[str] = []

    def handler(req):
        body = req.data.decode("utf-8")
        # Pull the urlencoded text out for size check
        from urllib.parse import parse_qs
        qs = parse_qs(body)
        sent_texts.append(qs["text"][0])
        return _ok({"message_id": 1})

    with fake_urlopen(handler):
        n = send(big, config=cfg)
    assert n >= 2
    # No single chunk exceeds the cap (after pagination footer).
    for t in sent_texts:
        assert len(t) <= MAX_TEXT_CHARS + 20  # +20 for "(i/n)" suffix


def test_split_for_telegram_short_string_unchanged():
    parts = _split_for_telegram("hi")
    assert parts == ["hi"]


def test_split_for_telegram_paginates_when_over_limit():
    text = ("a" * 100 + "\n\n") * 60  # 60 paragraphs of 100 chars each + sep
    parts = _split_for_telegram(text, limit=300)
    assert len(parts) > 1
    # Each part carries an "(i/n)" footer.
    for i, p in enumerate(parts, 1):
        assert f"({i}/{len(parts)})" in p


def test_split_for_telegram_hard_slices_long_paragraph():
    parts = _split_for_telegram("x" * 1000, limit=300)
    assert len(parts) >= 4  # ceil(1000/300) = 4
    for p in parts:
        # Each chunk (modulo footer) ≤ limit.
        assert len(p.replace("\n\n", "")) <= 320


def test_telegram_ok_false_surfaces():
    """Server returns 200 OK but {ok: false} — must still be NotifyError."""
    def handler(req):
        return _FakeResponse(json.dumps({"ok": False, "description": "weird"}).encode("utf-8"))

    with fake_urlopen(handler):
        with pytest.raises(NotifyError, match="weird"):
            validate_token("TKN")


def test_transport_error_surfaces():
    """A connection refusal / DNS failure / etc. must be NotifyError."""
    def handler(req):
        return urllib.error.URLError("Name or service not known")

    with fake_urlopen(handler):
        with pytest.raises(NotifyError, match="transport error"):
            validate_token("TKN")


# --------------------------------------------------------- run() entry


def test_run_message_without_config_returns_2(isolated_home, capsys):
    rc = notify.run(["hello"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not configured" in err.lower()


def test_run_check_unconfigured_returns_2(isolated_home):
    rc = notify.run(["--check"])
    assert rc == 2


def test_run_check_configured_healthy_returns_0(isolated_home, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TELEGRAM_BOT_TOKEN", "TKN")
    monkeypatch.setenv("HERMES_TELEGRAM_CHAT_ID", "42")

    def handler(req):
        if "getMe" in req.full_url:
            return _ok({"id": 1, "username": "hermes_bot"})
        if "getChat" in req.full_url:
            return _ok({"id": 42, "type": "private", "first_name": "Test"})
        raise AssertionError("unexpected: " + req.full_url)

    with fake_urlopen(handler):
        rc = notify.run(["--check"])
    assert rc == 0
    out = capsys.readouterr().out
    # Tighter substring than just "OK" — would otherwise match "NOT OK".
    assert "hermes notify: OK" in out
    assert "hermes_bot" in out


def test_run_check_token_ok_chat_404_returns_1(isolated_home, monkeypatch):
    """Token validates but chat is gone (user deleted the bot from the chat)."""
    monkeypatch.setenv("HERMES_TELEGRAM_BOT_TOKEN", "TKN")
    monkeypatch.setenv("HERMES_TELEGRAM_CHAT_ID", "42")

    def handler(req):
        if "getMe" in req.full_url:
            return _ok({"id": 1, "username": "hermes_bot"})
        if "getChat" in req.full_url:
            return _http_error(400, {
                "ok": False, "error_code": 400,
                "description": "Bad Request: chat not found",
            })
        raise AssertionError("unexpected: " + req.full_url)

    with fake_urlopen(handler):
        rc = notify.run(["--check"])
    assert rc == 1


def test_run_check_rejects_stray_positional(isolated_home, monkeypatch):
    """`hermes notify --check "ignored"` must error, not silently drop the arg."""
    monkeypatch.setenv("HERMES_TELEGRAM_BOT_TOKEN", "TKN")
    monkeypatch.setenv("HERMES_TELEGRAM_CHAT_ID", "42")
    with pytest.raises(SystemExit):
        notify.run(["--check", "ignored"])


def test_negative_chat_id_round_trips(isolated_home):
    """Telegram supergroups have negative chat IDs like -100123456789."""
    write_config("TKN", "-100123456789")
    cfg = load_config()
    assert cfg.chat_id == "-100123456789"


def test_run_check_token_revoked_returns_1(isolated_home, monkeypatch):
    monkeypatch.setenv("HERMES_TELEGRAM_BOT_TOKEN", "TKN")
    monkeypatch.setenv("HERMES_TELEGRAM_CHAT_ID", "42")

    def handler(req):
        return _http_error(401, {"ok": False, "error_code": 401, "description": "Unauthorized"})

    with fake_urlopen(handler):
        rc = notify.run(["--check"])
    assert rc == 1


def test_run_setup_and_check_mutually_exclusive(isolated_home):
    with pytest.raises(SystemExit):
        notify.run(["--setup", "--check"])


def test_run_no_args_errors(isolated_home):
    with pytest.raises(SystemExit):
        notify.run([])


def test_send_empty_string_rejected():
    cfg = TelegramConfig(bot_token="TKN", chat_id="42")
    with pytest.raises(NotifyError, match="empty"):
        send("", config=cfg)
    with pytest.raises(NotifyError, match="empty"):
        send("   \n  ", config=cfg)


def test_split_empty_string_yields_no_chunks():
    assert _split_for_telegram("") == []


# ---------------------------------------------------- poll_for_chat_id


def test_poll_for_chat_id_returns_first_message_chat_id():
    """Happy path: getUpdates returns one message with a chat_id."""
    chat_id_value = -100123456789  # supergroup-style negative ID

    calls: list[dict] = []

    def handler(req):
        body = req.data.decode("utf-8")
        from urllib.parse import parse_qs
        calls.append(parse_qs(body))
        # First call (no offset): return the message
        if "offset" not in body:
            return _ok([
                {"update_id": 5, "message": {"chat": {"id": chat_id_value}, "text": "hi"}},
            ])
        # Second call (the ack with offset=6): return empty
        return _ok([])

    with fake_urlopen(handler):
        result = notify.poll_for_chat_id("TKN", timeout_s=5.0, poll_interval_s=0.1)
    assert result == str(chat_id_value)
    # Two calls: the long-poll and the offset-ack.
    assert len(calls) == 2
    assert calls[1].get("offset") == ["6"]


def test_poll_for_chat_id_advances_offset_across_empty_results():
    """Bug guard: a non-message update advances offset so it isn't replayed."""
    handler_state = {"call": 0}
    seen_offsets: list = []

    def handler(req):
        from urllib.parse import parse_qs
        body = parse_qs(req.data.decode("utf-8"))
        handler_state["call"] += 1
        seen_offsets.append(body.get("offset"))
        if handler_state["call"] == 1:
            # Non-message update at update_id=10. Must NOT be returned.
            return _ok([{"update_id": 10, "edited_message": {"text": "edit"}}])
        if handler_state["call"] == 2:
            # Message at update_id=11. poll_for_chat_id should return "42".
            return _ok([{"update_id": 11, "message": {"chat": {"id": 42}}}])
        # Call 3 is the post-return ack (offset=12). Always succeed.
        return _ok([])

    with fake_urlopen(handler):
        result = notify.poll_for_chat_id("TKN", timeout_s=5.0, poll_interval_s=0.05)
    assert result == "42"
    # Call 1 had no offset; call 2 must have advanced past update_id=10.
    assert seen_offsets[0] is None
    assert seen_offsets[1] == ["11"]


def test_poll_for_chat_id_timeout():
    """Returns NotifyError if no message arrives before deadline."""
    def handler(req):
        return _ok([])

    with fake_urlopen(handler):
        with pytest.raises(NotifyError, match="no message received"):
            notify.poll_for_chat_id("TKN", timeout_s=0.5, poll_interval_s=0.1)


# ------------------------------------------------------ send_document


def test_send_document_success():
    cfg = TelegramConfig(bot_token="TKN", chat_id="42")
    captured = {}

    def handler(req):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = req.data
        return _ok({"message_id": 7})

    with fake_urlopen(handler):
        notify.send_document("summary.md", b"hello world", caption="hi", config=cfg)
    assert "/sendDocument" in captured["url"]
    ct = captured["headers"].get("Content-type") or captured["headers"].get("Content-Type")
    assert ct.startswith("multipart/form-data; boundary=")
    body = captured["body"]
    assert b'name="document"' in body
    assert b'filename="summary.md"' in body
    assert b"hello world" in body
    assert b'name="caption"' in body and b"hi" in body


def test_send_document_filename_sanitized():
    """A filename with embedded \\r\\n or \" must not corrupt the multipart frame."""
    cfg = TelegramConfig(bot_token="TKN", chat_id="42")
    captured = {}

    def handler(req):
        captured["body"] = req.data
        return _ok({"message_id": 1})

    with fake_urlopen(handler):
        notify.send_document(
            'evil"\r\nContent-Type: text/html\r\nname',
            b"x",
            config=cfg,
        )
    body = captured["body"].decode("utf-8", errors="replace")
    # Sanitized: no raw quote, no CR/LF in filename portion.
    cd_line = next(l for l in body.split("\r\n") if l.startswith('Content-Disposition: form-data; name="document"'))
    # Must not contain a raw " (other than the wrapping ones), \r, or \n.
    inner = cd_line.split('filename="', 1)[1].rsplit('"', 1)[0]
    assert '"' not in inner
    assert "\r" not in inner and "\n" not in inner


def test_send_document_http_error():
    cfg = TelegramConfig(bot_token="TKN", chat_id="42")

    def handler(req):
        return _http_error(413, {"ok": False, "description": "Request Entity Too Large"})

    with fake_urlopen(handler):
        with pytest.raises(NotifyError, match="413"):
            notify.send_document("big.bin", b"x" * 10, config=cfg)
