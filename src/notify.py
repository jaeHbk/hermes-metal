"""Telegram transport for hermes-metal — Phase A of the daily-summary cascade.

This module is the wire-level primitive. Higher layers (Phase D summarizer)
will call ``send(text)``; the rest of the file exists so the user can
configure a bot in 30 seconds and the doctor can validate the config.

Design choices:

* **Stdlib only.** Doctor links against this and stays stdlib-clean; tests
  mock at the urllib layer rather than fighting an httpx surface.
* **Config precedence:** env vars (``HERMES_TELEGRAM_BOT_TOKEN`` /
  ``HERMES_TELEGRAM_CHAT_ID``) win over the on-disk file. One-off
  testing without rewriting the config; permanent install via ``--setup``.
* **Config at ``~/.hermes/telegram.json``** (mode 0600). The bot token
  is roughly equivalent to a password — anyone with it can post as your
  bot to any chat that's added it. We don't put it in the repo or in
  any LaunchAgent plist.
* **Plain text only in Phase A.** Telegram's MarkdownV2 has nontrivial
  escape rules; switching to ``parse_mode=HTML`` (Phase D) is the
  right move when we need formatting. For now ``send()`` posts as plain
  text, which is escape-safe by definition.
* **Long-message chunking.** Telegram caps ``sendMessage`` at 4096 UTF-16
  code units. We split on paragraph boundaries when over the cap. A
  cleaner Phase D approach is to send a short headline + ``sendDocument``
  for the body — this module exposes ``send_document`` for that day.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------- types


CONFIG_PATH = Path.home() / ".hermes" / "telegram.json"
TELEGRAM_API = "https://api.telegram.org"

# 4096 is the documented hard cap on text length. We chunk slightly under
# to leave room for the trailing "(N/M)" pagination marker.
MAX_TEXT_CHARS = 4000


class NotifyError(Exception):
    """Base for any failure talking to Telegram."""


class NotConfiguredError(NotifyError):
    """No bot token or chat_id resolved from env or config file."""


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str
    chat_id: str  # str because chat_ids can be negative ints (groups) — keep verbatim


# --------------------------------------------------------------- config


def _read_config_file() -> dict[str, str]:
    if not CONFIG_PATH.is_file():
        return {}
    try:
        text = CONFIG_PATH.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for k in ("bot_token", "chat_id"):
        v = data.get(k)
        if isinstance(v, (str, int)):
            out[k] = str(v)
    return out


def load_config() -> TelegramConfig:
    """Resolve config from env > file. Raises NotConfiguredError otherwise.

    The two pieces (token, chat_id) can come from different sources — env
    token + file chat_id is fine. We document this in the docstring rather
    than mandating same-source because it's the natural debug pattern
    (``HERMES_TELEGRAM_BOT_TOKEN=... hermes notify --check``).
    """
    file_data = _read_config_file()
    token = os.environ.get("HERMES_TELEGRAM_BOT_TOKEN") or file_data.get("bot_token")
    chat_id = os.environ.get("HERMES_TELEGRAM_CHAT_ID") or file_data.get("chat_id")
    if not token or not chat_id:
        raise NotConfiguredError(
            "Telegram not configured. Run `hermes notify --setup`, "
            "or export HERMES_TELEGRAM_BOT_TOKEN and HERMES_TELEGRAM_CHAT_ID."
        )
    return TelegramConfig(bot_token=token, chat_id=str(chat_id))


def is_configured() -> bool:
    try:
        load_config()
    except NotConfiguredError:
        return False
    return True


def write_config(token: str, chat_id: str) -> Path:
    """Write the resolved token + chat_id to ``~/.hermes/telegram.json``.

    Mode 0600 so a multi-user box doesn't leak the bot token to other
    accounts. We pre-create the parent dir with mode 0700 for the same
    reason.
    """
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Use os.open + O_WRONLY|O_CREAT|O_TRUNC + mode=0o600 to set the mode
    # at create time; otherwise the file briefly exists with the umask
    # default before chmod tightens it.
    fd = os.open(
        CONFIG_PATH,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"bot_token": token, "chat_id": chat_id}, fh, indent=2)
            fh.write("\n")
    except Exception:
        # If something went wrong mid-write, don't leave a partial file
        # readable to anyone — best-effort delete and re-raise.
        try:
            CONFIG_PATH.unlink()
        except OSError:
            pass
        raise
    # Defense in depth: re-chmod even though O_CREAT already set the mode
    # (umask interacts with old kernels; cheap to be paranoid).
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass
    return CONFIG_PATH


# ---------------------------------------------------- transport (stdlib)


def _telegram_request(
    bot_token: str,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """POST to api.telegram.org/bot<token>/<method>, return the parsed body.

    Telegram returns ``{"ok": true, "result": ...}`` on success and
    ``{"ok": false, "error_code": N, "description": "..."}`` on failure.
    We surface failures as NotifyError so callers don't have to inspect
    the schema.
    """
    url = f"{TELEGRAM_API}/bot{bot_token}/{method}"
    data = urllib.parse.urlencode(params or {}).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        # Telegram puts the structured error in the body even on 4xx.
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        try:
            parsed = json.loads(body)
            description = parsed.get("description", body[:200])
            error_code = parsed.get("error_code", exc.code)
        except json.JSONDecodeError:
            description, error_code = body[:200], exc.code
        # 429 carries Retry-After in parameters; surface it so callers can
        # back off intelligently if they ever loop.
        if exc.code == 429:
            try:
                retry_after = parsed.get("parameters", {}).get("retry_after")
            except Exception:  # noqa: BLE001
                retry_after = None
            raise NotifyError(
                f"telegram {method}: 429 rate-limited"
                f"{f' (retry after {retry_after}s)' if retry_after else ''}: {description}"
            )
        raise NotifyError(f"telegram {method}: HTTP {error_code}: {description}")
    except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as exc:
        raise NotifyError(f"telegram {method}: transport error: {exc}")
    except ValueError as exc:
        # urlopen / Request raise ValueError on malformed URLs (e.g. an
        # empty token would build "/bot/method" which the server rejects
        # before this, but defensive in any case).
        raise NotifyError(f"telegram {method}: bad URL: {exc}")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NotifyError(f"telegram {method}: non-JSON body: {exc}")
    if not isinstance(body, dict) or not body.get("ok"):
        desc = body.get("description") if isinstance(body, dict) else "unknown"
        code = body.get("error_code") if isinstance(body, dict) else "?"
        raise NotifyError(f"telegram {method}: ok=false ({code}): {desc}")
    return body


# ------------------------------------------------------------ validation


def validate_token(bot_token: str) -> dict[str, Any]:
    """Call ``getMe`` and return the bot's profile.

    Used by ``--setup`` and by doctor. A 401 surfaces as NotifyError,
    which both call sites translate into a friendly message.
    """
    body = _telegram_request(bot_token, "getMe")
    return body.get("result") or {}


def validate_chat(bot_token: str, chat_id: str) -> dict[str, Any]:
    """Call ``getChat`` for the configured chat to confirm reachability.

    The bot must have been added to the chat at least once (or the user
    has DM'd it) for this to succeed.
    """
    body = _telegram_request(bot_token, "getChat", {"chat_id": chat_id})
    return body.get("result") or {}


# ------------------------------------------------------------------ send


def _split_for_telegram(text: str, *, limit: int = MAX_TEXT_CHARS) -> list[str]:
    """Split ``text`` into chunks of at most ``limit`` chars.

    Prefers paragraph boundaries (blank lines) so the recipient sees
    coherent sections. Falls back to a hard slice when a single
    paragraph exceeds the limit. Returns ``[]`` for empty input — Telegram
    rejects ``sendMessage`` with empty text, so callers must treat zero
    chunks as "nothing to send."
    """
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in text.split("\n\n"):
        # +2 for the "\n\n" we'll re-insert when joining.
        sep = 2 if current else 0
        if current_len + sep + len(para) > limit:
            if current:
                parts.append("\n\n".join(current))
                current, current_len = [], 0
            if len(para) > limit:
                # Very long single paragraph — slice hard.
                for i in range(0, len(para), limit):
                    parts.append(para[i:i + limit])
                continue
        current.append(para)
        current_len += sep + len(para)
    if current:
        parts.append("\n\n".join(current))
    # Tag with pagination markers so the recipient understands the order.
    if len(parts) <= 1:
        return parts
    total = len(parts)
    return [f"{p}\n\n({i + 1}/{total})" for i, p in enumerate(parts)]


def send(text: str, *, config: TelegramConfig | None = None) -> int:
    """Send ``text`` to the configured chat. Returns the number of chunks sent.

    Long messages are split on paragraph boundaries; each chunk is its own
    Telegram message with ``(i/n)`` pagination footer. Plain text only —
    no parse_mode set, so the recipient sees exactly what was sent.
    Empty / whitespace-only input is rejected (Telegram returns 400 on
    empty ``text``).
    """
    cfg = config or load_config()
    if not text or not text.strip():
        raise NotifyError("send: refusing to send empty message")
    chunks = _split_for_telegram(text)
    for chunk in chunks:
        _telegram_request(
            cfg.bot_token,
            "sendMessage",
            {"chat_id": cfg.chat_id, "text": chunk, "disable_web_page_preview": "true"},
            timeout=15.0,
        )
    return len(chunks)


def send_document(name: str, content: bytes, *, caption: str = "",
                  config: TelegramConfig | None = None) -> None:
    """Send ``content`` as a document attachment named ``name``.

    Phase D will use this to deliver the full markdown summary once the
    Phase A wire is proven. Implementation note: ``sendDocument`` requires
    multipart/form-data; we hand-roll it here so we don't pull in requests.
    """
    cfg = config or load_config()
    boundary = "----hermes-metal-" + str(time.time_ns())
    body = _multipart_body(boundary, {
        "chat_id": cfg.chat_id,
        "caption": caption,
    }, file_field="document", file_name=name, file_bytes=content)
    url = f"{TELEGRAM_API}/bot{cfg.bot_token}/sendDocument"
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise NotifyError(f"telegram sendDocument: HTTP {exc.code}: {body_text[:200]}")
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise NotifyError(f"telegram sendDocument: transport error: {exc}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NotifyError(f"telegram sendDocument: non-JSON body: {exc}")
    if not parsed.get("ok"):
        raise NotifyError(f"telegram sendDocument: ok=false: {parsed.get('description')}")


def _sanitize_filename(name: str) -> str:
    """Replace characters that would corrupt a Content-Disposition header.

    Per RFC 7578, a filename containing ``"``, ``\\r``, ``\\n``, or NUL
    breaks the multipart parser (or worse, lets the caller inject extra
    headers). We replace each with ``_`` rather than reject, because
    callers (e.g. summary.py) generate filenames from dates/topics and
    shouldn't have to worry about escaping.
    """
    out = []
    for ch in name:
        if ch in ('"', "\r", "\n", "\x00") or ord(ch) < 0x20:
            out.append("_")
        else:
            out.append(ch)
    cleaned = "".join(out).strip()
    return cleaned or "document"


def _multipart_body(boundary: str, fields: dict[str, str], *,
                    file_field: str, file_name: str, file_bytes: bytes) -> bytes:
    """Hand-rolled multipart/form-data body for sendDocument."""
    safe_name = _sanitize_filename(file_name)
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8")
        )
        parts.append(value.encode("utf-8"))
        parts.append(b"\r\n")
    parts.append(f"--{boundary}\r\n".encode("utf-8"))
    parts.append(
        f'Content-Disposition: form-data; name="{file_field}"; '
        f'filename="{safe_name}"\r\n'.encode("utf-8")
    )
    parts.append(b"Content-Type: application/octet-stream\r\n\r\n")
    parts.append(file_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts)


# ----------------------------------------------------- chat-id discovery


def poll_for_chat_id(bot_token: str, *, timeout_s: float = 60.0,
                     poll_interval_s: float = 2.0) -> str:
    """Wait for the user to DM the bot, return their chat_id.

    Telegram's ``getUpdates`` long-poll returns recent messages. We loop
    until we see one or the user gives up (Ctrl-C / timeout). The user
    DMing the bot first is unavoidable — bots cannot initiate
    conversations on Telegram.
    """
    deadline = time.monotonic() + timeout_s
    last_update_id: int | None = None
    while time.monotonic() < deadline:
        params: dict[str, Any] = {"timeout": str(int(min(poll_interval_s, 5)))}
        if last_update_id is not None:
            # Telegram acks consumed updates by passing offset = last_id+1.
            params["offset"] = str(last_update_id + 1)
        try:
            body = _telegram_request(bot_token, "getUpdates", params, timeout=poll_interval_s + 5.0)
        except NotifyError:
            # Transient transport hiccup — retry until the deadline.
            time.sleep(poll_interval_s)
            continue
        results = body.get("result") or []
        # Advance the offset across the WHOLE batch first. Telegram only acks
        # consumed updates when the next request supplies `offset = last+1`;
        # if we return early without bumping last_update_id, the same backlog
        # comes back on the next process run forever.
        found_chat_id: str | None = None
        for update in results:
            uid = update.get("update_id")
            if isinstance(uid, int) and (last_update_id is None or uid > last_update_id):
                last_update_id = uid
            if found_chat_id is None:
                message = update.get("message") or update.get("channel_post")
                if message and message.get("chat", {}).get("id") is not None:
                    found_chat_id = str(message["chat"]["id"])
        if found_chat_id is not None:
            # Best-effort ack so subsequent --setup runs don't replay this
            # message. A failure here is fine — we already have the chat_id.
            if last_update_id is not None:
                try:
                    _telegram_request(
                        bot_token, "getUpdates",
                        {"offset": str(last_update_id + 1), "timeout": "0"},
                        timeout=5.0,
                    )
                except NotifyError:
                    pass
            return found_chat_id
        time.sleep(poll_interval_s)
    raise NotifyError(
        f"poll_for_chat_id: no message received within {int(timeout_s)}s. "
        "Open Telegram, DM your bot, and re-run --setup."
    )


# ------------------------------------------------------------------- CLI


def _setup_interactive() -> int:
    """Interactive ``hermes notify --setup`` flow."""
    print("hermes notify — Telegram bot setup", file=sys.stderr)
    print("", file=sys.stderr)
    print("1. Create a bot: open @BotFather in Telegram, /newbot, follow prompts.", file=sys.stderr)
    print("2. BotFather gives you a token like 1234:AAA-bbb-... — paste it below.", file=sys.stderr)
    print("", file=sys.stderr)

    # Read token. Use input() rather than getpass: the token is also visible
    # in the user's shell history if they ever paste it on a command line,
    # so getpass is security theater here. Visible echo lets the user
    # confirm they pasted correctly.
    try:
        token = input("Bot token: ").strip()
    except EOFError:
        print("hermes notify: no input received.", file=sys.stderr)
        return 1
    if not token:
        print("hermes notify: empty token.", file=sys.stderr)
        return 1

    # Validate the token before going further.
    try:
        bot = validate_token(token)
    except NotifyError as exc:
        print(f"hermes notify: token rejected by Telegram: {exc}", file=sys.stderr)
        print("       Double-check you copied the full token from BotFather.", file=sys.stderr)
        return 1
    bot_username = bot.get("username", "?")
    print(f"  ✓ token OK — bot is @{bot_username}", file=sys.stderr)
    print("", file=sys.stderr)
    print(f"3. In Telegram, search for @{bot_username} and send it ANY message.", file=sys.stderr)
    print("   This is required: bots cannot DM you first.", file=sys.stderr)
    print("   Waiting up to 60s for your message...", file=sys.stderr)

    try:
        chat_id = poll_for_chat_id(token, timeout_s=60.0)
    except NotifyError as exc:
        print(f"hermes notify: {exc}", file=sys.stderr)
        return 1
    print(f"  ✓ chat_id captured: {chat_id}", file=sys.stderr)

    path = write_config(token, chat_id)
    print(f"  ✓ wrote {path} (mode 0600)", file=sys.stderr)

    # Confirmation message so the user sees end-to-end delivery.
    try:
        send("hermes-metal: notifications are wired up.",
             config=TelegramConfig(bot_token=token, chat_id=chat_id))
        print("  ✓ test message sent — check your Telegram client.", file=sys.stderr)
    except NotifyError as exc:
        print(f"  ! send failed (config saved anyway): {exc}", file=sys.stderr)
        return 1
    return 0


def _check() -> int:
    """``hermes notify --check`` — exit 0 healthy, 1 broken, 2 unconfigured."""
    try:
        cfg = load_config()
    except NotConfiguredError as exc:
        print(f"hermes notify: {exc}", file=sys.stderr)
        return 2
    try:
        bot = validate_token(cfg.bot_token)
    except NotifyError as exc:
        print(f"hermes notify: token: {exc}", file=sys.stderr)
        return 1
    try:
        chat = validate_chat(cfg.bot_token, cfg.chat_id)
    except NotifyError as exc:
        print(f"hermes notify: chat: {exc}", file=sys.stderr)
        return 1
    print(
        f"hermes notify: OK — bot @{bot.get('username','?')} "
        f"-> chat {chat.get('type','?')} "
        f"({chat.get('title') or chat.get('username') or chat.get('first_name','?')})"
    )
    return 0


def run(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="hermes notify",
        description="Send a message to Telegram, or set up the bot.",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--setup", action="store_true",
                   help="Interactive bot+chat configuration (writes ~/.hermes/telegram.json).")
    g.add_argument("--check", action="store_true",
                   help="Validate the configured bot+chat. Exit 0 OK, 1 broken, 2 unconfigured.")
    p.add_argument("message", nargs="?", default=None,
                   help="Message to send. Required unless --setup or --check.")
    args = p.parse_args(argv)

    # Reject stray positionals on --setup/--check before dispatching, so a
    # user typo'ing `hermes notify --check "ignored"` gets a clear error
    # instead of the message being silently dropped on the floor.
    if (args.setup or args.check) and args.message is not None:
        p.error("--setup and --check do not take a message argument.")
    if args.setup:
        return _setup_interactive()
    if args.check:
        return _check()
    if args.message is None:
        p.error("provide a message, or pass --setup / --check.")
    try:
        n = send(args.message)
    except NotConfiguredError as exc:
        print(f"hermes notify: {exc}", file=sys.stderr)
        return 2
    except NotifyError as exc:
        print(f"hermes notify: {exc}", file=sys.stderr)
        return 1
    if n > 1:
        print(f"sent in {n} chunks", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
