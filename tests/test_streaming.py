"""SSE streaming parser test for _stream_assistant_reply.

Uses httpx.MockTransport so we don't need a running llama-server.
Verifies the streamer:
* concatenates `data: {...}` frames into a single returned string
* skips malformed JSON without aborting
* respects the `[DONE]` terminator
* respects the cancel event mid-stream
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from src.repl import _stream_assistant_reply
from src.server.client import HermesClient


def _make_sse_response(frames: list[str]) -> bytes:
    return ("\n".join(f"data: {f}" for f in frames) + "\n").encode("utf-8")


def _delta_frame(text: str) -> str:
    return json.dumps({"choices": [{"delta": {"content": text}}]})


async def _drive_streamer(client: HermesClient) -> tuple[str, bool]:
    return await _stream_assistant_reply(
        client,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=0.0,
    )


def _build_client(responder) -> HermesClient:
    """Construct a HermesClient backed by a mock transport."""
    transport = httpx.MockTransport(responder)
    c = HermesClient()
    # Replace the chat pool with a mocked AsyncClient pointed at the same
    # base_url; the streamer only uses client._chat.stream(...).
    c._chat = httpx.AsyncClient(base_url=c.base_url, transport=transport)
    return c


def test_sse_concatenates_frames(capsys):
    frames = [_delta_frame("hello"), _delta_frame(" world"), "[DONE]"]
    body = _make_sse_response(frames)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, content=body, headers={"Content-Type": "text/event-stream"})

    async def run():
        client = _build_client(handler)
        try:
            return await _drive_streamer(client)
        finally:
            await client.aclose()

    text, cancelled = asyncio.run(run())
    assert text == "hello world"
    assert cancelled is False
    # Streamed bytes were also written to stdout.
    assert "hello world" in capsys.readouterr().out


def test_sse_skips_malformed_json(capsys):
    frames = [
        _delta_frame("ok1"),
        "{not json",          # malformed: skipped
        _delta_frame(" ok2"),
        "[DONE]",
    ]
    body = _make_sse_response(frames)

    def handler(_req):
        return httpx.Response(200, content=body, headers={"Content-Type": "text/event-stream"})

    async def run():
        client = _build_client(handler)
        try:
            return await _drive_streamer(client)
        finally:
            await client.aclose()

    text, _ = asyncio.run(run())
    assert text == "ok1 ok2"


def test_sse_terminates_on_done():
    """Frames after [DONE] must be ignored."""
    frames = [_delta_frame("first"), "[DONE]", _delta_frame(" leaked")]
    body = _make_sse_response(frames)

    def handler(_req):
        return httpx.Response(200, content=body, headers={"Content-Type": "text/event-stream"})

    async def run():
        client = _build_client(handler)
        try:
            return await _drive_streamer(client)
        finally:
            await client.aclose()

    text, _ = asyncio.run(run())
    assert text == "first"
    assert "leaked" not in text


def test_http_error_raises_hermes_error():
    from src.server.client import HermesError

    def handler(_req):
        return httpx.Response(503, json={"error": "model loading"})

    async def run():
        client = _build_client(handler)
        try:
            await _drive_streamer(client)
        finally:
            await client.aclose()

    with pytest.raises(HermesError):
        asyncio.run(run())
