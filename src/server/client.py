"""HTTP client wrappers for the local llama.cpp servers used by hermes-metal.

Topology (per CLAUDE.md and the verified spec corrections):

* Chat / completion server  : ``http://127.0.0.1:8080``
    Same llama-server binary built with ``-DGGML_METAL=ON``; loads
    ``Hermes-3-Llama-3.1-8B.Q4_K_M.gguf``. Supports ``/health``,
    ``/completion``, ``/v1/chat/completions`` (OpenAI-compatible), and the
    ``/slots/{id}?action=save|restore|erase`` slot-persistence endpoints.

* Embedding server          : ``http://127.0.0.1:8081``
    A second llama-server process spawned with ``--embedding`` loading
    ``nomic-embed-text-v1.5.f16.gguf``. Exposes the native ``/embedding``
    endpoint (preferred for batch + ``--pooling none`` flexibility) and the
    OpenAI-compatible ``/v1/embeddings``.

Divergences from CLAUDE.md (intentional, see verified-facts):

* CLAUDE.md §2 references ``--prompt-cache``; that flag is *llama-cli only*.
  The server's persistence story is the per-slot save/restore endpoints,
  which is what :meth:`HermesClient.slot_save` / :meth:`HermesClient.slot_restore`
  target.
* CLAUDE.md §3 mentions ``http://localhost:8080/v1`` as a single endpoint; in
  practice we keep two distinct base URLs (chat on 8080, embeddings on 8081)
  so the embedding model can be ``--embedding``-mode-locked without affecting
  chat throughput.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncIterator

import httpx

__all__ = ["HermesClient", "HermesError"]


class HermesError(Exception):
    """Raised for any failure talking to a local llama.cpp server.

    Wraps transport errors, non-2xx HTTP responses, and malformed payloads so
    callers only need to catch a single exception type.
    """


# httpx.Timeout(connect=5.0, read=60.0); write/pool inherit the read timeout.
# 60s read accommodates first-token latency on cold KV state for an 8B model.
_DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=5.0)

# A single shared connection pool per client instance keeps TCP/keep-alive hot
# between rapid-fire daemon requests (file-watcher → embed → /completion).
_DEFAULT_LIMITS = httpx.Limits(
    max_connections=16,
    max_keepalive_connections=8,
    keepalive_expiry=30.0,
)


class HermesClient:
    """Async client for the local llama.cpp chat + embedding servers.

    Parameters
    ----------
    base_url:
        Chat / completion server (``llama-server`` loading Hermes-3-8B).
        Default ``http://127.0.0.1:8080``.
    embed_url:
        Embedding server (``llama-server --embedding`` loading
        ``nomic-embed-text-v1.5``). Default ``http://127.0.0.1:8081``.
    timeout:
        ``httpx.Timeout`` override; defaults to 5s connect / 60s read.

    Use as an async context manager to guarantee pool cleanup::

        async with HermesClient() as h:
            await h.health()
    """

    def __init__(
        self,
        base_url: str | None = None,
        embed_url: str | None = None,
        *,
        timeout: httpx.Timeout | None = None,
        limits: httpx.Limits | None = None,
    ) -> None:
        # Env-var overrides keep the daemon plist (and ad-hoc overrides) wired
        # to a single source of truth without each call site re-reading os.environ.
        resolved_base = base_url or os.environ.get(
            "HERMES_CHAT_URL", "http://127.0.0.1:8080"
        )
        resolved_embed = embed_url or os.environ.get(
            "HERMES_EMBED_URL", "http://127.0.0.1:8081"
        )
        self.base_url = resolved_base.rstrip("/")
        self.embed_url = resolved_embed.rstrip("/")
        self._timeout = timeout or _DEFAULT_TIMEOUT
        self._limits = limits or _DEFAULT_LIMITS

        # One pool for chat, one for embeddings. Separate clients keep
        # head-of-line blocking on the chat socket from stalling embed calls.
        self._chat = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self._timeout,
            limits=self._limits,
        )
        self._embed = httpx.AsyncClient(
            base_url=self.embed_url,
            timeout=self._timeout,
            limits=self._limits,
        )

    # ------------------------------------------------------------------
    # async context manager
    # ------------------------------------------------------------------
    async def __aenter__(self) -> "HermesClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close both connection pools. Idempotent."""
        await self._chat.aclose()
        await self._embed.aclose()

    # ------------------------------------------------------------------
    # health
    # ------------------------------------------------------------------
    async def health(self) -> dict[str, Any]:
        """``GET /health`` on the chat server.

        Returns the parsed JSON body, e.g. ``{"status": "ok"}`` or
        ``{"status": "loading model", "progress": 0.42}``.
        """
        try:
            r = await self._chat.get("/health")
        except httpx.HTTPError as exc:
            raise HermesError(f"health: transport error: {exc}") from exc

        if r.status_code >= 500:
            # 503 with {"error": {"message": "Loading model"}} is the documented
            # response while the model is still mmap'ing; surface it verbatim.
            raise HermesError(f"health: HTTP {r.status_code}: {r.text}")

        try:
            return r.json()
        except json.JSONDecodeError as exc:
            raise HermesError(f"health: non-JSON body: {r.text!r}") from exc

    # ------------------------------------------------------------------
    # completion
    # ------------------------------------------------------------------
    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.4,
        stop: list[str] | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> str | AsyncIterator[str]:
        """``POST /completion`` — llama.cpp's native (non-OpenAI) endpoint.

        ``stream=False`` returns the final string content. ``stream=True``
        returns an async iterator yielding incremental token chunks (parsed
        from the server's SSE ``data: {...}`` frames).
        """
        # Server-side field name is ``n_predict``; ``max_tokens`` is also
        # accepted on recent builds but n_predict is universal.
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if stop is not None:
            payload["stop"] = stop
        payload.update(extra)

        if stream:
            return self._complete_stream(payload)
        return await self._complete_once(payload)

    async def _complete_once(self, payload: dict[str, Any]) -> str:
        try:
            r = await self._chat.post("/completion", json=payload)
            r.raise_for_status()
            body = r.json()
        except httpx.HTTPStatusError as exc:
            raise HermesError(
                f"complete: HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise HermesError(f"complete: transport error: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise HermesError(f"complete: non-JSON body: {exc}") from exc

        # Native /completion shape: {"content": "...", "stop": true, ...}
        if "content" not in body:
            raise HermesError(f"complete: missing 'content' in response: {body!r}")
        return str(body["content"])

    async def _complete_stream(
        self, payload: dict[str, Any]
    ) -> AsyncIterator[str]:
        # Defined as a generator factory so the public method can `return` an
        # AsyncIterator without triggering coroutine vs generator confusion.
        async def _gen() -> AsyncIterator[str]:
            try:
                async with self._chat.stream(
                    "POST", "/completion", json=payload
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            # Skip malformed frames rather than aborting the
                            # entire generation; llama-server occasionally
                            # interleaves keep-alive comments.
                            continue
                        piece = chunk.get("content", "")
                        if piece:
                            yield piece
                        if chunk.get("stop"):
                            return
            except httpx.HTTPStatusError as exc:
                raise HermesError(
                    f"complete[stream]: HTTP {exc.response.status_code}: "
                    f"{exc.response.text}"
                ) from exc
            except httpx.HTTPError as exc:
                raise HermesError(
                    f"complete[stream]: transport error: {exc}"
                ) from exc

        return _gen()

    # ------------------------------------------------------------------
    # chat (OpenAI-compatible)
    # ------------------------------------------------------------------
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.4,
        stop: list[str] | None = None,
        **extra: Any,
    ) -> str:
        """``POST /v1/chat/completions`` — OpenAI-compatible chat.

        Returns the assistant message ``content`` from
        ``choices[0].message.content``. Streaming is intentionally *not*
        exposed on this method; use :meth:`complete` for token-level streams.
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if stop is not None:
            payload["stop"] = stop
        payload.update(extra)

        try:
            r = await self._chat.post("/v1/chat/completions", json=payload)
            r.raise_for_status()
            body = r.json()
        except httpx.HTTPStatusError as exc:
            raise HermesError(
                f"chat: HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise HermesError(f"chat: transport error: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise HermesError(f"chat: non-JSON body: {exc}") from exc

        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise HermesError(f"chat: unexpected response shape: {body!r}") from exc

    # ------------------------------------------------------------------
    # embeddings
    # ------------------------------------------------------------------
    async def embed(self, text: str | list[str]) -> list[list[float]]:
        """``POST {embed_url}/embedding`` — native embedding endpoint.

        Accepts either a single string or a batch list. Returns a list of
        embedding vectors (rows aligned to the input order). For
        ``nomic-embed-text-v1.5`` each row is 768-dim float32.

        NOTE: Caller is responsible for prepending the Nomic v1.5 task
        prefix (``"search_document: "`` for indexing, ``"search_query: "``
        for queries) — that's a model-level contract the indexer layer owns,
        not the transport layer.
        """
        if isinstance(text, str):
            inputs: list[str] = [text]
        else:
            inputs = list(text)

        if not inputs:
            return []

        # llama-server's native /embedding accepts {"content": str | [str]}.
        # Single-string and batch use the same field; the response shape
        # differs (object vs. array), so we normalize below.
        payload: dict[str, Any] = {"content": inputs if len(inputs) > 1 else inputs[0]}

        try:
            r = await self._embed.post("/embedding", json=payload)
            r.raise_for_status()
            body = r.json()
        except httpx.HTTPStatusError as exc:
            raise HermesError(
                f"embed: HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise HermesError(f"embed: transport error: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise HermesError(f"embed: non-JSON body: {exc}") from exc

        return _coerce_embedding_rows(body, expected=len(inputs))

    # ------------------------------------------------------------------
    # slot save / restore (per-slot KV persistence on the chat server)
    # ------------------------------------------------------------------
    async def slot_save(self, slot_id: int, name: str) -> dict[str, Any]:
        """``POST /slots/{slot_id}?action=save`` — persist KV cache to disk.

        Requires the chat server to have been launched with
        ``--slot-save-path <dir>``; the ``name`` becomes the filename stem
        within that directory (server appends its own extension).
        """
        return await self._slot_action(slot_id, action="save", filename=name)

    async def slot_restore(self, slot_id: int, name: str) -> dict[str, Any]:
        """``POST /slots/{slot_id}?action=restore`` — reload a saved KV cache."""
        return await self._slot_action(slot_id, action="restore", filename=name)

    async def slot_erase(self, slot_id: int, name: str) -> dict[str, Any]:
        """``POST /slots/{slot_id}?action=erase`` — delete the saved KV cache file.

        Used by ``/forget-cache`` in the REPL so the user can drop a stale
        cache without touching the slot directory by hand.
        """
        return await self._slot_action(slot_id, action="erase", filename=name)

    async def _slot_action(
        self, slot_id: int, *, action: str, filename: str
    ) -> dict[str, Any]:
        if action not in ("save", "restore", "erase"):
            raise HermesError(f"slot: unknown action {action!r}")
        if slot_id < 0:
            raise HermesError(f"slot: invalid slot_id {slot_id}")

        try:
            r = await self._chat.post(
                f"/slots/{slot_id}",
                params={"action": action},
                json={"filename": filename},
            )
            r.raise_for_status()
            # Erase returns 200 with no body on some builds; tolerate that.
            if not r.content:
                return {"action": action, "slot_id": slot_id, "filename": filename}
            return r.json()
        except httpx.HTTPStatusError as exc:
            raise HermesError(
                f"slot[{action}]: HTTP {exc.response.status_code}: "
                f"{exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise HermesError(f"slot[{action}]: transport error: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise HermesError(f"slot[{action}]: non-JSON body: {exc}") from exc

    # ------------------------------------------------------------------
    # sync convenience wrappers (for CLI usage; do NOT call from async code)
    # ------------------------------------------------------------------
    def health_sync(self) -> dict[str, Any]:
        """Blocking wrapper around :meth:`health` for CLI scripts."""
        return asyncio.run(self._scoped(self.health()))

    def complete_sync(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.4,
        stop: list[str] | None = None,
    ) -> str:
        """Blocking wrapper around :meth:`complete` (non-streaming only)."""
        result = asyncio.run(
            self._scoped(
                self.complete(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    stream=False,
                )
            )
        )
        # complete() with stream=False always returns str; type-narrow for callers.
        assert isinstance(result, str)
        return result

    def chat_sync(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Blocking wrapper around :meth:`chat` for CLI scripts."""
        return asyncio.run(self._scoped(self.chat(messages, **kwargs)))

    def embed_sync(self, text: str | list[str]) -> list[list[float]]:
        """Blocking wrapper around :meth:`embed` for CLI scripts."""
        return asyncio.run(self._scoped(self.embed(text)))

    async def _scoped(self, coro):
        # Each *_sync call runs inside its own event loop via asyncio.run, but
        # the AsyncClient pools were created on whatever loop instantiated this
        # HermesClient. To keep sync usage robust we close+recreate the pools
        # for the duration of the call.
        # Simpler, equivalent approach: just await the coroutine; httpx's
        # AsyncClient binds its connection pool lazily on first request, so
        # routing through a fresh loop works as long as we don't reuse the
        # same client across multiple *_sync calls from different loops.
        try:
            return await coro
        finally:
            # Drop pooled connections so the next *_sync call (potentially on
            # a different event loop) doesn't see stale sockets.
            await self._chat.aclose()
            await self._embed.aclose()
            self._chat = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                limits=self._limits,
            )
            self._embed = httpx.AsyncClient(
                base_url=self.embed_url,
                timeout=self._timeout,
                limits=self._limits,
            )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _coerce_embedding_rows(
    body: Any, *, expected: int
) -> list[list[float]]:
    """Normalize the various shapes llama-server's /embedding can return.

    Observed shapes (all from current upstream; we tolerate each):

    * Single input: ``{"embedding": [..768..]}``
    * Single input (newer):
        ``{"embedding": [[..768..]]}`` (1-row matrix; pooling=mean default)
    * Batch input: ``[{"embedding": [..]}, {"embedding": [..]}, ...]``
    * Batch input (older):
        ``[{"embedding": [[..]]}, ...]`` (each row wrapped in a 1-row matrix)
    """
    rows: list[list[float]] = []

    if isinstance(body, dict) and "embedding" in body:
        rows.append(_unwrap_vector(body["embedding"]))
    elif isinstance(body, list):
        for entry in body:
            if not isinstance(entry, dict) or "embedding" not in entry:
                raise HermesError(
                    f"embed: malformed batch entry: {entry!r}"
                )
            rows.append(_unwrap_vector(entry["embedding"]))
    else:
        raise HermesError(f"embed: unexpected response shape: {body!r}")

    if len(rows) != expected:
        raise HermesError(
            f"embed: expected {expected} vectors, got {len(rows)}"
        )
    return rows


def _unwrap_vector(value: Any) -> list[float]:
    """Accept either ``[float, ...]`` or ``[[float, ...]]`` (1-row matrix)."""
    if not isinstance(value, list):
        raise HermesError(f"embed: vector is not a list: {value!r}")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            # Multi-row pooling=none is not supported by this client because
            # the indexer relies on one vector per input.
            raise HermesError(
                "embed: server returned multi-row vector (pooling=none?); "
                "configure llama-server with --pooling mean"
            )
        value = value[0]
    return [float(x) for x in value]
