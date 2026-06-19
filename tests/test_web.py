"""Tests for src/web.py — URL fetch + article extraction.

Uses httpx.MockTransport so the suite needs no network (same pattern as
tests/test_streaming.py). trafilatura runs for real against canned HTML —
it is deterministic on fixed input.
"""
from __future__ import annotations

import httpx
import pytest

from src import web


_ARTICLE_HTML = """
<!DOCTYPE html>
<html><head><title>Attention Explained</title>
<meta name="author" content="Jane Roe">
<meta property="article:published_time" content="2024-11-02"></head>
<body>
<nav>home about contact</nav>
<article>
<h1>Attention Explained</h1>
<p>The attention mechanism lets a model weigh the relevance of each input
token when producing each output token. This paragraph is intentionally long
enough that trafilatura treats it as the main article body rather than
boilerplate, which requires a few sentences of real prose to trip its
content-density heuristic reliably across versions.</p>
<p>A second substantial paragraph reinforces that the article body is the
dominant text block on the page, well clear of the 200-character floor that
fetch_article enforces before it accepts an extraction as real content.</p>
</article>
<footer>copyright 2024</footer>
</body></html>
"""


def _client(responder) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(responder))


def test_fetch_article_extracts_body_and_title():
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=_ARTICLE_HTML)

    art = web.fetch_article("https://blog.example/attn", _client=_client(responder))
    assert art.url == "https://blog.example/attn"
    assert art.title == "Attention Explained"
    assert "attention mechanism" in art.text.lower()
    # Boilerplate stripped.
    assert "home about contact" not in art.text.lower()
    assert "copyright" not in art.text.lower()
    # Metadata extracted from the page head.
    assert art.author == "Jane Roe"
    assert art.date == "2024-11-02"


def test_fetch_article_http_error_raises_weberror():
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    with pytest.raises(web.WebError) as exc:
        web.fetch_article("https://paywall.example/x", _client=_client(responder))
    assert "403" in str(exc.value)


def test_fetch_article_empty_content_raises_weberror():
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html="<html><body><nav>menu</nav></body></html>")

    with pytest.raises(web.WebError) as exc:
        web.fetch_article("https://empty.example/x", _client=_client(responder))
    assert "no extractable content" in str(exc.value).lower()


def test_fetch_article_timeout_raises_weberror():
    def responder(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    with pytest.raises(web.WebError) as exc:
        web.fetch_article("https://slow.example/x", _client=_client(responder))
    assert "timeout" in str(exc.value).lower()


def test_fetch_article_non_html_raises_weberror():
    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"%PDF-1.4 binary garbage",
                              headers={"Content-Type": "application/pdf"})

    with pytest.raises(web.WebError):
        web.fetch_article("https://files.example/x.pdf", _client=_client(responder))


def test_fetch_article_titleless_keeps_empty_title():
    body = ("<html><body><article><p>" + ("Plain prose without any heading. " * 20)
            + "</p></article></body></html>")

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=body)

    art = web.fetch_article("https://notitle.example/page", _client=_client(responder))
    assert art.title == ""
    assert "plain prose" in art.text.lower()


def test_fetch_article_follows_redirect():
    """follow_redirects must be honored: a 301 to the canonical URL is
    followed and the final page is extracted (many article URLs redirect)."""
    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"Location": "https://blog.example/new"})
        return httpx.Response(200, html=_ARTICLE_HTML)

    client = httpx.Client(
        transport=httpx.MockTransport(responder), follow_redirects=True
    )
    art = web.fetch_article("https://blog.example/old", _client=client)
    assert art.title == "Attention Explained"
