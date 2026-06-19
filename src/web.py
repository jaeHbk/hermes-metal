"""Fetch a URL and extract its readable article text.

This module is the one piece the wiki-ingest pipeline was missing for URLs:
turn a web page into clean markdown the LLM can summarize. It knows nothing
about the wiki or the chat server — it only does HTTP + boilerplate removal,
so it is testable with a mocked httpx transport and has no daemon dependency.

We use httpx (already a project dependency) for the fetch and trafilatura for
extraction. trafilatura strips nav/ads/footer chrome and emits the main
article body as markdown, and surfaces title/author/date metadata when the
page provides them.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx
import trafilatura


# A real browser User-Agent: many sites 403 a bare python-httpx client.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Below this, an "extraction" is almost certainly nav/boilerplate residue
# rather than a real article. Skip rather than ingest noise.
_MIN_CONTENT_CHARS = 200


class WebError(Exception):
    """Any failure fetching or extracting a URL (HTTP error, timeout,
    non-HTML, or no extractable article content)."""


@dataclass
class Article:
    url: str
    title: str = ""
    text: str = ""
    author: str = ""
    date: str = ""


def fetch_article(
    url: str,
    *,
    timeout: float = 20.0,
    _client: httpx.Client | None = None,
) -> Article:
    """Fetch ``url`` and return its extracted article as an :class:`Article`.

    Raises :class:`WebError` on HTTP error, timeout, non-HTML response, or when
    no article-like content can be extracted. ``_client`` is for tests only
    (inject an ``httpx.Client`` backed by a MockTransport).
    """
    client = _client or httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": _USER_AGENT},
    )
    try:
        try:
            resp = client.get(url)
        except httpx.TimeoutException as exc:
            # Don't quote `timeout` in the message: on an injected client the
            # effective timeout is the caller's, not this arg.
            raise WebError(f"timeout fetching {url}") from exc
        except httpx.HTTPError as exc:
            raise WebError(f"fetch error for {url}: {exc}") from exc

        if resp.status_code >= 400:
            raise WebError(f"HTTP {resp.status_code} fetching {url}")

        ctype = resp.headers.get("Content-Type", "")
        if ctype and "html" not in ctype.lower():
            raise WebError(f"not HTML ({ctype}) at {url}")

        html = resp.text
    finally:
        if _client is None:
            client.close()

    # with_metadata=False: in trafilatura 2.x, with_metadata=True prepends a
    # YAML frontmatter block (title/date) to the markdown body. We get those
    # fields separately via extract_metadata() below, so keep the body clean.
    extracted = trafilatura.extract(
        html,
        output_format="markdown",
        with_metadata=False,
        include_comments=False,
        include_tables=True,
    )
    if not extracted or len(extracted.strip()) < _MIN_CONTENT_CHARS:
        raise WebError(f"no extractable content at {url}")

    meta = trafilatura.extract_metadata(html)
    title = (getattr(meta, "title", None) or "") if meta else ""
    author = (getattr(meta, "author", None) or "") if meta else ""
    date = (getattr(meta, "date", None) or "") if meta else ""

    return Article(
        url=url,
        title=title.strip(),
        text=extracted.strip(),
        author=author.strip(),
        date=date.strip(),
    )
