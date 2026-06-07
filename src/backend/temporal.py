"""Temporal query parsing for hermes-metal (Phase B).

Turns a small, *high-precision* set of date phrases in a natural-language
query into a ``[start, end)`` window of POSIX mtimes, which the retrieval
layer applies as a LanceDB ``mtime >= X AND mtime < Y`` filter. The point is
to make "what did I write **yesterday**?" actually scope to yesterday's
files instead of relying on the model to notice the date.

Design constraint (the load-bearing one, called out in IMPROVEMENTS.md):
**a misparse that silently scopes retrieval to the wrong window is worse
than no scoping at all** — the user gets a confident answer drawn from the
wrong notes. So this parser is deliberately conservative:

* It matches only unambiguous phrases ("yesterday", "today", "last week",
  "past N days", "this month", ISO dates, ``YYYY-MM``). Anything fuzzy
  ("recently", "a while ago") is intentionally NOT matched — we fall through
  to an unfiltered vector search.
* It returns ``None`` when nothing matches, and the caller treats ``None`` as
  "no temporal scoping."
* It also reports the matched phrase so the REPL/CLI can *show* the user
  "(scoped to 2026-06-06)" — visible scoping beats silent scoping.

Two filtering modes, because mtime and content-dated notes differ:

* **mtime window** — for "what did I edit", filter on the file's mtime.
* **path-date hint** — many vaults name daily notes ``journal/2026-06-06.md``.
  For an explicit single calendar day we also emit a ``source_path LIKE
  '%YYYY-MM-DD%'`` clause OR'd with the mtime window, so a daily note written
  on a *later* day but *about* that date still matches. The schema's
  ``.hermes-agents.md`` documents the daily-note path scheme.

Pure stdlib (``datetime``, ``re``). ``now`` is injected for deterministic
tests.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Optional


@dataclass(frozen=True)
class DateWindow:
    """A half-open ``[start, end)`` window plus how it was derived.

    ``start`` / ``end`` are POSIX timestamps (seconds, local-time-derived).
    ``phrase`` is the matched text (for user-visible "(scoped to …)").
    ``iso_day`` is set only when the window is exactly one calendar day, so
    the caller can add a ``source_path LIKE '%iso_day%'`` clause for vaults
    that date daily notes in the filename.
    """

    start: float
    end: float
    phrase: str
    iso_day: Optional[str] = None

    def mtime_clause(self) -> str:
        return f"(mtime >= {self.start:.0f} AND mtime < {self.end:.0f})"

    def where_clause(self) -> str:
        """The full LanceDB filter for this window.

        For a single day we OR the mtime window with a path-date LIKE so
        daily notes named after the date match even if their mtime drifted.
        The LIKE value is a fixed ``YYYY-MM-DD`` we control, so there's no
        injection surface.
        """
        clause = self.mtime_clause()
        if self.iso_day:
            clause = f"({clause} OR source_path LIKE '%{self.iso_day}%')"
        return clause


# A bare ``\b`` word boundary does NOT stop a match adjacent to a hyphen (a
# hyphen is a non-word char, so ``\b`` happily matches between it and a digit).
# That let dates embedded in hyphenated identifiers / filenames — e.g.
# ``report-2026-06-07.md`` or ``restore archive-2026-06-backup`` — get parsed
# as temporal phrases and mis-scope retrieval. The ``(?<![-\d]) ... (?![-\d])``
# guards require the date to NOT be flanked by a hyphen or another digit, so it
# only matches a standalone date token in the query text.
_ISO_DATE_RE = re.compile(r"(?<![-\d])(\d{4})-(\d{2})-(\d{2})(?![-\d])")
_ISO_MONTH_RE = re.compile(r"(?<![-\d])(\d{4})-(\d{2})(?![-\d])")
_PAST_N_RE = re.compile(r"\b(?:past|last)\s+(\d{1,3})\s+days?\b", re.IGNORECASE)
_LAST_N_WEEKS_RE = re.compile(r"\b(?:past|last)\s+(\d{1,2})\s+weeks?\b", re.IGNORECASE)


def _day_bounds(d: datetime) -> tuple[float, float]:
    """[midnight, next midnight) for the calendar day of ``d`` (local tz)."""
    start = datetime.combine(d.date(), time.min, tzinfo=d.tzinfo)
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp()


def parse_window(query: str, *, now: Optional[datetime] = None) -> Optional[DateWindow]:
    """Return a :class:`DateWindow` if ``query`` contains a high-precision
    temporal phrase, else ``None``.

    ``now`` defaults to the local-time current moment; tests pass a fixed
    aware datetime. Order matters: more specific patterns are checked before
    broad ones (explicit ISO date before "today", "past 7 days" before
    "this week").
    """
    if not query:
        return None
    if now is None:
        now = datetime.now().astimezone()
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    q = query.lower()

    # --- explicit ISO date: 2026-06-06 (single day, path-hint eligible)
    m = _ISO_DATE_RE.search(query)
    if m:
        y, mo, da = (int(g) for g in m.groups())
        try:
            day = datetime(y, mo, da, tzinfo=now.tzinfo)
        except ValueError:
            day = None
        if day is not None:
            start, end = _day_bounds(day)
            return DateWindow(start, end, m.group(0), iso_day=m.group(0))

    # --- "yesterday" (single day, path-hint eligible)
    if re.search(r"\byesterday\b", q):
        y = now - timedelta(days=1)
        start, end = _day_bounds(y)
        return DateWindow(start, end, "yesterday", iso_day=y.strftime("%Y-%m-%d"))

    # --- "today" (single day, path-hint eligible)
    if re.search(r"\btoday\b", q):
        start, end = _day_bounds(now)
        return DateWindow(start, end, "today", iso_day=now.strftime("%Y-%m-%d"))

    # --- "past/last N days" (rolling window, no single-day path hint)
    m = _PAST_N_RE.search(query)
    if m:
        n = max(1, int(m.group(1)))
        end = now.timestamp()
        start = (now - timedelta(days=n)).timestamp()
        return DateWindow(start, end, m.group(0))

    # --- "past/last N weeks"
    m = _LAST_N_WEEKS_RE.search(query)
    if m:
        n = max(1, int(m.group(1)))
        end = now.timestamp()
        start = (now - timedelta(weeks=n)).timestamp()
        return DateWindow(start, end, m.group(0))

    # --- "last week" / "this week" (Mon-anchored ISO weeks)
    if re.search(r"\blast week\b", q):
        monday_this = (now - timedelta(days=now.weekday()))
        start_dt = datetime.combine((monday_this - timedelta(days=7)).date(), time.min, tzinfo=now.tzinfo)
        end_dt = datetime.combine(monday_this.date(), time.min, tzinfo=now.tzinfo)
        return DateWindow(start_dt.timestamp(), end_dt.timestamp(), "last week")
    if re.search(r"\bthis week\b", q):
        monday_this = (now - timedelta(days=now.weekday()))
        start_dt = datetime.combine(monday_this.date(), time.min, tzinfo=now.tzinfo)
        end_dt = start_dt + timedelta(days=7)
        return DateWindow(start_dt.timestamp(), end_dt.timestamp(), "this week")

    # --- "last month" / "this month"
    if re.search(r"\blast month\b", q):
        first_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_month_end = first_this
        prev = first_this - timedelta(days=1)
        first_prev = prev.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return DateWindow(first_prev.timestamp(), last_month_end.timestamp(), "last month")
    if re.search(r"\bthis month\b", q):
        first_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        days_in_month = calendar.monthrange(now.year, now.month)[1]
        end_dt = first_this + timedelta(days=days_in_month)
        return DateWindow(first_this.timestamp(), end_dt.timestamp(), "this month")

    # --- "YYYY-MM" whole month (checked last; ISO date above is more specific)
    m = _ISO_MONTH_RE.search(query)
    if m and not _ISO_DATE_RE.search(query):
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            first = datetime(y, mo, 1, tzinfo=now.tzinfo)
            days_in_month = calendar.monthrange(y, mo)[1]
            end_dt = first + timedelta(days=days_in_month)
            return DateWindow(first.timestamp(), end_dt.timestamp(), m.group(0))

    return None
