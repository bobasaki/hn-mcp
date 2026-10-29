"""Normalized HN domain models.

HN exposes two upstreams that name the same concepts differently:

    concept        Firebase v0      Algolia v1
    -----------    -------------    ----------------
    id             id               objectID (a string)
    author         by               author
    score          score            points
    comment count  descendants      num_comments
    timestamp      time             created_at_i

Tools must never leak that split to the agent, so both upstreams parse into
the shapes below. One vocabulary, whatever the source.

Text normalization happens inside the classmethods rather than in the
caller. Constructing a model always cleans its text, so no call site can
forget to and leak raw HTML into the model's context.
"""

import html
import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

HN_ITEM_URL = "https://news.ycombinator.com/item?id={}"
HN_USER_URL = "https://news.ycombinator.com/user?id={}"

# All four compiled at module level, in the order they are applied.
_P_RE = re.compile(r"<p>", re.IGNORECASE)        # paragraph -> blank line
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)  # line break -> newline
_TAG_RE = re.compile(r"<[^>]+>")                  # every other tag -> gone
_WS_RE = re.compile(r"[ \t]+")                    # runs of spaces -> one


def html_to_text(raw: str | None) -> str | None:
    """HN item text is HTML. The model pays tokens for markup it cannot use.

    Order matters. <p> and <br> are converted before the generic tag strip,
    or the structure would be deleted as ordinary tags. html.unescape runs
    last, after stripping: unescaping first would turn a quoted "&lt;b&gt;"
    into "<b>" and the tag strip would then delete the very thing the
    comment was talking about.
    """
    if not raw:
        return None
    text = _P_RE.sub("\n\n", raw)
    text = _BR_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = _WS_RE.sub(" ", html.unescape(text))
    return text.strip() or None


def truncate(text: str | None, limit: int) -> tuple[str | None, bool]:
    """Cap one field. Returns (text, was_truncated) so callers can report it."""
    if text is None or limit <= 0 or len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + "...", True


def to_iso(unix_ts: int | None) -> str | None:
    """Unix int -> ISO 8601 UTC.

    tz=timezone.utc is load-bearing: without it Python uses the machine's
    local timezone and the server reports different times depending on
    where it runs.
    """
    if unix_ts is None:
        return None
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()


class Story(BaseModel):
    """A story, job or poll - the unit an agent reasons about.

    Deliberately omits HN's `kids`: a list of comment ids the model cannot
    read is pure token cost. `num_comments` says how many exist, and
    get_discussion is how they are read.
    """

    id: int = Field(description="HN item id. Pass to get_discussion for comments.")
    title: str | None = Field(default=None, description="Story headline.")
    url: str | None = Field(
        default=None, description="Outbound link. Null for Ask HN and text-only posts."
    )
    author: str | None = Field(default=None, description="Submitter's username.")
    score: int | None = Field(default=None, description="Upvote count.")
    num_comments: int | None = Field(
        default=None, description="Total comments in the thread, including nested."
    )
    created_at: str | None = Field(
        default=None, description="Submission time, ISO 8601 UTC."
    )
    text: str | None = Field(
        default=None,
        description="Body text for Ask HN / Show HN posts, HTML stripped.",
    )
    type: str | None = Field(default=None, description="story, job or poll.")
    hn_url: str = Field(description="Permalink to the HN discussion page.")

    @classmethod
    def from_firebase(cls, raw: dict[str, Any]) -> "Story":
        return cls(
            id=raw["id"],
            title=raw.get("title"),
            url=raw.get("url"),
            author=raw.get("by"),
            score=raw.get("score"),
            num_comments=raw.get("descendants"),
            created_at=to_iso(raw.get("time")),
            text=html_to_text(raw.get("text")),
            type=raw.get("type"),
            hn_url=HN_ITEM_URL.format(raw["id"]),
        )

    @classmethod
    def from_algolia(cls, hit: dict[str, Any]) -> "Story":
        # objectID is a string here; every other surface uses int ids.
        item_id = int(hit["objectID"])
        return cls(
            id=item_id,
            title=hit.get("title"),
            url=hit.get("url"),
            author=hit.get("author"),
            score=hit.get("points"),
            num_comments=hit.get("num_comments"),
            created_at=to_iso(hit.get("created_at_i")),
            text=html_to_text(hit.get("story_text") or hit.get("comment_text")),
            type="story",
            hn_url=HN_ITEM_URL.format(item_id),
        )


class Comment(BaseModel):
    """One comment, flattened out of the tree.

    `depth` replaces nesting: the agent reconstructs structure from an
    integer instead of walking a recursive object, which keeps the response
    a flat list that can be truncated at any point.
    """

    id: int
    author: str | None = Field(default=None, description="Comment author.")
    text: str | None = Field(default=None, description="Comment body, HTML stripped.")
    text_truncated: bool = Field(
        default=False,
        description="True when this comment's text was cut by comment_chars. "
        "Distinguishes our cut from an ellipsis the author wrote.",
    )
    created_at: str | None = Field(default=None, description="ISO 8601 UTC.")
    depth: int = Field(
        description="0 for top-level replies, 1 for replies to those, and so on."
    )
    num_replies: int = Field(
        default=0,
        description="Direct replies. Non-zero at the deepest returned level means "
        "replies exist that were not included.",
    )
    hn_url: str

    @classmethod
    def from_firebase(cls, raw: dict[str, Any], depth: int) -> "Comment":
        return cls(
            id=raw["id"],
            author=raw.get("by"),
            text=html_to_text(raw.get("text")),
            created_at=to_iso(raw.get("time")),
            depth=depth,
            num_replies=len(raw.get("kids") or []),
            hn_url=HN_ITEM_URL.format(raw["id"]),
        )


class User(BaseModel):
    """A Hacker News account."""

    id: str = Field(description="Username. Case-sensitive.")
    created_at: str | None = Field(
        default=None, description="Account creation time, ISO 8601 UTC."
    )
    karma: int | None = Field(default=None, description="Accumulated karma.")
    about: str | None = Field(default=None, description="Profile bio, HTML stripped.")
    submission_count: int | None = Field(
        default=None, description="Items in the user's public history."
    )
    hn_url: str = Field(description="Permalink to the user's HN profile.")

    @classmethod
    def from_firebase(cls, raw: dict[str, Any]) -> "User":
        return cls(
            id=raw["id"],
            created_at=to_iso(raw.get("created")),
            karma=raw.get("karma"),
            about=html_to_text(raw.get("about")),
            submission_count=len(raw.get("submitted") or []),
            hn_url=HN_USER_URL.format(raw["id"]),
        )
