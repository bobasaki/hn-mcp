"""Hacker News MCP server.

This module is protocol only: it declares tools, validates their inputs via
type hints, and shapes responses into typed models. All I/O lives in
hn_client.py and all data shaping in models.py, so nothing here fetches,
parses or cleans anything itself.

Five tools, deliberately. Current guidance converges on 5-8 per server:
past that, tool selection accuracy drops because the model is choosing from
a crowd rather than a menu.

    list_stories     browse a feed
    get_story        one story by id
    get_discussion   a story plus its comments, truncated and reported
    search_stories   full-text and date-filtered search
    get_user         a profile

Run locally:   uv run mcp dev src/hn_mcp/server.py
Serve on HTTP: uv run mcp run src/hn_mcp/server.py --transport streamable-http
"""

import time
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from hn_mcp.hn_client import Feed, HNClient
from hn_mcp.models import Comment, Story, User

mcp = MCPServer(
    "hn",
    version="0.1.0",
    instructions=(
        "Read-only access to Hacker News: feeds, stories, comment threads, "
        "search and user profiles. Story ids flow between tools - take an id "
        "from list_stories or search_stories and pass it to get_discussion."
    ),
)

# Every tool here only reads, and every tool talks to a third-party API whose
# contents are outside this server's control. Declaring both lets a host skip
# confirmation prompts it would otherwise raise for each call.
READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True)

# One client for the process: it owns the connection pool, the TTL cache and
# the concurrency bound, so those are shared across every tool call.
client = HNClient()


# --- response models ------------------------------------------------------
# Typed returns are what generate each tool's outputSchema, so the model
# knows the shape of a reply before it spends a call finding out.


class StoryList(BaseModel):
    """A page of stories from a feed."""

    stories: list[Story]
    returned: int = Field(description="Stories in this response.")
    feed_total: int = Field(description="Ids available in the feed before filtering.")


class Discussion(BaseModel):
    """A story with a flattened, capped slice of its comment tree."""

    story: Story
    comments: list[Comment] = Field(
        description="Flat list ordered breadth-first. Use each comment's "
        "`depth` to reconstruct nesting."
    )
    returned: int = Field(description="Comments in this response.")
    total_in_thread: int | None = Field(
        default=None, description="Comments in the whole thread, including omitted."
    )
    top_level_shown: int = Field(
        description="Top-level comments included, bounded by max_top_level."
    )
    top_level_total: int = Field(
        description="Top-level comments the story actually has."
    )
    comments_with_hidden_replies: int = Field(
        description="Returned comments that have replies not included here."
    )
    truncated: bool = Field(
        description="True when comments were omitted by a depth or count cap."
    )
    truncation_hint: str | None = Field(
        default=None,
        description="When truncated, names the argument to raise to see more.",
    )


class SearchResults(BaseModel):
    """Search hits from HN's Algolia index."""

    results: list[Story]
    returned: int = Field(description="Hits in this response.")
    total_hits: int = Field(description="Total matches, which may exceed `returned`.")


# --- tools ----------------------------------------------------------------


@mcp.tool(annotations=READ_ONLY)
async def list_stories(
    feed: Annotated[
        Feed,
        Field(
            description="Which HN feed to read. top is the front page, "
            "new is newest-first, best is multi-day high scorers, and "
            "ask/show/job are the Ask HN, Show HN and jobs sections. "
            "See the tool description for when to choose each."
        ),
    ] = "top",
    limit: Annotated[
        int, Field(ge=1, le=100, description="How many stories to return, 1-100.")
    ] = 20,
    offset: Annotated[
        int, Field(ge=0, description="Stories to skip, for paging through a feed.")
    ] = 0,
    min_score: Annotated[
        int | None,
        Field(ge=0, description="Drop stories scoring below this. Omit for no filter."),
    ] = None,
) -> StoryList:
    """Browse Hacker News story feeds.

    Returns each story's title, score, author, comment count and links -
    enough to answer "what's on HN right now" in a single call.

    Feeds:
      top   front page, ranked by HN's algorithm. Also contains job posts
      new   most recently submitted, unranked
      best  highest-scoring stories of the last few days
      ask   Ask HN text and discussion posts
      show  Show HN project launches
      job   YC portfolio job postings

    top and new hold up to 500 ids; ask, show and job hold up to 200, so a
    large limit on those feeds returns fewer stories than requested.

    Every story carries an `id`. Pass it to get_discussion to read comments -
    this tool returns no comment text.
    """
    stories, total = await client.stories(
        feed, limit=limit, offset=offset, min_score=min_score
    )
    return StoryList(stories=stories, returned=len(stories), feed_total=total)


@mcp.tool(annotations=READ_ONLY)
async def get_story(story_id: int) -> Story:
    """Fetch one Hacker News story, job or poll by id.

    Use this when an id is already known - from list_stories, from
    search_stories, or from a news.ycombinator.com/item?id=N URL. Returns the
    story only; call get_discussion to read its comments.
    """
    raw = await client.raw_item(story_id)
    if not raw:
        raise ValueError(
            f"No Hacker News item with id {story_id}. Ids come from "
            "list_stories or search_stories, or from a "
            "news.ycombinator.com/item?id=N URL."
        )
    # An actionable error beats a Story of null fields: a comment has no
    # title and no score, so it would validate into a useless shell.
    if raw.get("type") == "comment":
        raise ValueError(
            f"Item {story_id} is a comment, not a story. To read comments, "
            f"call get_discussion with the story's id (this comment's parent "
            f"is {raw.get('parent')})."
        )
    return Story.from_firebase(raw)


@mcp.tool(annotations=READ_ONLY)
async def get_discussion(
    story_id: int,
    max_depth: Annotated[
        int,
        Field(
            ge=0,
            le=5,
            description="How deep to descend. 0 is top-level replies only, "
            "2 is a good default for getting the sense of a thread.",
        ),
    ] = 2,
    max_comments: Annotated[
        int, Field(ge=1, le=200, description="Hard cap on total comments returned.")
    ] = 40,
    max_top_level: Annotated[
        int,
        Field(
            ge=1,
            le=100,
            description="How many top-level comments to open. Low values leave "
            "budget for replies and give conversation; high values give a survey "
            "of opinions with few replies.",
        ),
    ] = 10,
    comment_chars: Annotated[
        int, Field(ge=100, le=5000, description="Per-comment character cap.")
    ] = 1200,
) -> Discussion:
    """Read a Hacker News comment thread in one call.

    Returns the story plus a flat list of comments ordered breadth-first:
    all top-level replies, then all of their replies, and so on. HN ranks
    `kids` by its own ordering, so breadth-first spends the comment budget
    on the highest-ranked comments across the whole thread rather than
    exhausting it inside one argument.

    Nesting is carried by each comment's `depth` rather than by nested
    objects, so the list can be cut at any point and stay valid.

    Popular threads are wide - often over a hundred top-level comments - so
    max_top_level bounds how many of them are opened. That is what leaves
    budget for replies: without it the comment cap is spent entirely on
    opening statements and no reply is ever reached. Lower it to read an
    argument in depth, raise it to survey opinions broadly.

    Big threads will not fit. When anything is omitted, `truncated` is true
    and `truncation_hint` names which argument to raise and why.
    `comments_with_hidden_replies` counts how many returned comments have
    replies that were left out, and `top_level_shown` against
    `top_level_total` shows how much of the first level you are seeing.

    Deleted and dead comments are skipped entirely.
    """
    story, comments, meta = await client.discussion(
        story_id,
        max_depth=max_depth,
        max_comments=max_comments,
        max_top_level=max_top_level,
        comment_chars=comment_chars,
    )
    return Discussion(
        story=story,
        comments=comments,
        returned=meta["returned"],
        total_in_thread=meta["total_in_thread"],
        top_level_shown=meta["top_level_shown"],
        top_level_total=meta["top_level_total"],
        comments_with_hidden_replies=meta["comments_with_hidden_replies"],
        truncated=meta["truncated"],
        truncation_hint=meta.get("truncation_hint"),
    )


@mcp.tool(annotations=READ_ONLY)
async def search_stories(
    query: str,
    sort: Annotated[
        Literal["relevance", "date"],
        Field(description="relevance ranks by match quality; date is newest first."),
    ] = "relevance",
    limit: Annotated[int, Field(ge=1, le=100, description="Hits to return.")] = 20,
    page: Annotated[int, Field(ge=0, description="Zero-based page of results.")] = 0,
    days_back: Annotated[
        int | None,
        Field(
            ge=1,
            le=7300,
            description="Only items from the last N days. Omit to search all history.",
        ),
    ] = None,
) -> SearchResults:
    """Full-text search across all of Hacker News history.

    Use this for anything the feeds cannot answer: a topic, a company, a
    phrase, or anything older than the current front page. The feed tools
    only see what is on HN right now; this sees everything ever posted.

    sort="relevance" is right for topic questions. sort="date" is right for
    "what is the latest on X", since the best match may be years old.

    days_back filters by age, so days_back=30 with sort="date" answers
    "what has HN said about X recently".
    """
    since = int(time.time()) - days_back * 86400 if days_back else None
    results, total = await client.search(
        query, limit=limit, page=page, sort=sort, since=since
    )
    return SearchResults(
        results=results, returned=len(results), total_hits=total
    )


@mcp.tool(annotations=READ_ONLY)
async def get_user(username: str) -> User:
    """Look up a Hacker News user profile.

    Returns karma, account age, bio and submission count. Usernames are
    case-sensitive, and only accounts with public activity are exposed by
    HN's API.

    This returns the profile, not the user's posts. To find what someone
    submitted, use search_stories.
    """
    return await client.user(username)


# --- prompts ---------------------------------------------------------------
# Prompts are user-invoked templates, not model-invoked like tools. They
# exist to encode what the server author knows about these tools and that
# the user has no way to know: which arguments suit which question, and
# which fields in the response are worth paying attention to.


@mcp.prompt()
def summarize_thread(story_id: str) -> str:
    """Summarize an HN discussion, organized by disagreement rather than order."""
    return (
        f"Read the Hacker News discussion for story {story_id} using "
        "get_discussion with max_depth=2 and max_comments=60.\n\n"
        "Then write a summary that:\n"
        "1. States the story's central claim in one sentence.\n"
        "2. Identifies the two or three main points of disagreement.\n"
        "3. Gives the strongest argument on each side, naming the commenter.\n"
        "4. Notes any consensus the thread reached, or says there was none.\n\n"
        "Group comments by argument, not by their order in the thread.\n\n"
        "If the response comes back with truncated=true, say so, and say what "
        "the truncation_hint suggests might be missing. A comment whose "
        "num_replies is above zero at the deepest level has replies you did "
        "not see, which may include the strongest rebuttal."
    )


@mcp.prompt()
def whats_new(topic: str, days: str = "30") -> str:
    """Find what HN has said about a topic recently, newest first."""
    return (
        f"Find what Hacker News has said about \"{topic}\" in the last {days} "
        f"days.\n\n"
        f"Use search_stories with query=\"{topic}\", sort=\"date\" and "
        f"days_back={days}. Do not use the feed tools: they only see what is "
        "on HN right now, and they cannot search.\n\n"
        "sort=\"date\" matters here. With sort=\"relevance\" the best match "
        "may be years old, which answers a different question.\n\n"
        "Report what changed: new projects, new arguments, shifts in opinion. "
        "Note each item's score and comment count, since a 400-point story "
        "and a 4-point story are not equally representative. If a thread "
        "looks contentious, read it with get_discussion before characterising "
        "what people think."
    )


@mcp.prompt()
def front_page_brief(limit: str = "30") -> str:
    """A themed digest of the HN front page, rather than a list."""
    return (
        f"Call list_stories with feed=\"top\" and limit={limit}.\n\n"
        "Group the stories into three to five themes you infer from the "
        "titles, and name each theme. Within each, mention the stories worth "
        "attention and say why in a clause - a high score, an unusually high "
        "comment-to-score ratio, or a surprising claim.\n\n"
        "A comment count far above what the score suggests usually means "
        "an argument rather than agreement; those threads are often the "
        "interesting ones.\n\n"
        "Do not recite every story. Finish with one line on what the front "
        "page says about what HN cares about today."
    )


def main() -> None:
    """Entry point. Transport comes from the mcp CLI, defaulting to stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
