"""HN data access over two upstreams.

Firebase v0 serves feeds, items and users but returns *id lists only*, so a
front page costs 1 + N requests. Algolia serves full-text and date-filtered
search, which Firebase cannot do at all.

Everything here is transport-blind and returns normalized models. The MCP
tool layer above adds no I/O; this is the only place that talks to HN.
"""

import asyncio
from typing import Any, Iterable, Literal

import httpx

from hn_mcp.cache import TTLCache
from hn_mcp.models import Comment, Story, User, truncate

FIREBASE = "https://hacker-news.firebaseio.com/v0"
ALGOLIA = "https://hn.algolia.com/api/v1"

Feed = Literal["top", "new", "best", "ask", "show", "job"]

_FEED_PATHS: dict[str, str] = {
    "top": "topstories",
    "new": "newstories",
    "best": "beststories",
    "ask": "askstories",
    "show": "showstories",
    "job": "jobstories",
}

ITEM_TTL = 900.0   # items settle; 15 min is safe and cuts fan-out hard
FEED_TTL = 60.0    # the front page reorders constantly
USER_TTL = 300.0

class HNError(RuntimeError):
    """Upstream failure, surfaced to the agent with an actionable message."""


class HNClient:
    """Async HN client with caching and bounded fan-out.

    The httpx client is injected so tests can drive it with MockTransport and
    so the concurrency limit is owned in one place. `max_concurrency` is the
    politeness knob: HN documents no rate limit, which is not the same thing
    as having none.
    """

    def __init__(
        self,
        http: httpx.AsyncClient | None = None,
        *,
        max_concurrency: int = 10,
        cache: TTLCache | None = None,
    ) -> None:
        self._http = http or httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            headers={"User-Agent": "hn-mcp (+https://github.com/)"},
            follow_redirects=True,
        )
        self._owns_http = http is None
        self._sem = asyncio.Semaphore(max_concurrency)
        self._cache = cache or TTLCache()

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    # ---- raw fetch -------------------------------------------------------

    async def _get_json(self, url: str) -> Any:
        async with self._sem:
            try:
                resp = await self._http.get(url)
                resp.raise_for_status()
                try:
                    return resp.json()
                except ValueError as exc:
                    raise HNError(
                        f"HN returned a non-JSON body for {url}. "
                        "This is usually a transient upstream error; retry once."
                    ) from exc
            except httpx.HTTPStatusError as exc:
                raise HNError(
                    f"HN returned {exc.response.status_code} for {url}. "
                    "The item or user may not exist; verify the id and retry."
                ) from exc
            except httpx.HTTPError as exc:
                raise HNError(
                    f"Could not reach HN ({type(exc).__name__}). "
                    "This is usually transient; retry once before giving up."
                ) from exc

    async def _cached_json(self, key: str, url: str, ttl: float) -> Any:
        hit = await self._cache.get(key)
        if hit is not None:
            return hit
        value = await self._get_json(url)
        if value is not None:
            await self._cache.set(key, value, ttl)
        return value

    # ---- items -----------------------------------------------------------

    async def raw_item(self, item_id: int) -> dict[str, Any] | None:
        return await self._cached_json(
            f"item:{item_id}", f"{FIREBASE}/item/{item_id}.json", ITEM_TTL
        )

    async def raw_items(self, ids: Iterable[int]) -> list[dict[str, Any]]:
        """Hydrate many ids concurrently, preserving input order.

        This is the method that turns Firebase's id-list design into
        something an agent can consume in one tool call. Failures are dropped
        rather than raised: one dead item should not fail a whole feed.
        """
        ids = list(ids)
        if not ids:
            return []
        results = await asyncio.gather(
            *(self.raw_item(i) for i in ids), return_exceptions=True
        )
        return [r for r in results if isinstance(r, dict)]

    # ---- feeds -----------------------------------------------------------

    async def feed_ids(self, feed: Feed) -> list[int]:
        path = _FEED_PATHS.get(feed)
        if path is None:
            raise HNError(
                f"Unknown feed {feed!r}. Valid feeds: {', '.join(_FEED_PATHS)}."
            )
        ids = await self._cached_json(f"feed:{feed}", f"{FIREBASE}/{path}.json", FEED_TTL)
        return list(ids or [])

    async def stories(
        self,
        feed: Feed = "top",
        *,
        limit: int = 20,
        offset: int = 0,
        min_score: int | None = None,
    ) -> tuple[list[Story], int]:
        """Return hydrated stories plus the total available in the feed.

        Over-fetches when min_score is set, because filtering happens after
        hydration and would otherwise under-fill the page.
        """
        ids = await self.feed_ids(feed)
        total = len(ids)
        window = ids[offset : offset + (limit * 3 if min_score else limit)]
        raws = await self.raw_items(window)
        out: list[Story] = []
        for raw in raws:
            story = Story.from_firebase(raw)
            if min_score is not None and (story.score or 0) < min_score:
                continue
            out.append(story)
            if len(out) >= limit:
                break
        return out, total

    # ---- discussion ------------------------------------------------------

    async def discussion(
        self,
        story_id: int,
        *,
        max_depth: int = 2,
        max_comments: int = 40,
        max_top_level: int = 10,
        comment_chars: int = 1200,
    ) -> tuple[Story, list[Comment], dict[str, Any]]:
        """Breadth-first walk of a comment tree, hard-capped.

        BFS rather than DFS because HN orders `kids` by rank, so breadth
        first reaches the better comments before the budget runs out.

        max_top_level is what makes that work on a real thread. HN threads
        are wide - a popular story has well over a hundred top-level
        comments - so an unbounded first level eats the entire comment
        budget on opening statements and never reaches a single reply. On a
        470-comment thread that produced 39 top-level comments and exactly
        one reply: a list of opinions, not a conversation. Bounding breadth
        leaves budget for the replies, which is where threads argue.
        """
        raw_story = await self.raw_item(story_id)
        if not raw_story:
            raise HNError(
                f"No HN item with id {story_id}. Ids come from list_stories or "
                "search_stories; a URL id from news.ycombinator.com also works."
            )
        story = Story.from_firebase(raw_story)

        all_top_level = list(raw_story.get("kids") or [])
        frontier: list[int] = all_top_level[:max_top_level]
        breadth_limited = len(all_top_level) > len(frontier)

        comments: list[Comment] = []
        kids_of: dict[int, list[int]] = {}
        depth = 0
        truncated_by_cap = False
        hidden_by_depth = 0

        while frontier and depth <= max_depth:
            remaining = max_comments - len(comments)
            if remaining <= 0:
                truncated_by_cap = True
                break
            batch_ids = frontier[:remaining]
            if len(frontier) > remaining:
                truncated_by_cap = True
            raws = await self.raw_items(batch_ids)
            next_frontier: list[int] = []
            for raw in raws:
                if raw.get("deleted") or raw.get("dead"):
                    continue
                comment = Comment.from_firebase(raw, depth)
                comment.text, comment.text_truncated = truncate(
                    comment.text, comment_chars
                )
                comments.append(comment)
                kids = list(raw.get("kids") or [])
                kids_of[raw["id"]] = kids
                next_frontier.extend(kids)
            if depth == max_depth:
                hidden_by_depth = len(next_frontier)
            frontier = next_frontier
            depth += 1

        # Count what is actually missing rather than asserting a rule of
        # thumb: a comment has hidden replies when some child id of it never
        # made it into the returned set.
        included = {c.id for c in comments}
        with_hidden = sum(
            1 for kids in kids_of.values() if kids and not set(kids) <= included
        )

        meta: dict[str, Any] = {
            "returned": len(comments),
            "total_in_thread": story.num_comments,
            "top_level_shown": min(len(all_top_level), max_top_level),
            "top_level_total": len(all_top_level),
            "max_depth_reached": max(depth - 1, 0),
            "comments_with_hidden_replies": with_hidden,
            "truncated": truncated_by_cap or breadth_limited or hidden_by_depth > 0,
        }
        if meta["truncated"]:
            hints: list[str] = []
            if breadth_limited:
                hints.append(
                    f"raise max_top_level to see more of the {len(all_top_level)} "
                    "top-level comments"
                )
            if truncated_by_cap:
                hints.append("raise max_comments to see more replies")
            if hidden_by_depth:
                hints.append(
                    f"raise max_depth to reach {hidden_by_depth} deeper replies"
                )
            hint = "; ".join(hints)
            if with_hidden:
                hint += (
                    f". {with_hidden} of the {len(comments)} comments shown have "
                    "replies that were not included"
                )
            meta["truncation_hint"] = hint
        return story, comments, meta

    # ---- search ----------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        tags: str = "story",
        limit: int = 20,
        page: int = 0,
        sort: Literal["relevance", "date"] = "relevance",
        since: int | None = None,
        until: int | None = None,
    ) -> tuple[list[Story], int]:
        """Full-text search via Algolia. Firebase cannot do this at all."""
        endpoint = "search" if sort == "relevance" else "search_by_date"
        params: dict[str, Any] = {
            "query": query,
            "tags": tags,
            "hitsPerPage": max(1, min(limit, 100)),
            "page": page,
        }
        filters = []
        if since:
            filters.append(f"created_at_i>{since}")
        if until:
            filters.append(f"created_at_i<{until}")
        if filters:
            params["numericFilters"] = ",".join(filters)
        url = str(httpx.URL(f"{ALGOLIA}/{endpoint}", params=params))
        payload = await self._get_json(url)
        hits = payload.get("hits", [])
        stories = []
        for hit in hits:
            story = Story.from_algolia(hit)
            stories.append(story)
        return stories, payload.get("nbHits", len(stories))

    # ---- users -----------------------------------------------------------

    async def user(self, username: str) -> User:
        raw = await self._cached_json(
            f"user:{username}", f"{FIREBASE}/user/{username}.json", USER_TTL
        )
        if not raw:
            raise HNError(
                f"No HN user named {username!r}. Usernames are case-sensitive "
                "and only users with public activity are exposed by the API."
            )
        user = User.from_firebase(raw)
        return user

    def cache_stats(self) -> dict[str, int]:
        return self._cache.stats()
