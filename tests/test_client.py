import pytest

from hn_mcp.hn_client import HNClient, HNError
from hn_mcp.models import Story, html_to_text, truncate

from .fake_hn import FakeHN

@pytest.fixture
def fake() -> FakeHN:
    return FakeHN()


@pytest.fixture
def client(fake: FakeHN) -> HNClient:
    return HNClient(http=fake.client())


# ---- normalization ------------------------------------------------------

def test_both_upstreams_produce_the_same_shape() -> None:
    """The agent must never see Firebase vs Algolia field names."""
    fb = Story.from_firebase({
        "id": 8863, "title": "Dropbox", "by": "dhouston", "score": 104,
        "descendants": 71, "time": 1175714200, "url": "https://x", "type": "story",
    })
    algo = Story.from_algolia({
        "objectID": "8863", "title": "Dropbox", "author": "dhouston", "points": 104,
        "num_comments": 71, "created_at_i": 1175714200, "url": "https://x",
    })
    assert fb.model_dump() == algo.model_dump()


def test_algolia_string_id_becomes_int() -> None:
    assert Story.from_algolia({"objectID": "8863", "created_at_i": 1}).id == 8863


# ---- html handling ------------------------------------------------------

def test_html_to_text_strips_markup_and_entities() -> None:
    assert html_to_text("<p>a <b>b</b> &amp; c</p>") == "a b & c"


def test_html_to_text_handles_none_and_empty() -> None:
    assert html_to_text(None) is None
    assert html_to_text("<p></p>") is None


def test_truncate_flags_when_it_cuts() -> None:
    text, cut = truncate("x" * 100, 10)
    assert cut is True and text.endswith("...") and len(text) <= 13
    assert truncate("short", 100) == ("short", False)


# ---- discussion traversal ----------------------------------------------

async def test_bfs_returns_breadth_before_depth(client: HNClient) -> None:
    """Top-level comments must all appear before any reply."""
    _, comments, _ = await client.discussion(1, max_depth=2, max_comments=50)
    depths = [c.depth for c in comments]
    assert depths == sorted(depths), "BFS must not interleave depths"
    assert depths[:3] == [0, 0, 0]


async def test_max_comments_cap_is_respected_and_flagged(client: HNClient) -> None:
    _, comments, meta = await client.discussion(1, max_depth=3, max_comments=2)
    assert len(comments) == 2
    assert meta["truncated"] is True
    assert "max_comments" in meta["truncation_hint"]


async def test_max_depth_stops_descent_and_reports_hidden(client: HNClient) -> None:
    _, comments, meta = await client.discussion(1, max_depth=0, max_comments=50)
    assert {c.depth for c in comments} == {0}
    assert meta["truncated"] is True
    assert "max_depth" in meta["truncation_hint"]


async def test_deleted_and_dead_comments_are_skipped(client: HNClient) -> None:
    _, comments, _ = await client.discussion(1, max_depth=2, max_comments=50)
    ids = {c.id for c in comments}
    assert 110 not in ids, "deleted comment leaked"
    assert 111 not in ids, "dead comment leaked"


async def test_num_replies_signals_hidden_children(client: HNClient) -> None:
    _, comments, _ = await client.discussion(1, max_depth=0, max_comments=50)
    assert all(c.num_replies == 2 for c in comments)


async def test_comment_text_is_truncated_per_comment(client: HNClient) -> None:
    _, comments, _ = await client.discussion(1, max_depth=0, comment_chars=5)
    assert all(len(c.text) <= 8 for c in comments if c.text)


async def test_missing_story_raises_actionable_error(client: HNClient) -> None:
    with pytest.raises(HNError) as exc:
        await client.discussion(999999)
    assert "list_stories" in str(exc.value), "error must tell the agent what to do"


# ---- caching / fan-out --------------------------------------------------

async def test_item_cache_prevents_refetch(fake: FakeHN) -> None:
    client = HNClient(http=fake.client())
    await client.discussion(1, max_depth=1, max_comments=50)
    first = fake.item_requests()
    await client.discussion(1, max_depth=1, max_comments=50)
    assert fake.item_requests() == first, "second call must be served from cache"


async def test_feed_hydration_is_one_call_per_story(client: HNClient, fake: FakeHN) -> None:
    stories, total = await client.stories("top", limit=3)
    assert total == 3
    assert fake.item_requests() == 3, "no redundant fetches during fan-out"


async def test_min_score_filters_after_hydration(client: HNClient) -> None:
    stories, _ = await client.stories("top", limit=3, min_score=1000)
    assert stories == []


async def test_unknown_feed_names_valid_options(client: HNClient) -> None:
    with pytest.raises(HNError) as exc:
        await client.feed_ids("frontpage")  # type: ignore[arg-type]
    assert "top" in str(exc.value)


# ---- search / users -----------------------------------------------------

async def test_search_normalizes_algolia_hits(client: HNClient) -> None:
    stories, total = await client.search("dropbox")
    assert total == 1
    assert stories[0].id == 8863 and stories[0].author == "dhouston"
    assert stories[0].hn_url.endswith("id=8863")


async def test_user_lookup_strips_html(client: HNClient) -> None:
    user = await client.user("alice")
    assert user.karma == 4200 and user.about == "Hi & hello"
    assert user.submission_count == 3


async def test_unknown_user_raises_actionable_error(client: HNClient) -> None:
    with pytest.raises(HNError) as exc:
        await client.user("nobody")
    assert "case-sensitive" in str(exc.value)


# ---- wide threads (the real-world shape) --------------------------------


async def test_wide_thread_still_reaches_replies(client: HNClient) -> None:
    """The bug this guards: on a thread with 60 top-level comments, an
    unbounded first level spends the whole budget on opening statements and
    returns zero replies. Observed on a real 470-comment thread: 39 at
    depth 0, 1 at depth 1."""
    _, comments, _ = await client.discussion(2, max_depth=2, max_comments=40)
    by_depth = {}
    for c in comments:
        by_depth[c.depth] = by_depth.get(c.depth, 0) + 1
    assert by_depth.get(0, 0) <= 10, "breadth must be bounded"
    assert by_depth.get(1, 0) >= 20, f"expected real replies, got {by_depth}"


async def test_max_top_level_is_honoured(client: HNClient) -> None:
    _, comments, meta = await client.discussion(
        2, max_top_level=3, max_comments=100, max_depth=1
    )
    assert sum(1 for c in comments if c.depth == 0) == 3
    assert meta["top_level_shown"] == 3
    assert meta["top_level_total"] == 60


async def test_hint_reports_breadth_limiting(client: HNClient) -> None:
    _, _, meta = await client.discussion(2, max_top_level=5, max_comments=100)
    assert meta["truncated"] is True
    assert "max_top_level" in meta["truncation_hint"]
    assert "60" in meta["truncation_hint"], "should name the real total"


async def test_hidden_reply_count_is_measured_not_asserted(client: HNClient) -> None:
    """The old hint claimed only the deepest level had hidden children.
    On the real thread 24 of 40 comments did, across both levels."""
    _, comments, meta = await client.discussion(2, max_top_level=10, max_comments=12)
    assert meta["comments_with_hidden_replies"] > 0
    assert str(meta["comments_with_hidden_replies"]) in meta["truncation_hint"]


async def test_per_comment_truncation_is_flagged(client: HNClient) -> None:
    """A trailing ellipsis is ambiguous; the flag is not."""
    _, comments, _ = await client.discussion(2, comment_chars=100)
    assert all(c.text_truncated is False for c in comments), "short texts uncut"

    _, comments, _ = await client.discussion(1, max_depth=0, comment_chars=5)
    assert any(c.text_truncated for c in comments), "long texts must be flagged"
