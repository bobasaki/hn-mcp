"""Contract tests for the MCP surface.

These assert the *schema* the model sees, not the data. They exist because
every one of these properties is a deliberate design decision that a
refactor could silently undo: an enum collapsing to a bare string, a bound
disappearing, an outputSchema vanishing when someone changes a return type
to dict for convenience.

Note the snake_case attribute names. SDK v2 renamed every model field from
camelCase in Python, so `tool.inputSchema` is now `tool.input_schema`. The
JSON on the wire is unchanged - this only shows up when introspecting.
"""

import pytest

from hn_mcp.server import mcp

EXPECTED_TOOLS = {
    "list_stories",
    "get_story",
    "get_discussion",
    "search_stories",
    "get_user",
}


@pytest.fixture
async def tools():
    return {t.name: t for t in await mcp.list_tools()}


async def test_exactly_the_intended_tools_are_exposed(tools) -> None:
    """Guidance is 5-8 tools; a server that grows past that silently is a bug."""
    assert set(tools) == EXPECTED_TOOLS


async def test_every_tool_is_marked_read_only(tools) -> None:
    for name, tool in tools.items():
        assert tool.annotations is not None, f"{name} has no annotations"
        assert tool.annotations.read_only_hint is True, f"{name} not read-only"
        assert tool.annotations.open_world_hint is True, f"{name} not open-world"


async def test_every_tool_has_an_output_schema(tools) -> None:
    """A tool returning a bare dict teaches the model nothing about the reply."""
    for name, tool in tools.items():
        assert tool.output_schema is not None, f"{name} has no outputSchema"


async def test_every_tool_has_a_substantive_description(tools) -> None:
    """The description is the prompt. A one-liner is not enough to choose on."""
    for name, tool in tools.items():
        assert tool.description, f"{name} has no description"
        assert len(tool.description) > 120, f"{name} description is too thin"


async def test_feed_is_an_enum_not_a_free_string(tools) -> None:
    """Literal[...] must reach the schema, or invalid feeds get through."""
    feed = tools["list_stories"].input_schema["properties"]["feed"]
    assert set(feed["enum"]) == {"top", "new", "best", "ask", "show", "job"}


async def test_sort_is_an_enum(tools) -> None:
    sort = tools["search_stories"].input_schema["properties"]["sort"]
    assert set(sort["enum"]) == {"relevance", "date"}


@pytest.mark.parametrize(
    "tool_name,param,minimum,maximum",
    [
        ("list_stories", "limit", 1, 100),
        ("get_discussion", "max_depth", 0, 5),
        ("get_discussion", "max_comments", 1, 200),
        ("get_discussion", "comment_chars", 100, 5000),
        ("search_stories", "limit", 1, 100),
    ],
)
async def test_numeric_params_are_bounded(
    tools, tool_name, param, minimum, maximum
) -> None:
    """Unbounded numbers invite a request for 10,000 comments."""
    schema = tools[tool_name].input_schema["properties"][param]
    assert schema["minimum"] == minimum
    assert schema["maximum"] == maximum


async def test_only_genuinely_required_params_are_required(tools) -> None:
    """Everything with a sensible default should be optional, so the model
    can call a tool with no arguments and still get something useful."""
    required = {
        name: set(t.input_schema.get("required", [])) for name, t in tools.items()
    }
    assert required["list_stories"] == set()
    assert required["get_story"] == {"story_id"}
    assert required["get_discussion"] == {"story_id"}
    assert required["search_stories"] == {"query"}
    assert required["get_user"] == {"username"}


async def test_documented_params_carry_descriptions(tools) -> None:
    """A param the model cannot interpret gets guessed at."""
    undocumented = [
        f"{name}.{param}"
        for name, tool in tools.items()
        for param, schema in tool.input_schema["properties"].items()
        if not schema.get("description") and "anyOf" not in schema
        # single-param id/name tools are self-describing via the description
        and param not in {"story_id", "username", "query"}
    ]
    assert not undocumented, f"missing descriptions: {undocumented}"


async def test_story_output_omits_kids(tools) -> None:
    """`kids` is a list of ids the model cannot read - it must not leak."""
    schema = tools["get_story"].output_schema
    assert "kids" not in schema["properties"]
    assert "hn_url" in schema["properties"], "permalink must be present"


async def test_discussion_output_exposes_truncation(tools) -> None:
    """The model must be able to tell a short thread from a cut-off one,
    and to see how much of the first level it is looking at."""
    props = tools["get_discussion"].output_schema["properties"]
    for field in (
        "truncated",
        "truncation_hint",
        "total_in_thread",
        "returned",
        "top_level_shown",
        "top_level_total",
        "comments_with_hidden_replies",
    ):
        assert field in props, f"discussion output missing {field}"


async def test_discussion_exposes_the_breadth_control(tools) -> None:
    """max_top_level is what keeps a wide thread from returning only
    opening statements; it must be reachable and bounded."""
    schema = tools["get_discussion"].input_schema["properties"]["max_top_level"]
    assert schema["minimum"] == 1
    assert schema["maximum"] == 100
    assert schema["description"]


# --- prompts --------------------------------------------------------------


@pytest.fixture
async def prompts():
    return {p.name: p for p in await mcp.list_prompts()}


async def test_expected_prompts_are_registered(prompts) -> None:
    assert set(prompts) == {"summarize_thread", "whats_new", "front_page_brief"}


async def test_prompts_have_descriptions(prompts) -> None:
    for name, p in prompts.items():
        assert p.description, f"{name} has no description"


async def test_only_the_unguessable_args_are_required(prompts) -> None:
    """Anything with a sensible default should not block invocation."""
    req = {
        name: {a.name for a in (p.arguments or []) if a.required}
        for name, p in prompts.items()
    }
    assert req["summarize_thread"] == {"story_id"}
    assert req["whats_new"] == {"topic"}
    assert req["front_page_brief"] == set()


async def test_prompt_renders_with_arguments_substituted() -> None:
    result = await mcp.get_prompt("whats_new", {"topic": "rust", "days": "7"})
    text = result.messages[0].content.text
    assert "rust" in text
    assert "days_back=7" in text


async def test_prompts_name_the_tools_they_expect() -> None:
    """A prompt that does not name a tool leaves the model guessing."""
    expected = {
        "summarize_thread": ("get_discussion", {"story_id": "1"}),
        "whats_new": ("search_stories", {"topic": "x"}),
        "front_page_brief": ("list_stories", {}),
    }
    for prompt_name, (tool_name, args) in expected.items():
        result = await mcp.get_prompt(prompt_name, args)
        text = result.messages[0].content.text
        assert tool_name in text, f"{prompt_name} never mentions {tool_name}"
