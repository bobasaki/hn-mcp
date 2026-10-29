# hn-mcp

An MCP server for Hacker News. Built against the **2026-07-28** protocol
revision and **MCP Python SDK v2**.

Five read-only tools:

| Tool | Does |
|---|---|
| `list_stories` | Browse a feed (top / new / best / ask / show / job) |
| `get_story` | One story, job or poll by id |
| `get_discussion` | A story plus its comment thread, flattened and capped |
| `search_stories` | Full-text and date-filtered search over all HN history |
| `get_user` | A user profile |

## Run it

```bash
uv sync
uv run mcp dev src/hn_mcp/server.py          # Inspector, stdio
uv run mcp run src/hn_mcp/server.py --transport streamable-http   # HTTP
uv run pytest                                 # 34 tests, no network
```

## Architecture

```
server.py     MCP layer      tool schemas, annotations, response models
    |                        no I/O, no parsing
    v
hn_client.py  service layer  Firebase + Algolia, fan-out, BFS, caching
    |
    v
models.py     data layer     Story / Comment / User, normalization
cache.py                     TTL cache
```

Dependencies point one way only. `models.py` does not know MCP exists, which
is why it can be tested without starting a server.

## Design notes

**Two upstreams, one vocabulary.** Feeds, items and users come from HN's
Firebase API; search comes from HN's Algolia index, because Firebase has no
search endpoint at all. They name the same concepts differently (`by` vs
`author`, `score` vs `points`, `descendants` vs `num_comments`), so both
parse into the same models. The model never learns there were two APIs.

**Server-side fan-out.** Firebase returns id lists, so a 30-story feed is 31
requests. Doing that in the server costs one tool call; making the model walk
the ids costs 31 round trips with raw JSON in context each time.

**Comment trees are breadth-first.** HN ranks `kids` by its own ordering, so
breadth-first spends the comment budget on the highest-ranked comments across
the whole thread instead of exhausting it inside one argument. Nesting is
carried by an integer `depth` rather than nested objects, so the list stays
valid wherever it is cut.

**Truncation is reported, never silent.** When caps bite, `truncated` is true
and `truncation_hint` names the argument to raise. Each comment carries
`num_replies`, so the model can tell "no replies" from "replies you were not
shown" — a distinction it cannot otherwise make.

**Errors are prompts.** A failure says what went wrong, which tool to use
instead, and hands back the id needed to recover. Asking `get_story` for a
comment id returns the parent story's id rather than a validation error.

**Constraints live in types.** `Literal` feeds become schema enums;
`Field(ge=, le=)` becomes `minimum`/`maximum`. The docstring explains what
options *mean*; the type system enforces which are *legal*.

**Stateless.** The 2026-07-28 revision removed sessions, so the same code
serves stdio and Streamable HTTP with no session plumbing. Any request can
land on any instance.

## Verified against live HN

`search_stories` and `get_discussion` have been run against the real API.
Doing so found three bugs the fixtures could not:

- Breadth-first traversal with only a total cap spent the entire comment
  budget on top-level comments and returned one reply out of forty. Real
  threads are far wider than they are deep. Fixed with `max_top_level`.
- The truncation hint asserted a rule of thumb instead of measuring, and was
  wrong about which comments had hidden replies. It now counts them.
- Per-comment truncation was computed and discarded, leaving a trailing
  ellipsis indistinguishable from the author's own. Now `text_truncated`.

`tests/fake_hn.py` carries a wide thread modelled on the real one so the
first of those cannot regress.

## Not done yet

- `list_stories`, `get_story` and `get_user` have not been exercised against
  live HN, only against fixtures.
- Never loaded into a host, so tool *selection* is untested: whether a model
  picks `search_stories` over `list_stories` unprompted is the real measure
  of the docstrings, and it has not been measured.
- No rate limiting. Fine for stdio, needed before a public HTTP deployment.
- `search_stories` only searches stories, not comments (`tags` is fixed).
