"""A synthetic HN served through httpx.MockTransport.

The sandbox proxy blocks hacker-news.firebaseio.com, so the test suite runs
against a tree we control. Field names and shapes were captured from the real
API, so parsing is still exercised honestly.
"""

import json
import re
from typing import Any

import httpx


def build_tree() -> dict[int, dict[str, Any]]:
    """Story 1 with 3 top-level comments; each has 2 replies; one is deleted.

        1 (story)
        +- 10, 11, 12          depth 0
           +- 100,101 / 110,111 / 120,121   depth 1
              +- 1000          depth 2
    """
    items: dict[int, dict[str, Any]] = {
        1: {
            "id": 1, "type": "story", "by": "alice", "score": 250,
            "descendants": 10, "time": 1700000000,
            "title": "A story about Rust", "url": "https://example.com/rust",
            "text": "<p>Body with <i>markup</i> &amp; entities.</p>",
            "kids": [10, 11, 12],
        }
    }
    for i, top in enumerate([10, 11, 12]):
        items[top] = {
            "id": top, "type": "comment", "by": f"user{top}",
            "text": f"<p>Top level {i}</p>", "time": 1700000100 + top,
            "kids": [top * 10, top * 10 + 1],
        }
        for leaf in (top * 10, top * 10 + 1):
            items[leaf] = {
                "id": leaf, "type": "comment", "by": f"user{leaf}",
                "text": f"<p>Reply {leaf}</p>", "time": 1700000200 + leaf,
                "kids": [1000] if leaf == 100 else [],
            }
    items[1000] = {
        "id": 1000, "type": "comment", "by": "deep",
        "text": "Deep reply", "time": 1700000999,
    }
    items[110]["deleted"] = True          # must be skipped
    items[111]["dead"] = True             # must be skipped

    # Story 2: a WIDE thread, shaped like a real popular HN post.
    # 60 top-level comments, each with 5 replies. This is the shape that
    # broke the original BFS: an unbounded first level ate the whole
    # comment budget and no reply was ever reached.
    items[2] = {
        "id": 2, "type": "story", "by": "wide", "score": 848,
        "descendants": 360, "time": 1700001000,
        "title": "A very popular story", "url": "https://example.com/wide",
        "kids": [2000 + i for i in range(60)],
    }
    for i in range(60):
        top = 2000 + i
        items[top] = {
            "id": top, "type": "comment", "by": f"top{i}",
            "text": f"Opening statement {i}", "time": 1700001100 + i,
            "kids": [top * 100 + j for j in range(5)],
        }
        for j in range(5):
            leaf = top * 100 + j
            items[leaf] = {
                "id": leaf, "type": "comment", "by": f"reply{leaf}",
                "text": f"Reply {j} to {top}", "time": 1700001200 + leaf,
            }
    return items


class FakeHN:
    def __init__(self) -> None:
        self.items = build_tree()
        self.request_log: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.request_log.append(url)

        m = re.search(r"/v0/item/(\d+)\.json", url)
        if m:
            item = self.items.get(int(m.group(1)))
            # Real HN answers "null" (a JSON body) for a missing id.
            return httpx.Response(200, content=json.dumps(item),
                                  headers={"content-type": "application/json"})

        m = re.search(r"/v0/(\w+stories)\.json", url)
        if m:
            return httpx.Response(200, json=[1, 2, 3])

        m = re.search(r"/v0/user/(\w+)\.json", url)
        if m:
            name = m.group(1)
            if name != "alice":
                return httpx.Response(200, content="null",
                                      headers={"content-type": "application/json"})
            return httpx.Response(200, json={
                "id": "alice", "created": 1600000000, "karma": 4200,
                "about": "<p>Hi &amp; hello</p>", "submitted": [1, 10, 11],
            })

        if "/api/v1/search" in url:
            return httpx.Response(200, json={
                "nbHits": 1,
                "hits": [{
                    "objectID": "8863", "title": "Dropbox", "author": "dhouston",
                    "points": 104, "num_comments": 71,
                    "created_at_i": 1175714200, "url": "https://example.com",
                }],
            })

        return httpx.Response(404, json={"error": "not found"})

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    def item_requests(self) -> int:
        return len([u for u in self.request_log if "/v0/item/" in u])
