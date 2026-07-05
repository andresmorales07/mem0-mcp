import os
from typing import Any, Optional

import httpx
from mcp.server.fastmcp import FastMCP

MEM0_API_URL = os.environ.get("MEM0_API_URL", "").rstrip("/")
MEM0_API_KEY = os.environ.get("MEM0_API_KEY", "")
MEM0_DEFAULT_USER_ID = os.environ.get("MEM0_DEFAULT_USER_ID") or None
MEM0_DEFAULT_AGENT_ID = os.environ.get("MEM0_DEFAULT_AGENT_ID") or None

if not MEM0_API_URL:
    raise RuntimeError("MEM0_API_URL is required, e.g. http://dockerhost.home:8888")
if not MEM0_API_KEY:
    raise RuntimeError("MEM0_API_KEY is required (an m0sk_... key from the mem0 dashboard)")

client = httpx.Client(
    base_url=MEM0_API_URL,
    headers={"X-API-Key": MEM0_API_KEY, "Content-Type": "application/json"},
    timeout=30,
)

mcp = FastMCP("mem0")


def _identifiers(
    user_id: Optional[str],
    agent_id: Optional[str],
    run_id: Optional[str],
    tag_agent: bool = False,
) -> dict[str, str]:
    """Merge explicit scope with defaults.

    tag_agent=True (writes only) applies MEM0_DEFAULT_AGENT_ID unless the
    caller passes its own agent_id - it tags every memory with which MCP
    server wrote it. It must stay off for reads (list/search): since MCP tool
    calls can't distinguish "no agent_id given" from "search across all
    agents", auto-tagging reads would silently and permanently scope every
    query to this one agent, making it impossible to ever search a user's
    memories across agents again.

    MEM0_DEFAULT_USER_ID always applies as a last resort, when the caller
    supplies no user_id/run_id - reads and writes both benefit from a sane
    default "who is this" scope.
    """
    ids = {k: v for k, v in {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}.items() if v}
    if tag_agent and "agent_id" not in ids and MEM0_DEFAULT_AGENT_ID:
        ids["agent_id"] = MEM0_DEFAULT_AGENT_ID
    if "user_id" not in ids and "run_id" not in ids and MEM0_DEFAULT_USER_ID:
        ids["user_id"] = MEM0_DEFAULT_USER_ID
    if not ids:
        raise ValueError("Provide at least one of user_id, agent_id, or run_id.")
    return ids


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"mem0 API error ({response.status_code}): {detail}")


@mcp.tool()
def add_memory(
    text: str,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    """Store a new memory. Provide at least one of user_id, agent_id, or run_id
    to scope it; if none are given, falls back to MEM0_DEFAULT_USER_ID. Tagged
    with MEM0_DEFAULT_AGENT_ID unless agent_id is given explicitly."""
    body = {
        "messages": [{"role": "user", "content": text}],
        **_identifiers(user_id, agent_id, run_id, tag_agent=True),
    }
    response = client.post("/memories", json=body)
    _raise_for_status(response)
    return response.json()


@mcp.tool()
def search_memories(
    query: str,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    top_k: Optional[int] = None,
) -> dict[str, Any]:
    """Semantic search over stored memories, scoped to a user_id/agent_id/run_id."""
    body: dict[str, Any] = {"query": query, "filters": _identifiers(user_id, agent_id, run_id)}
    if top_k is not None:
        body["top_k"] = top_k
    response = client.post("/search", json=body)
    _raise_for_status(response)
    return response.json()


@mcp.tool()
def list_memories(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    top_k: Optional[int] = None,
) -> dict[str, Any]:
    """List stored memories for a given user_id/agent_id/run_id."""
    params = _identifiers(user_id, agent_id, run_id)
    if top_k is not None:
        params["top_k"] = str(top_k)
    response = client.get("/memories", params=params)
    _raise_for_status(response)
    return response.json()


@mcp.tool()
def get_memory(memory_id: str) -> dict[str, Any]:
    """Retrieve a single memory by its id."""
    response = client.get(f"/memories/{memory_id}")
    _raise_for_status(response)
    return response.json()


@mcp.tool()
def update_memory(
    memory_id: str,
    text: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Update an existing memory's text and/or metadata. Confirm the memory_id
    with the user first if it wasn't just returned by another tool call."""
    body: dict[str, Any] = {}
    if text is not None:
        body["text"] = text
    if metadata is not None:
        body["metadata"] = metadata
    response = client.put(f"/memories/{memory_id}", json=body)
    _raise_for_status(response)
    return response.json()


@mcp.tool()
def delete_memory(memory_id: str) -> str:
    """Delete a single memory by its id. Confirm the memory_id with the user
    first if it wasn't just returned by another tool call."""
    response = client.delete(f"/memories/{memory_id}")
    _raise_for_status(response)
    return "Memory deleted successfully"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
