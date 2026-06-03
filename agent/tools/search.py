import os
from pydantic import BaseModel
from tavily import TavilyClient
from agent.tools.base import Tool


class _WebSearchInput(BaseModel):
    query: str
    max_results: int = 5


def _web_search(inputs: _WebSearchInput) -> list[dict]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set in the environment.")

    client = TavilyClient(api_key=api_key)
    response = client.search(query=inputs.query, max_results=inputs.max_results)
    return [
        {"url": r.get("url"), "title": r.get("title"), "snippet": r.get("content")}
        for r in response.get("results", [])
    ]


web_search = Tool(
    name="web_search",
    description=(
        "Searches the web for recent information using Tavily. "
        "Returns a list of results with url, title, and snippet fields. "
        "Use for news, current events, or any real-time information lookup."
    ),
    input_schema=_WebSearchInput,
    handler=_web_search,
)
