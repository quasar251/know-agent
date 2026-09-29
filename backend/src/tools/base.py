"""ToolRegistry — async tool abstraction with Anthropic-compatible schema."""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    text: str
    latency_ms: int
    raw: Any = None
    error: str | None = None


class Tool(abc.ABC):
    name: str
    description: str
    input_schema: dict[str, Any]

    @abc.abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult: ...

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all_schemas(self) -> list[dict[str, Any]]:
        return [t.to_anthropic() for t in self._tools.values()]

    async def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(text="", latency_ms=0, error=f"Unknown tool: {name}")
        start = time.perf_counter()
        try:
            result = await tool.execute(**args)
            if result.latency_ms == 0:
                result.latency_ms = int((time.perf_counter() - start) * 1000)
            return result
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                text="", latency_ms=int((time.perf_counter() - start) * 1000), error=str(exc)
            )


def build_default_registry(
    kb=None,
    embedding_cfg=None,
    *,
    reranker_cfg=None,
) -> ToolRegistry:
    """Build the agent's tool set based on which KB (if any) is active.

    Three cases (v2-M4):
      1. kb=None — general chat mode. No tools: pure model direct answer.
      2. kb=<system travel demo KB> — travel four-tool kit (weather + restaurant_kb
         + amap + the `generate_travel_report` skill that's wired in `graph.py`).
         Travel behavior is reachable only via this explicit selection.
      3. kb=<user KB> — KB-bound mode: `search_kb` only. Answers are grounded
         strictly in local KB chunks (no web fallback).

    `embedding_cfg` (v2-M1): per-user embedding override, threaded through to
    `KBSearchTool` so query embedding uses the user's chosen provider. None =
    fall back to env config.

    `reranker_cfg` (v3-M4): per-user cross-encoder reranker override, threaded
    through to `KBSearchTool` for second-stage rerank of search hits. None =
    skip rerank (default). System KBs ignore this regardless.
    """
    from src.kb.models import SYSTEM_TRAVEL_KB_ID

    reg = ToolRegistry()

    # General chat mode — no tools. LLM answers from pretraining knowledge only;
    # keeps the toolset minimal so the agent doesn't drift toward travel / KB
    # tools when no KB is selected.
    if kb is None:
        return reg

    # Built-in travel demo KB — keep v1 four-tool kit.
    if kb.id == SYSTEM_TRAVEL_KB_ID:
        from src.tools.amap_fallback import AmapFallbackTool
        from src.tools.restaurant_rag import RestaurantRagTool
        from src.tools.weather import WeatherTool

        reg.register(WeatherTool())
        reg.register(RestaurantRagTool())
        reg.register(AmapFallbackTool())
        return reg

    # User-created KB — search_kb only (strictly grounded in local chunks).
    from src.tools.kb_search import KBSearchTool

    reg.register(KBSearchTool(kb=kb, embedding_cfg=embedding_cfg, reranker_cfg=reranker_cfg))
    return reg
