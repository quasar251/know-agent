"""LangGraph graph construction."""
from __future__ import annotations

from typing import Any, Awaitable, Callable, TYPE_CHECKING

from langgraph.graph import END, StateGraph

from src.agent.nodes import call_tools_node, plan_node, retrieve_node, should_continue
from src.agent.prompts import (
    SYSTEM_PROMPT_GENERAL,
    SYSTEM_PROMPT_TRAVEL,
    build_kb_system_prompt,
)
from src.agent.state import AgentState
from src.infra.llm import CostTracker
from src.tools.base import ToolRegistry, build_default_registry

if TYPE_CHECKING:
    from src.settings_user import UserEmbeddingConfig, UserLLMConfig, UserRerankerConfig

Emitter = Callable[[dict[str, Any]], Awaitable[None]]


def build_graph(
    registry: ToolRegistry | None = None,
    emit: Emitter | None = None,
    *,
    kb=None,  # KB row from src.kb.models, or None for general chat mode
    llm_cfg: "UserLLMConfig | None" = None,
    embedding_cfg: "UserEmbeddingConfig | None" = None,
    reranker_cfg: "UserRerankerConfig | None" = None,
):
    """Wire up plan → call_tools loop, parameterized by KB context.

    Three modes (v2-M4):
      - kb=None: general chat — no tools, neutral assistant prompt (pure model
        direct answer). No travel fallback (that was v1, fixed in v2-M4).
      - kb=<system travel demo KB>: travel agent (weather + restaurant_kb +
        amap + generate_travel_report skill, travel prompt). Reachable only
        by explicitly selecting "TravelGPT 演示库".
      - kb=<user KB>: KB-bound mode (search_kb only, KB-specific prompt with
        strict local-grounding rules).

    v2-M1: `llm_cfg` and `embedding_cfg` are per-user overrides; None falls back
    to env-scoped defaults (so existing alice/bob keep working without
    visiting the settings page).

    v3-M4: `reranker_cfg` is a per-user opt-in cross-encoder reranker (default
    None = disabled). When set AND a user KB is selected, search_kb over-fetches
    candidates and reorders them via the configured /rerank endpoint. System
    KBs ignore reranker regardless. Hit `score` stays cosine.
    """
    from src.kb.models import SYSTEM_TRAVEL_KB_ID

    if registry is None:
        registry = build_default_registry(
            kb=kb,
            embedding_cfg=embedding_cfg,
            reranker_cfg=reranker_cfg,
        )

    if kb is None:
        system_prompt = SYSTEM_PROMPT_GENERAL
        include_travel_skill = False
        include_kb_skill = False
    elif kb.id == SYSTEM_TRAVEL_KB_ID:
        system_prompt = SYSTEM_PROMPT_TRAVEL
        include_travel_skill = True
        include_kb_skill = False
    else:
        system_prompt = build_kb_system_prompt(
            kb.name,
            kb.description or "",
        )
        include_travel_skill = False
        include_kb_skill = True

    cost = CostTracker()

    # v3-M8 (perf): KB-bound chats retrieve once up-front (retrieve → plan)
    # instead of letting the LLM loop over ``search_kb`` (a full non-streaming
    # round-trip per retrieval). ``search_kb`` is hidden from the plan schema so
    # it cannot be re-invoked; the chunks arrive as a synthetic tool_result.
    is_kb = kb is not None and kb.id != SYSTEM_TRAVEL_KB_ID

    async def _noop_emit(_evt: dict[str, Any]) -> None:
        return None

    em = emit or _noop_emit

    # Use functools.partial instead of lambda to avoid coroutine issues
    from functools import partial

    g = StateGraph(AgentState)
    g.add_node(
        "plan",
        partial(
            plan_node,
            registry=registry,
            cost=cost,
            system_prompt=system_prompt,
            include_travel_skill=include_travel_skill,
            include_kb_skill=include_kb_skill,
            llm_cfg=llm_cfg,
            emit=em,
            hidden_tools=frozenset({"search_kb"}) if is_kb else frozenset(),
        ),
    )
    g.add_node(
        "call_tools",
        partial(call_tools_node, registry=registry, emit=em, llm_cfg=llm_cfg),
    )

    if is_kb:
        g.add_node(
            "retrieve",
            partial(retrieve_node, registry=registry, emit=em),
        )
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "plan")
    else:
        g.set_entry_point("plan")

    g.add_conditional_edges("plan", should_continue, {"tools": "call_tools", "end": END})
    g.add_edge("call_tools", "plan")
    return g.compile(), cost
