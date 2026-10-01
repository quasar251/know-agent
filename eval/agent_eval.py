"""Agent end-to-end harness: drives the real LangGraph pipeline in-process.

`eval/retrieval.py` measures the retrieval half by re-implementing KBSearchTool.
This module instead runs the *actual agent* — the same `build_graph()` the
``POST /api/chat`` route uses — so it captures everything retrieval-only eval
cannot see: retrieval-then-generate ordering, prompt grounding, streaming
behaviour, token cost, and the sources the answer actually cites.

Why in-process rather than HTTP: the route's only extras are JWT auth, rate
limiting and SSE framing, none of which change the numbers we care about, while
standing up a server + login flow makes the harness slow and brittle. Config
resolution below deliberately mirrors ``app.py::_run_chat_session`` so the agent
behaves identically.

Timings come from timestamping the emitted event stream, which maps onto the real
user experience:

    t0 ──tool_end(search_kb)──► first token ──────────► done
         └── retrieve_ms ──┘   └──── generate_ms ────┘
         └────────────── total_ms ───────────────────┘
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.agent.graph import build_graph
from src.infra.database import get_session_factory
from src.kb.models import KB

log = logging.getLogger(__name__)


@dataclass
class KbRef:
    """Detached, self-contained copy of the KB fields the graph actually reads.

    `build_graph` / `KBSearchTool` only ever touch these six attributes, so this
    avoids passing an ORM instance whose session has been closed.
    """

    id: str
    name: str
    description: str
    collection_name: str
    is_system: bool
    grouping_enabled: bool


@dataclass
class AgentContext:
    kb: KbRef
    user_id: str | None
    user_email: str | None
    llm_cfg: Any
    embedding_cfg: Any
    reranker_cfg: Any

    @property
    def reranker_active(self) -> bool:
        return self.reranker_cfg is not None


@dataclass
class AgentOutcome:
    query: str
    answer: str = ""
    sources: list[str] = field(default_factory=list)  # cited filenames, in order
    contexts: str = ""  # chunks the agent actually retrieved (for the judge)
    # Which context layers (L0/L1/L2/L4/L5) actually reached the prompt, as
    # {section_key: char_count}. Only populated when the runner was asked to
    # capture it — this is how the memory eval proves a layer was injected
    # rather than inferring it from the answer.
    layers: dict[str, int] = field(default_factory=dict)
    ttft_ms: float | None = None
    last_token_ms: float | None = None  # ts of the final `token` event (client-visible)
    stream_ms: float | None = None  # last_token - ttft = pure token-generation window
    tail_ms: float | None = None  # total - last_token = post-stream overhead
    retrieve_ms: float | None = None
    generate_ms: float | None = None
    total_ms: float = 0.0
    iterations: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    cost_usd: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0
    error: str | None = None


async def load_agent_context(kb_id: str, *, model_override: str | None = None) -> AgentContext:
    """Resolve a KB + owner into the exact cfg set `_run_chat_session` builds.

    KB-level embedding / reranker cfg wins (v3-M7); LLM cfg is user-level with an
    env fallback (v2-M1), plus the optional per-request model override (v3-M6).
    """
    from dataclasses import replace as dc_replace

    from src.auth.models import User
    from src.settings_user.kb_resolvers import resolve_kb_embedding, resolve_kb_reranker
    from src.settings_user.models import resolve_env_llm, resolve_user_llm

    session_factory = get_session_factory()
    async with session_factory() as session:
        kb = await session.get(KB, kb_id)
        if kb is None:
            raise SystemExit(f"KB not found: {kb_id}")
        user = await session.get(User, kb.user_id) if kb.user_id else None

        # Materialize before the session closes — see KbRef docstring.
        kb_ref = KbRef(
            id=kb.id,
            name=kb.name or "",
            description=kb.description or "",
            collection_name=kb.collection_name,
            is_system=bool(getattr(kb, "is_system", False)),
            grouping_enabled=bool(getattr(kb, "grouping_enabled", False)),
        )
        user_id = user.id if user is not None else None
        user_email = user.email if user is not None else None

        llm_cfg = resolve_user_llm(user) if user is not None else None
        if model_override:
            if llm_cfg is None:
                llm_cfg = resolve_env_llm()
            llm_cfg = dc_replace(
                llm_cfg, default_model=model_override, complex_model=model_override
            )

        if user is not None:
            embedding_cfg = resolve_kb_embedding(kb, user)
            reranker_cfg = resolve_kb_reranker(kb, user)
        else:
            # No owner row → nothing to resolve user-level cfg from; the graph's
            # env fallback still covers the LLM.
            embedding_cfg = None
            reranker_cfg = None

    return AgentContext(
        kb=kb_ref,
        user_id=user_id,
        user_email=user_email,
        llm_cfg=llm_cfg,
        embedding_cfg=embedding_cfg,
        reranker_cfg=reranker_cfg,
    )


class AgentRunner:
    """Runs one KB-bound agent turn per query, in-process."""

    def __init__(self, ctx: AgentContext, *, capture_layers: bool = False) -> None:
        self.ctx = ctx
        # Opt-in prompt-layer instrumentation (see _spy_layers). Off by default so
        # the plain latency harness pays nothing.
        self.capture_layers = capture_layers

    @staticmethod
    def _spy_layers(sink: dict[str, int]):
        """Return a context manager that records which sections reach the prompt.

        ``plan_node`` imports ``build_context_sections`` lazily at call time, so
        patching the module attribute is enough to observe the real invocation.
        Sequential harness → no need for thread-local bookkeeping.
        """
        import contextlib

        import src.agent.prompts as prompts

        original = prompts.build_context_sections

        def spy(*args: Any, **kwargs: Any):
            sections = original(*args, **kwargs)
            sink.clear()
            for sec in sections:
                sink[sec.section_key] = len(str(sec.content))
            return sections

        @contextlib.contextmanager
        def _cm():
            prompts.build_context_sections = spy
            try:
                yield
            finally:
                prompts.build_context_sections = original

        return _cm()

    async def run(
        self,
        query: str,
        *,
        history: list[dict[str, str]] | None = None,
        conversation_id: str | None = None,
        user_id: str | None = None,
    ) -> AgentOutcome:
        """Run one turn.

        ``history`` is the accumulated prior user/assistant messages — the
        frontend replays the full thread on every POST /api/chat, so a multi-turn
        harness must do the same. ``conversation_id`` + ``user_id`` are both
        required for the memory layers (L1/L2/L4) to be fetched at all; omitting
        them reproduces the pre-memory baseline exactly (see app.py).
        """
        q = (query or "").strip()
        if not q:
            return AgentOutcome(query=query, error="query is empty")

        # Each turn gets a fresh graph so the CostTracker (created inside
        # build_graph) accounts for exactly this query — same as one HTTP request.
        events: list[tuple[float, dict[str, Any]]] = []
        t0 = time.perf_counter()

        async def emit(evt: dict[str, Any]) -> None:
            events.append((time.perf_counter() - t0, evt))

        graph, cost = build_graph(
            emit=emit,
            kb=self.ctx.kb,
            llm_cfg=self.ctx.llm_cfg,
            embedding_cfg=self.ctx.embedding_cfg,
            reranker_cfg=self.ctx.reranker_cfg,
        )

        initial_state: dict[str, Any] = {
            "messages": [*(history or []), {"role": "user", "content": q}],
            "iterations": 0,
            "tool_call_log": [],
        }
        if conversation_id and user_id:
            initial_state["conversation_id"] = conversation_id
            initial_state["user_id"] = user_id

        layers: dict[str, int] = {}
        try:
            if self.capture_layers:
                with self._spy_layers(layers):
                    final_state = await graph.ainvoke(initial_state)
            else:
                final_state = await graph.ainvoke(initial_state)
        except Exception as exc:  # noqa: BLE001
            log.warning("agent turn failed: %s", exc)
            return AgentOutcome(
                query=q,
                total_ms=(time.perf_counter() - t0) * 1000.0,
                error=f"agent failed: {exc}",
                layers=dict(layers),
            )

        total_ms = (time.perf_counter() - t0) * 1000.0
        outcome = self._collect(q, final_state, events, cost, total_ms)
        outcome.layers = dict(layers)
        return outcome

    @staticmethod
    def _collect(
        query: str,
        final_state: dict[str, Any],
        events: list[tuple[float, dict[str, Any]]],
        cost: Any,
        total_ms: float,
    ) -> AgentOutcome:
        ttft_ms: float | None = None
        retrieve_ms: float | None = None
        first_token_ts: float | None = None
        last_token_ts: float | None = None
        sources: list[str] = []

        for ts, evt in events:
            kind = evt.get("event")
            if kind == "tool_end" and evt.get("name") == "search_kb":
                retrieve_ms = ts * 1000.0
            elif kind == "sources":
                sources = [
                    str(s.get("filename", ""))
                    for s in (evt.get("sources") or [])
                    if s.get("filename")
                ]
            elif kind == "token":
                if first_token_ts is None:
                    first_token_ts = ts
                last_token_ts = ts  # keep the latest so we can time the last token
                _ = evt  # token text is only needed for TTFT/TTLT here
            elif kind == "error":
                return AgentOutcome(
                    query=query, total_ms=total_ms, error=f"agent error: {evt.get('message')}"
                )

        if first_token_ts is not None:
            ttft_ms = first_token_ts * 1000.0

        # Split the "生成段" into its two halves: the streaming window (first →
        # last token, i.e. real model throughput) and the tail after the last
        # token (graph bookkeeping / flush / SSE teardown).
        last_token_ms = last_token_ts * 1000.0 if last_token_ts is not None else None
        stream_ms = (
            last_token_ms - ttft_ms
            if last_token_ms is not None and ttft_ms is not None
            else None
        )
        tail_ms = total_ms - last_token_ms if last_token_ms is not None else None

        answer = (final_state.get("final_report") or "").strip()

        # The chunks the agent retrieved live in tool_call_log — this is what the
        # judge must see, since the SSE `sources` event carries filenames only.
        log_entries = final_state.get("tool_call_log") or []
        contexts = "\n\n".join(
            str(e.get("result") or "")
            for e in log_entries
            if e.get("name") == "search_kb" and not e.get("error")
        ).strip()

        tool_errors = sum(1 for e in log_entries if e.get("error"))

        return AgentOutcome(
            query=query,
            answer=answer,
            sources=sources,
            contexts=contexts,
            ttft_ms=ttft_ms,
            last_token_ms=last_token_ms,
            stream_ms=stream_ms,
            tail_ms=tail_ms,
            retrieve_ms=retrieve_ms,
            generate_ms=(total_ms - retrieve_ms) if retrieve_ms is not None else None,
            total_ms=total_ms,
            iterations=int(final_state.get("iterations", 0) or 0),
            tool_calls=len(log_entries),
            tool_errors=tool_errors,
            cost_usd=float(final_state.get("cost_usd", 0.0) or 0.0),
            in_tokens=int(getattr(cost, "input_tokens", 0) or 0),
            out_tokens=int(getattr(cost, "output_tokens", 0) or 0),
        )


async def warmup(ctx: AgentContext) -> None:
    """Load the Milvus collection so the first measured turn isn't penalised."""
    from src.infra.vector_store import get_store

    store = get_store()
    loader = getattr(store, "_ensure_loaded", None)
    if loader is None:
        return
    try:
        await loader(ctx.kb.collection_name)
    except Exception as exc:  # noqa: BLE001
        log.warning("agent warmup failed: %s", exc)


def load_judge_client() -> tuple[Any, str]:
    """Return (client, model) for the LLM judge, or (None, "") if unconfigured.

    Reuses the same env LLM the agent falls back to, so the harness needs no
    extra credentials.
    """
    from src.infra.llm import get_client
    from src.settings_user.models import resolve_env_llm

    cfg = resolve_env_llm()
    if not (cfg.api_key and cfg.default_model):
        return None, ""
    return get_client(cfg), cfg.default_model