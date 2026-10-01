"""LangGraph nodes: plan, call_tools, skill_report."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, TYPE_CHECKING

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.state import AgentState
from src.infra.llm import (
    CostTracker,
    get_client,
    get_shared_client,
    pick_model,
    with_cache_control,
    with_extra_body,
    convert_to_openai_format,
)

# Captured so _resolve_llm_client can tell a monkeypatched factory from the
# real one (tests inject fake clients by replacing nodes.get_client).
_DEFAULT_GET_CLIENT = get_client
from src.safety.output_filter import StreamRedactor
from src.safety.tool_guard import is_tool_allowed
from src.skills.loader import invoke_skill
from src.tools.base import ToolRegistry

if TYPE_CHECKING:
    from src.settings_user import UserLLMConfig

MAX_ITERATIONS = 10
_KB_RETRIEVAL_LIMIT = 5

# v3-M9 (perf): `generate_kb_report` + its ~700-token system section used to be
# mounted on every KB turn. Mounting the tool also puts the request into
# tool-calling mode, which costs prefill tokens even when the tool is never
# called. Gate both on an explicit-report request in the user's message.
#
# Deliberately over-inclusive: false positives only cost the (old) prefill,
# while false negatives would silently break the report feature.
_KB_REPORT_INTENT_RE = re.compile(
    r"报告|报道|汇报|report"
    r"|(?:总结|汇总|归纳|梳理|整理|概括|提炼).{0,8}(?:文档|报告|成文|一份|材料|清单|表格|markdown|md)"
    r"|(?:生成|输出|导出|写成|整理成|做成).{0,4}(?:报告|文档|材料|markdown|md)"
    r"|\bmarkdown\b|\bmd\b",
    re.IGNORECASE,
)


def _wants_kb_report(text: str) -> bool:
    """True when the user's turn looks like an explicit "make me a report" ask."""
    return bool(text) and _KB_REPORT_INTENT_RE.search(text) is not None


def _resolve_llm_client(llm_cfg: "UserLLMConfig | None"):
    """Return the SDK client for a plan step, preferring a shared/cached one.

    A module-level override of ``get_client`` (used by tests to inject fake
    clients) always wins over the cache.
    """
    if get_client is not _DEFAULT_GET_CLIENT:
        return get_client(llm_cfg)
    return get_shared_client(llm_cfg)


async def retrieve_node(
    state: AgentState,
    *,
    registry: ToolRegistry,
    emit,
    limit: int = _KB_RETRIEVAL_LIMIT,
) -> AgentState:
    """v3-M8 (perf) KB-mode entry: run ONE ``search_kb`` up-front.

    KB chat used to let the LLM decide how many times to search, which cost a
    full (non-streaming) LLM round-trip per retrieval. Instead we embed the
    current question once, inject the chunks as a synthetic
    ``assistant(tool_use search_kb)`` + ``user(tool_result)`` pair, then hand
    off to ``plan_node`` which generates the answer directly. ``search_kb`` is
    hidden from the plan schema (see build_graph) so it can't be re-invoked.
    """
    if state.get("retrieved"):
        return state
    query = _last_user_text(state.get("messages", []))
    if not query:
        return state

    # v3-M9 (perf): configurable — these chunks are ~75% of the prompt, so the
    # count is the main input-side latency dial (prefill ≈ 0.22 ms / token).
    from src.settings import get_settings

    limit = limit or get_settings().kb_retrieval_limit

    await emit({"event": "tool_start", "name": "search_kb", "input": {"query": query}})
    result = await registry.call("search_kb", {"query": query, "limit": limit})
    await emit(
        {
            "event": "tool_end",
            "name": "search_kb",
            "latency_ms": result.latency_ms,
            "ok": result.error is None,
            "error": result.error,
        }
    )
    # Surface the retrieved sources (filename + relevance) so the chat UI can
    # render a "参考来源" list under the answer.
    raw = result.raw if isinstance(result.raw, dict) else {}
    await emit({"event": "sources", "sources": raw.get("sources") or []})
    chunks = result.text if result.error is None else f"[tool error] {result.error}"

    tcid = "kb-autoretrieval-1"
    messages = list(state.get("messages") or [])
    messages.append(
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": tcid, "name": "search_kb", "input": {"query": query}}
            ],
        }
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tcid,
                    "content": chunks,
                    "is_error": result.error is not None,
                }
            ],
        }
    )

    log = list(state.get("tool_call_log") or [])
    log.append(
        {
            "id": tcid,
            "name": "search_kb",
            "input": {"query": query},
            "result": chunks,
            "latency_ms": result.latency_ms,
            "error": result.error,
        }
    )

    return {**state, "messages": messages, "tool_call_log": log, "retrieved": True}


async def plan_node(
    state: AgentState,
    *,
    registry: ToolRegistry,
    cost: CostTracker,
    system_prompt: str = SYSTEM_PROMPT,
    include_travel_skill: bool = True,
    include_kb_skill: bool = False,
    llm_cfg: "UserLLMConfig | None" = None,
    emit=None,
    hidden_tools: frozenset[str] = frozenset(),
    kb_report_skill_prompt: str = "",
) -> AgentState:
    """LLM decides next action: call tools, call skill, or finish.

    The agent's prompt and the schema for the optional "skill" tools are
    injected by build_graph. KB-mode conversations get a different
    system_prompt + the generic `generate_kb_report` skill (v2-M8); travel
    KB gets `generate_travel_report`. Unbound chat mounts neither.

    v3-M1 (memory-optimization): constructs a layered prompt via
    ``build_context_sections`` + ``build_layered_prompt`` instead of
    passing ``system_prompt`` and ``messages`` directly to the LLM.

    v3-M2: injects the L4 early-summary layer when the session has compressed
    history. Requires ``conversation_id`` + ``user_id`` in state; when either
    is missing (old frontend) or short-term memory is off, L4 is simply
    omitted → identical to M1.

    v3-M8 (perf): when ``emit`` is wired the LLM call is streamed and text
    deltas are forwarded as ``token`` SSE events as they arrive (true
    streaming — first token reaches the client without waiting for the whole
    answer). ``hidden_tools`` lets KB mode drop ``search_kb`` from the schema
    once the ``retrieve`` node has already fetched chunks.
    """
    from src.agent.context_builder import build_layered_prompt
    from src.agent.prompts import build_context_sections
    from src.settings import get_settings

    # Early exit if final_report already set (by skill_report from prev tool wave)
    if state.get("final_report"):
        return {**state, "pending_tool_calls": []}

    iters = state.get("iterations", 0)
    if iters >= MAX_ITERATIONS:
        return {**state, "final_report": "超出最大推理轮数限制。", "pending_tool_calls": []}

    s = get_settings()

    messages = state.get("messages", [])
    extra: list[dict[str, Any]] = []
    if include_travel_skill:
        extra.append(_skill_tool_schema())
    if include_kb_skill and (
        not kb_report_skill_prompt
        or not s.kb_report_skill_gating
        or _wants_kb_report(_last_user_text(messages))
    ):
        # Mounting the tool also means mounting its system section — they are a
        # pair, otherwise the LLM would be told to call a tool it can't see.
        extra.append(_kb_skill_tool_schema())
        if kb_report_skill_prompt:
            system_prompt = f"{system_prompt}\n{kb_report_skill_prompt}"
    tools_schema = [
        t for t in registry.all_schemas() if t.get("name") not in hidden_tools
    ] + extra
    model = pick_model(messages, tools_schema, llm_cfg)
    # v3-M9 (perf): cached client — a fresh SDK client per turn meant a fresh
    # httpx pool, i.e. a DNS + TCP + TLS handshake before every first token.
    client = _resolve_llm_client(llm_cfg)

    # v3-M2: fetch the early-history summary (L4) for this conversation. Only
    # when short-term memory is on AND the session id flowed in; any failure or
    # missing id degrades to "" → L4 omitted (M1 behavior). Best-effort read
    # never blocks the plan step. Fetched at most ONCE per request: the result
    # (even "") is cached into agent state so later plan iterations (tool
    # rounds, up to MAX_ITERATIONS) don't re-hit Redis/PG.
    early_summary = state.get("early_summary")
    user_profile = state.get("user_profile")
    long_term_memory = state.get("long_term_memory")

    # v3-M9 (perf): L4 / L1 / L2 are three *independent* IO reads (Redis, PG,
    # Milvus + an embedding call) that used to be awaited one after another, so
    # the plan step paid the sum of their latencies before the LLM was even
    # called. They are issued concurrently below; each one keeps its original
    # best-effort contract (failure → empty layer, never a broken plan step).
    user_id = state.get("user_id")
    conv_id = state.get("conversation_id")
    need_l4 = early_summary is None and bool(conv_id and user_id)
    need_l1l2 = "long_term_memory" not in state and bool(user_id)

    if need_l4 or need_l1l2:
        from src.conversations.long_term_memory import (
            get_user_profile,
            retrieve_long_term_memories,
        )
        from src.conversations.short_term_memory import get_context_summary

        async def _l4() -> str:
            try:
                return (await get_context_summary(user_id, conv_id)) or ""
            except Exception:  # noqa: BLE001 — L4 is best-effort.
                return ""

        async def _l1() -> dict:
            try:
                return await get_user_profile(user_id)
            except Exception:  # noqa: BLE001 — L1 best-effort.
                return {}

        async def _l2() -> list:
            try:
                # L2 query is the current user input (last plain user message).
                return (
                    await retrieve_long_term_memories(user_id, _last_user_text(messages))
                    or []
                )
            except Exception:  # noqa: BLE001 — L2 best-effort.
                return []

        pending_coros: list[Any] = []
        if need_l4:
            pending_coros.append(_l4())
        if need_l1l2:
            pending_coros.append(_l1())
            pending_coros.append(_l2())

        results = await asyncio.gather(*pending_coros, return_exceptions=True)

        def _ok(idx: int, default: Any) -> Any:
            if idx >= len(results):
                return default
            val = results[idx]
            return default if isinstance(val, BaseException) else val

        cursor = 0
        if need_l4:
            early_summary = _ok(cursor, "")
            cursor += 1
        if need_l1l2:
            user_profile = _ok(cursor, {})
            long_term_memory = _ok(cursor + 1, [])
    elif early_summary is None:
        early_summary = ""

    # Build layered context (M1: L0 + L5; M2 adds L4; M3 adds L1 + L2 when present).
    sections = build_context_sections(
        system_prompt_text=system_prompt,
        recent_messages=messages,
        memory_window_size=s.memory_window_size,
        early_summary=early_summary,
        user_profile=user_profile,
        long_term_memories=long_term_memory,
    )
    layered = build_layered_prompt(
        sections,
        total_budget=s.context_total_budget,
    )

    # Decide API shape: anthropic vs openai-compat. User cfg wins; env fallback otherwise.
    if llm_cfg is not None:
        is_anthropic = llm_cfg.provider == "anthropic"
    else:
        is_anthropic = s.llm_provider == "anthropic"

    # v3-M8: with emit wired we stream the completion and forward text deltas
    # as ``token`` events as they arrive. StreamRedactor holds back a short tail
    # so a PII pattern is never split across two emitted chunks.
    streaming = emit is not None
    redactor = StreamRedactor() if streaming else None

    async def _emit_text(delta: str) -> None:
        if redactor is None or not delta:
            return
        piece = redactor.feed(delta)
        if piece:
            await emit({"event": "token", "text": piece})

    async def _flush_text() -> None:
        if redactor is None:
            return
        tail = redactor.flush()
        if tail:
            await emit({"event": "token", "text": tail})

    if not is_anthropic:
        # OpenAI-compatible (DeepSeek, OpenAI, vLLM, Together, Groq, LMStudio, etc.)
        _, openai_messages, openai_tools = convert_to_openai_format(
            layered.messages, tools_schema,
        )
        req: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": layered.system_text}] + openai_messages,
            "tools": openai_tools if openai_tools else None,
            # v3-M9 (perf): 2048 was a magic number ~14x the observed mean
            # output (142 tok). It never helped quality but it is what an
            # unbounded/hallucinating generation can grow into, so it is the
            # main lever on the latency long tail. Now configurable.
            "max_tokens": s.llm_max_tokens,
        }
        # Vendor extras (e.g. turning off a default thinking phase). Empty by
        # default so no provider sees an unexpected field.
        req = with_extra_body(req)
        answer_text = ""
        tool_calls: list[dict[str, Any]] = []

        if streaming:
            stream = await client.chat.completions.create(**req, stream=True)
            acc: dict[int, dict[str, str]] = {}
            usage = None
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = choices[0].delta
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if content:
                    answer_text += content
                    await _emit_text(content)
                for tc in getattr(delta, "tool_calls", None) or []:
                    slot = acc.setdefault(
                        getattr(tc, "index", 0) or 0,
                        {"id": "", "name": "", "arguments": ""},
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if fn.name:
                            slot["name"] = fn.name
                        if fn.arguments:
                            slot["arguments"] += fn.arguments
            if usage is not None:
                cost.add(model, usage)
            tool_calls = [
                {
                    "id": s["id"],
                    "name": s["name"],
                    "input": json.loads(s["arguments"]) if s["arguments"] else {},
                }
                for _, s in sorted(acc.items())
            ]
        else:
            resp = await client.chat.completions.create(**req)
            cost.add(model, resp.usage)
            choice = resp.choices[0]
            answer_text = choice.message.content or ""
            for tc in choice.message.tool_calls or []:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments) if tc.function.arguments else {}
                })

        # Build assistant message for history
        assistant_content = []
        if answer_text:
            assistant_content.append({"type": "text", "text": answer_text})
        for tc in tool_calls:
            assistant_content.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["name"],
                "input": tc["input"]
            })
    else:
        # Anthropic API
        system_blocks = with_cache_control(layered.system_blocks, llm_cfg)
        areq: dict[str, Any] = {
            "model": model,
            "max_tokens": s.llm_max_tokens,
            "system": system_blocks,
            "messages": layered.messages,
            "tools": tools_schema or None,
        }
        answer_text = ""
        tool_calls = []

        if streaming:
            parts: list[str] = []
            async with client.messages.stream(**areq) as stream:
                async for delta in stream.text_stream:
                    parts.append(delta)
                    await _emit_text(delta)
                final_msg = await stream.get_final_message()
            cost.add(model, final_msg.usage)
            answer_text = "".join(parts)
            tool_calls = [
                {"id": b.id, "name": b.name, "input": b.input}
                for b in final_msg.content
                if b.type == "tool_use"
            ]
            assistant_content = [
                b.model_dump() if hasattr(b, "model_dump") else dict(b)
                for b in final_msg.content
            ]
        else:
            resp = await client.messages.create(**areq)
            cost.add(model, resp.usage)
            for block in resp.content:
                if block.type == "text":
                    answer_text = f"{answer_text}\n{block.text}" if answer_text else block.text
                elif block.type == "tool_use":
                    tool_calls.append({"id": block.id, "name": block.name, "input": block.input})
            assistant_content = [
                b.model_dump() if hasattr(b, "model_dump") else dict(b) for b in resp.content
            ]

    if streaming:
        await _flush_text()

    new_messages = messages + [{"role": "assistant", "content": assistant_content}]
    final_report: str | None = state.get("final_report")

    # Stop condition: model returns text only AND no pending tools.
    answer_streamed = False
    if not tool_calls and answer_text and not final_report:
        final_report = answer_text
        # Signal the SSE layer that this answer already left via token events.
        answer_streamed = streaming

    return {
        **state,
        "messages": new_messages,
        "pending_tool_calls": tool_calls,
        "iterations": iters + 1,
        "final_report": final_report,
        # v3-M8 (perf): True when this final answer was already pushed to the
        # client token-by-token from plan_node → SSE layer must not replay it.
        "answer_streamed": answer_streamed,
        "cost_usd": cost.usd,
        # v3-M2: persist the (possibly empty) L4 summary so subsequent plan
        # iterations within this request skip the Redis/PG re-read.
        "early_summary": early_summary,
        # v3-M3: persist the (possibly empty) L1 profile + L2 memories so
        # subsequent plan iterations skip the re-read (once-per-request cache).
        "user_profile": user_profile,
        "long_term_memory": long_term_memory,
    }


async def call_tools_node(
    state: AgentState,
    *,
    registry: ToolRegistry,
    emit,
    llm_cfg: "UserLLMConfig | None" = None,
) -> AgentState:
    """Execute all pending tool calls concurrently.

    v2-M8: `llm_cfg` flows through to `invoke_skill` so the report skill
    uses the user's own LLM (v2-M1) instead of always env defaults.
    """
    pending = state.get("pending_tool_calls", [])
    if not pending:
        return state

    async def _run(tc: dict[str, Any]) -> dict[str, Any]:
        name = tc["name"]
        args = tc.get("input") or {}
        ok, reason = is_tool_allowed(
            name,
            registry.names() + ["generate_travel_report", "generate_kb_report"],
        )
        if not ok:
            await emit({"event": "tool_blocked", "name": name, "reason": reason})
            return {
                "type": "tool_result",
                "tool_use_id": tc["id"],
                "content": f"[blocked by safety] {reason}",
                "is_error": True,
            }
        await emit({"event": "tool_start", "name": name, "input": args})

        if name == "generate_travel_report":
            text = await invoke_skill("travel_report", args, llm_cfg=llm_cfg)
            await emit({"event": "tool_end", "name": name, "latency_ms": 0, "ok": True})
            return {"type": "tool_result", "tool_use_id": tc["id"], "content": text}

        if name == "generate_kb_report":
            text = await invoke_skill("general_report", args, llm_cfg=llm_cfg)
            await emit({"event": "tool_end", "name": name, "latency_ms": 0, "ok": True})
            return {"type": "tool_result", "tool_use_id": tc["id"], "content": text}

        result = await registry.call(name, args)
        await emit(
            {
                "event": "tool_end",
                "name": name,
                "latency_ms": result.latency_ms,
                "ok": result.error is None,
                "error": result.error,
            }
        )
        return {
            "type": "tool_result",
            "tool_use_id": tc["id"],
            "content": result.text if result.error is None else f"[tool error] {result.error}",
            "is_error": result.error is not None,
        }

    results = await asyncio.gather(*[_run(tc) for tc in pending])

    log = list(state.get("tool_call_log") or [])
    for tc, r in zip(pending, results, strict=False):
        log.append(
            {
                "id": tc["id"],
                "name": tc["name"],
                "input": tc.get("input") or {},
                "result": r["content"],
                "latency_ms": 0,
                "error": "yes" if r.get("is_error") else None,
            }
        )

    messages = list(state.get("messages") or [])
    messages.append({"role": "user", "content": results})

    # If a report skill was called, treat its result as final_report.
    skill_names = {"generate_travel_report", "generate_kb_report"}
    skill_call = next((p for p in pending if p["name"] in skill_names), None)
    final_report = state.get("final_report")
    if skill_call:
        for r in results:
            if r["tool_use_id"] == skill_call["id"] and not r.get("is_error"):
                final_report = r["content"]
                break

    return {
        **state,
        "messages": messages,
        "pending_tool_calls": [],
        "tool_call_log": log,
        "final_report": final_report,
    }


def should_continue(state: AgentState) -> str:
    if state.get("final_report"):
        return "end"
    if state.get("pending_tool_calls"):
        return "tools"
    return "end"


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """Extract the current user query for L2 memory recall (v3-M3).

    Returns the text of the most recent plain user turn — a ``user`` message
    whose content is a string, OR the concatenated ``text`` blocks of a
    list-form user message. Tool-result user messages (content is a list of
    ``tool_result`` blocks, no free text) contribute nothing, so this reliably
    returns the human's question even mid tool-loop. "" when none found.
    """
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            if content.strip():
                return content.strip()
            continue
        if isinstance(content, list):
            texts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            joined = " ".join(t for t in texts if t).strip()
            if joined:
                return joined
    return ""


def _skill_tool_schema() -> dict[str, Any]:
    return {
        "name": "generate_travel_report",
        "description": (
            "调用 travel_report skill 生成结构化 Markdown 旅行报告。"
            "数据齐全后调用此工具，传入收集到的所有信息。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "date": {"type": "string"},
                "weather": {"type": "string"},
                "restaurants": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "餐厅列表，每项含 name/addr/signature_dishes/why_recommended",
                },
                "user_intent": {"type": "string", "description": "用户原始诉求摘要"},
            },
            "required": ["city", "date"],
        },
    }


def _kb_skill_tool_schema() -> dict[str, Any]:
    """v2-M8: generic report skill for user KBs.

    Mounted on KB-bound conversations (non-travel). The LLM should call this
    only when the user explicitly asks for a report / summary / structured
    document, not for every Q&A turn — KB chat default behavior is still
    direct prose answers grounded in search_kb chunks.
    """
    return {
        "name": "generate_kb_report",
        "description": (
            "把当前对话基于知识库 chunks 整理成一份"
            "结构化 Markdown 报告。**仅当用户明确要求**「生成报告」/「总结成文档」/"
            "「整理一份」时调用；普通问答**不要**调用本工具，直接基于 chunks 作答即可。"
            "调用前你必须已经通过 search_kb 拿到足够内容；citations 字段必须如实引用使用过的来源。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "报告标题（名词短语，概括主旨）",
                },
                "tldr": {
                    "type": "string",
                    "description": "一句话结论，≤80 中文字",
                },
                "sections": {
                    "type": "array",
                    "description": "正文段落列表，按逻辑顺序排",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string"},
                            "content": {
                                "type": "string",
                                "description": "完整段落 Markdown，可含列表 / 引用 / 加粗",
                            },
                        },
                        "required": ["heading", "content"],
                    },
                },
                "citations": {
                    "type": "array",
                    "description": "引用来源列表，按引用顺序排",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tag": {
                                "type": "string",
                                "enum": ["📚 KB"],
                                "description": "📚 KB = search_kb chunk",
                            },
                            "source": {
                                "type": "string",
                                "description": "KB chunk 的 filename",
                            },
                            "score": {
                                "type": "number",
                                "description": "KB chunk 的相关度（0-1）",
                            },
                        },
                        "required": ["tag", "source"],
                    },
                },
            },
            "required": ["title", "tldr", "sections", "citations"],
        },
    }
