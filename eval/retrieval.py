"""Retrieval pipeline replica for evaluation.

`KBSearchTool.execute()` returns a *formatted* result whose `raw["sources"]` is
sorted by cosine score and de-duplicated to filenames — the rerank order is
lost. To evaluate quality we need the full, ordered hit list, so this module
re-implements the exact same pipeline against the same infra layer:

    embed(query, cfg)
      → get_store()
      → collection_supports_hybrid() ? hybrid_search(...) : search(...)
      → rerank(query, texts, top_n=original_limit, cfg)   (if enabled)
      → hits[:original_limit]

with per-stage wall-clock timings. The constants and ordering mirror
`src/tools/kb_search.py` line-for-line so numbers measured here describe the
production retrieval path.

This module is read-only w.r.t. the vector store — it never upserts or deletes.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

from src.infra.database import get_session_factory
from src.infra.embedding import clear_cache as clear_embed_cache
from src.infra.embedding import embed
from src.infra.reranker import rerank
from src.infra.vector_store import get_store
from src.kb.models import KB
from src.settings_user.kb_resolvers import resolve_kb_embedding, resolve_kb_reranker

log = logging.getLogger(__name__)

# Mirror src/tools/kb_search.py. The over-fetch knobs are resolved through the
# production helper (not a second hard-coded copy) so an env override can never
# apply to the agent but silently not to this harness.
from src.tools.kb_search import overfetch_params as _overfetch_params

_MAX_LIMIT = 20


@dataclass
class Hit:
    rank: int  # 1-based position in the final result list
    doc_id: str
    chunk_idx: int
    filename: str
    score: float  # cosine similarity (dense), Qdrant semantics
    rerank_score: float | None  # cross-encoder score when reranked, else None
    text: str


@dataclass
class Timings:
    embed_ms: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class Outcome:
    query: str
    hits: list[Hit] = field(default_factory=list)
    timings: Timings = field(default_factory=Timings)
    reranked: bool = False
    error: str | None = None

    @property
    def ranked_docs(self) -> list[str]:
        return [h.filename for h in self.hits]


@dataclass
class Pipeline:
    """Bound to one KB collection with resolved embedding/reranker configs."""

    kb_id: str
    kb_name: str
    collection_name: str
    embedding_cfg: Any
    reranker_cfg: Any
    grouping_enabled: bool

    @property
    def reranker_active(self) -> bool:
        return self.reranker_cfg is not None

    async def warmup(self) -> None:
        """Load the collection so the first measured query isn't penalised by
        Milvus's load-collection cost."""
        store = get_store()
        loader = getattr(store, "_ensure_loaded", None)
        if loader is not None:
            try:
                await loader(self.collection_name)
            except Exception as exc:  # noqa: BLE001
                log.warning("warmup failed for %s: %s", self.collection_name, exc)

    async def retrieve(self, query: str, limit: int = 5) -> Outcome:
        q = (query or "").strip()
        if not q:
            return Outcome(query=query, error="query is empty")

        original_limit = max(1, min(int(limit) if limit else 5, _MAX_LIMIT))
        if self.reranker_cfg:
            overfetch_multiplier, overfetch_cap = _overfetch_params()
            fetch_limit = min(original_limit * overfetch_multiplier, overfetch_cap)
        else:
            fetch_limit = original_limit

        # The production code memoizes query embeddings (infra/embedding.py), so
        # `--repeat N` on the same query would report embed ≈ 0 ms for every
        # repeat after the first and the latency table would stop being
        # comparable with a cold run. This harness measures the *cold* cost of
        # one retrieval, so the memo is busted per measured query.
        clear_embed_cache()

        t_start = time.perf_counter()

        # --- stage 1: query embedding ---
        t0 = time.perf_counter()
        try:
            vec = await embed(q, cfg=self.embedding_cfg)
        except Exception as exc:  # noqa: BLE001
            return Outcome(query=q, error=f"embed failed: {exc}")
        embed_ms = (time.perf_counter() - t0) * 1000.0

        # --- stage 2: vector (or hybrid) search ---
        t0 = time.perf_counter()
        try:
            store = get_store()
            if not hasattr(store, "search") or not self.collection_name:
                return Outcome(
                    query=q,
                    error="KB search requires a multi-collection backend (qdrant or milvus)",
                )
            supports_hybrid = (
                hasattr(store, "hybrid_search")
                and hasattr(store, "collection_supports_hybrid")
                and await store.collection_supports_hybrid(self.collection_name)
            )
            if supports_hybrid:
                hits = await store.hybrid_search(
                    query_vector=vec,
                    query_text=q,
                    collection_name=self.collection_name,
                    limit=fetch_limit,
                    group_by="doc_id" if self.grouping_enabled else None,
                )
            else:
                hits = await store.search(
                    vec,
                    collection_name=self.collection_name,
                    limit=fetch_limit,
                )
        except Exception as exc:  # noqa: BLE001
            return Outcome(query=q, error=f"vector search failed: {exc}")
        search_ms = (time.perf_counter() - t0) * 1000.0

        # --- stage 3: cross-encoder rerank (opt-in) ---
        reranked = False
        rerank_ms = 0.0
        # Parallel to `hits`: the rerank score for each kept hit (None if not reranked).
        ordered_scores: list[float | None] = [None] * len(hits)
        if self.reranker_cfg and len(hits) >= 2:
            texts = [(h.get("payload") or {}).get("text", "") or "" for h in hits]
            t0 = time.perf_counter()
            try:
                reordered = await rerank(
                    q, texts, top_n=original_limit, cfg=self.reranker_cfg
                )
                if reordered:
                    kept = [hits[idx] for idx, _ in reordered if 0 <= idx < len(hits)]
                    ordered_scores = [sc for idx, sc in reordered if 0 <= idx < len(hits)]
                    hits = kept
                    reranked = True
            except Exception as exc:  # noqa: BLE001
                log.warning("rerank failed (falling back to dense order): %s", exc)
            rerank_ms = (time.perf_counter() - t0) * 1000.0

        hits = hits[:original_limit]
        ordered_scores = ordered_scores[:original_limit]

        out_hits: list[Hit] = []
        for i, h in enumerate(hits, start=1):
            p = h.get("payload") or {}
            out_hits.append(
                Hit(
                    rank=i,
                    doc_id=str(p.get("doc_id", "")),
                    chunk_idx=int(p.get("chunk_idx", -1) or -1),
                    filename=str(p.get("filename", "(unknown)") or "(unknown)"),
                    score=float(h.get("score", 0.0)),
                    rerank_score=ordered_scores[i - 1],
                    text=(p.get("text") or "").strip(),
                )
            )

        total_ms = (time.perf_counter() - t_start) * 1000.0
        return Outcome(
            query=q,
            hits=out_hits,
            timings=Timings(
                embed_ms=embed_ms,
                search_ms=search_ms,
                rerank_ms=rerank_ms,
                total_ms=total_ms,
            ),
            reranked=reranked,
        )


async def load_pipeline(kb_id: str) -> Pipeline:
    """Resolve a KB row + owner configs into a ready-to-run Pipeline.

    Mirrors the config resolution KBSearchTool receives from the search node:
    KB-level embedding / reranker cfg wins, owner-level is the fallback, and
    system KBs bypass the reranker entirely.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        kb = await session.get(KB, kb_id)
        if kb is None:
            raise SystemExit(f"KB not found: {kb_id}")
        user = None
        if getattr(kb, "user_id", None):
            from src.auth.models import User

            user = await session.get(User, kb.user_id)

        embedding_cfg = resolve_kb_embedding(kb, user)
        reranker_cfg = resolve_kb_reranker(kb, user)
        if bool(getattr(kb, "is_system", False)):
            reranker_cfg = None  # v3-M4: system KBs always bypass reranker

        return Pipeline(
            kb_id=kb.id,
            kb_name=kb.name,
            collection_name=kb.collection_name,
            embedding_cfg=embedding_cfg,
            reranker_cfg=reranker_cfg,
            grouping_enabled=bool(getattr(kb, "grouping_enabled", False)),
        )


# ---------------------------------------------------------------------------
# Chunk sampling — used to bootstrap a dataset from the KB's own content
# ---------------------------------------------------------------------------
async def sample_chunks(
    pipeline: Pipeline, n: int, *, seed: int = 13, min_chars: int = 80
) -> list[dict[str, Any]]:
    """Return up to `n` random chunks from the KB collection.

    Each item: {"doc_id", "chunk_idx", "filename", "text"}. Sampling is
    deterministic for a given (seed, collection) so re-runs are reproducible.
    """
    store = get_store()
    rows = await _fetch_all_chunks(store, pipeline.collection_name, n)
    rows = [r for r in rows if len((r.get("text") or "").strip()) >= min_chars]
    if not rows:
        return []
    rng = random.Random(seed)
    if len(rows) <= n:
        rng.shuffle(rows)
        return rows
    return rng.sample(rows, n)


async def _fetch_all_chunks(
    store: Any, collection_name: str, n: int
) -> list[dict[str, Any]]:
    """Fetch a candidate pool of chunks from Milvus or Qdrant."""
    fetch_cap = min(max(n * 10, 200), 2000)

    # Milvus Lite: query() with an all-match filter, then load collection first.
    if hasattr(store, "_client") and hasattr(store._client, "query"):
        loader = getattr(store, "_ensure_loaded", None)
        if loader is not None:
            await loader(collection_name)
        fields = ["text", "filename", "doc_id", "chunk_idx"]
        try:
            raw = await asyncio.to_thread(
                store._client.query,
                collection_name=collection_name,
                filter='id != ""',
                output_fields=fields,
                limit=fetch_cap,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("milvus chunk sampling failed: %s", exc)
            return []
        return [
            {
                "doc_id": str(r.get("doc_id", "")),
                "chunk_idx": int(r.get("chunk_idx", -1) or -1),
                "filename": str(r.get("filename", "(unknown)") or "(unknown)"),
                "text": (r.get("text") or ""),
            }
            for r in raw
        ]

    # Qdrant: scroll through points.
    if hasattr(store, "_client") and hasattr(store._client, "scroll"):
        try:
            points, _ = await store._client.scroll(
                collection_name=collection_name,
                limit=fetch_cap,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("qdrant chunk sampling failed: %s", exc)
            return []
        out = []
        for p in points:
            payload = p.payload or {}
            out.append(
                {
                    "doc_id": str(payload.get("doc_id", "")),
                    "chunk_idx": int(payload.get("chunk_idx", -1) or -1),
                    "filename": str(payload.get("filename", "(unknown)") or "(unknown)"),
                    "text": (payload.get("text") or ""),
                }
            )
        return out

    log.warning("chunk sampling unsupported for store %s", type(store).__name__)
    return []