"""Dataset I/O + automatic question construction for retrieval evaluation.

Ground truth is document-level: each item pairs a natural-language query with
the set of source filenames that contain a valid answer. Two construction modes:

  * ``llm``        — sample real chunks from the KB and ask an LLM to write one
                     natural question answerable *only* by that chunk. The
                     chunk's filename becomes the relevant document.
  * ``extractive`` — offline fallback (no LLM key needed): derive a query
                     directly from the chunk text via sentence extraction.

Items are persisted as JSONL so a dataset built once can be reused across runs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class EvalItem:
    query: str
    relevant: list[str]  # filenames that count as correct answers
    meta: dict[str, Any] = field(default_factory=dict)


def load_dataset(path: str | Path) -> list[EvalItem]:
    """Load a JSONL dataset. Each line: {"query", "relevant": [...], "meta": {...}}."""
    items: list[EvalItem] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning("skipping malformed dataset line %d: %s", lineno, exc)
            continue
        query = (obj.get("query") or "").strip()
        relevant = [r for r in (obj.get("relevant") or []) if r]
        if not query or not relevant:
            log.warning("skipping incomplete dataset line %d", lineno)
            continue
        items.append(EvalItem(query=query, relevant=relevant, meta=obj.get("meta") or {}))
    return items


def save_dataset(path: str | Path, items: list[EvalItem]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Automatic dataset construction
# ---------------------------------------------------------------------------
_LLM_PROMPT = (
    "下面是一段来自知识库的文档片段。请根据这段内容，提出一个用户可能会问的、"
    "自然的中文问题，要求该问题的答案能在这段内容里找到，且问题不要直接照抄原文。"
    "只输出问题本身，不要任何解释或标点以外的多余文字。\n\n"
    "文档片段：\n{chunk}\n\n问题："
)


def _extractive_query(text: str, max_len: int = 60) -> str:
    """Derive a question-ish query offline by taking the first informative
    sentence of the chunk. Weak signal, but lets the harness run with no LLM."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    # Split on Chinese / ASCII sentence terminators.
    parts = re.split(r"[。！？!?\n]", cleaned)
    first = next((p for p in parts if len(p.strip()) >= 8), cleaned)
    first = first.strip()
    if len(first) > max_len:
        first = first[:max_len]
    return first


async def _llm_query(client: Any, model: str, text: str) -> str | None:
    prompt = _LLM_PROMPT.format(chunk=text[:1800])
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=128,
        )
        content = (resp.choices[0].message.content or "").strip()
        # Strip any wrapping quotes the model adds.
        return content.strip("「」\"'“”") or None
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM question generation failed: %s", exc)
        return None


async def build_dataset(
    pipeline: Any,
    n: int,
    *,
    generator: str = "llm",
    seed: int = 13,
    concurrency: int = 5,
) -> list[EvalItem]:
    """Construct `n` items by sampling the KB and generating queries.

    `generator="llm"` uses the env LLM (resolve_env_llm + get_client); if it is
    unavailable or a call fails, the item transparently falls back to the
    extractive query so the dataset never comes back empty.
    """
    from eval.retrieval import sample_chunks

    chunks = await sample_chunks(pipeline, n, seed=seed)
    if not chunks:
        return []

    client = None
    model = ""
    if generator == "llm":
        try:
            from src.infra.llm import get_client
            from src.settings_user.models import resolve_env_llm

            cfg = resolve_env_llm()
            client = get_client(cfg)
            model = cfg.default_model
            if not (client and model and cfg.api_key):
                log.warning("LLM not fully configured — falling back to extractive generator")
                client = None
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM init failed (%s) — falling back to extractive generator", exc)
            client = None

    sem = asyncio.Semaphore(max(1, concurrency))
    items: list[EvalItem] = []

    async def _one(chunk: dict[str, Any]) -> EvalItem:
        text = (chunk.get("text") or "").strip()
        query = None
        used = "extractive"
        if client is not None:
            async with sem:
                query = await _llm_query(client, model, text)
            if query:
                used = "llm"
        if not query:
            query = _extractive_query(text)
        return EvalItem(
            query=query,
            relevant=[chunk["filename"]],
            meta={
                "generator": used,
                "doc_id": chunk.get("doc_id", ""),
                "chunk_idx": chunk.get("chunk_idx", -1),
                "source_filename": chunk["filename"],
            },
        )

    results = await asyncio.gather(*(_one(c) for c in chunks))
    items = [r for r in results if r.query.strip()]
    if client is not None:
        await _aclose(client)
    return items


async def _aclose(client: Any) -> None:
    closer = getattr(client, "close", None) or getattr(client, "aclose", None)
    if closer is None:
        return
    try:
        res = closer()
        if asyncio.iscoroutine(res):
            await res
    except Exception as exc:  # noqa: BLE001
        log.debug("closing LLM client failed (ignored): %s", exc)