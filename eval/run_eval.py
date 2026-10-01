"""Evaluation CLI: measure retrieval quality + latency for a KB.

Usage (from the repo root, using the backend venv):

    python -m eval.run_eval
    python -m eval.run_eval --kb <kb_id> --limit 5 --top-k 1,3,5 --repeat 3
    python -m eval.run_eval --gen extractive --rebuild-dataset

The dataset is auto-built from the KB's own chunks on first run (LLM-generated
questions, or extractive when --gen extractive / no LLM key) and cached under
eval/datasets/. Results are printed as a markdown report and saved to
eval/results/.

Runs fully offline except for embedding / rerank / (optional) LLM network calls.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow `python eval/run_eval.py` in addition to `python -m eval.run_eval`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import metrics as M
from eval.dataset import (
    EvalItem,
    build_dataset,
    load_dataset,
    save_dataset,
)
from eval.retrieval import Outcome, load_pipeline

_DEFAULT_KB = "c5a60b1d-c4a2-475a-b921-1242d93f51c2"
_EVAL_DIR = Path(__file__).resolve().parent


def _parse_top_k(raw: str, limit: int) -> list[int]:
    ks: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ks.append(int(part))
        except ValueError:
            pass
    if not ks:
        ks = [k for k in (1, 3, 5) if k <= limit] or [limit]
    return sorted({k for k in ks if k > 0})


def _score_item(outcome: Outcome, relevant: list[str], ks: list[int]) -> dict[str, float]:
    rel = set(relevant)
    ranked = outcome.ranked_docs
    scores: dict[str, float] = {"MRR": M.reciprocal_rank(ranked, rel)}
    for k in ks:
        scores[f"recall@{k}"] = M.recall_at_k(ranked, rel, k)
        scores[f"precision@{k}"] = M.precision_at_k(ranked, rel, k)
        scores[f"hit@{k}"] = M.hit_at_k(ranked, rel, k)
        scores[f"ndcg@{k}"] = M.ndcg_at_k(ranked, rel, k)
    return scores


async def _run(args: argparse.Namespace) -> int:
    pipeline = await load_pipeline(args.kb)
    print(f"KB: {pipeline.kb_name}  ({pipeline.kb_id})")
    print(f"collection: {pipeline.collection_name}  reranker={pipeline.reranker_active}  "
          f"grouping={pipeline.grouping_enabled}")

    ks = _parse_top_k(args.top_k, args.limit)
    if max(ks) > args.limit:
        print(f"note: top-k {max(ks)} > limit {args.limit}; raising limit to {max(ks)}")
        args.limit = max(ks)

    # --- dataset: load or build ---
    dataset_path = Path(args.dataset) if args.dataset else _EVAL_DIR / "datasets" / f"{args.kb}.jsonl"
    items: list[EvalItem]
    if dataset_path.exists() and not args.rebuild_dataset:
        items = load_dataset(dataset_path)
        print(f"dataset: {dataset_path} ({len(items)} items)")
    else:
        print(f"building dataset ({args.gen}, n={args.n}) from KB chunks ...")
        items = await build_dataset(
            pipeline, args.n, generator=args.gen, seed=args.seed
        )
        if not items:
            print("error: could not build a dataset — no chunks sampled from the KB.")
            return 1
        save_dataset(dataset_path, items)
        print(f"dataset: {dataset_path} ({len(items)} items)")

    if not items:
        print("error: dataset is empty.")
        return 1

    # --- warmup once so the first measured query isn't penalised by load ---
    await pipeline.warmup()

    # --- run ---
    per_query: list[dict[str, Any]] = []
    metric_values: dict[str, list[float]] = {}
    lat_embed: list[float] = []
    lat_search: list[float] = []
    lat_rerank: list[float] = []
    lat_total: list[float] = []
    errors = 0

    for i, item in enumerate(items, start=1):
        last: Outcome | None = None
        for _ in range(max(1, args.repeat)):
            outcome = await pipeline.retrieve(item.query, limit=args.limit)
            last = outcome
            if outcome.error:
                continue
            lat_embed.append(outcome.timings.embed_ms)
            lat_search.append(outcome.timings.search_ms)
            lat_total.append(outcome.timings.total_ms)
            if outcome.reranked:
                lat_rerank.append(outcome.timings.rerank_ms)

        assert last is not None
        if last.error:
            errors += 1
            per_query.append(
                {
                    "query": item.query,
                    "relevant": item.relevant,
                    "retrieved": [],
                    "metrics": {},
                    "total_ms": 0.0,
                    "error": last.error,
                }
            )
            print(f"[{i}/{len(items)}] ERROR: {last.error}  ({item.query[:30]})")
            continue

        scores = _score_item(last, item.relevant, ks)
        for name, val in scores.items():
            metric_values.setdefault(name, []).append(val)
        per_query.append(
            {
                "query": item.query,
                "relevant": item.relevant,
                "retrieved": last.ranked_docs,
                "metrics": scores,
                "total_ms": round(last.timings.total_ms, 2),
                "error": None,
            }
        )
        print(f"[{i}/{len(items)}] r@1={scores.get(f'recall@{ks[0]}', 0):.2f} "
              f"MRR={scores['MRR']:.2f}  ({last.timings.total_ms:.0f} ms)")

    quality = {name: (sum(vals) / len(vals) if vals else 0.0) for name, vals in metric_values.items()}
    latency = {
        "embed": M.latency_stats(lat_embed),
        "search": M.latency_stats(lat_search),
        "rerank": M.latency_stats(lat_rerank),
        "total": M.latency_stats(lat_total),
    }

    now = datetime.now(timezone.utc)
    report = {
        "kb_id": pipeline.kb_id,
        "kb_name": pipeline.kb_name,
        "collection_name": pipeline.collection_name,
        "reranker_active": pipeline.reranker_active,
        "grouping_enabled": pipeline.grouping_enabled,
        "generated_at": now.isoformat(),
        "config": {
            "limit": args.limit,
            "top_k": ks,
            "repeat": args.repeat,
            "generator": args.gen,
            "dataset_path": str(dataset_path),
            "n_items": len(items),
        },
        "quality": quality,
        "latency_ms": latency,
        "errors": errors,
        "per_query": per_query,
    }

    _print_report(report)

    out_path = Path(args.out) if args.out else (
        _EVAL_DIR / "results" / f"{args.kb}_{now.astimezone().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nresults written to {out_path}")
    return 0


def _print_report(report: dict[str, Any]) -> None:
    ks = report["config"]["top_k"]
    print("\n## 检索质量 (mean over {} queries)\n".format(report["config"]["n_items"]))
    header = "| metric | " + " | ".join(f"@{k}" for k in ks) + " |"
    sep = "| --- | " + " | ".join("---" for _ in ks) + " |"
    print(header)
    print(sep)
    for name in ("recall", "precision", "hit", "ndcg"):
        cells = [f"{report['quality'].get(f'{name}@{k}', 0.0):.3f}" for k in ks]
        print(f"| {name} | " + " | ".join(cells) + " |")
    print(f"\nMRR: {report['quality'].get('MRR', 0.0):.3f}")

    print("\n## 延迟 (ms)\n")
    print("| stage | mean | p50 | p90 | p95 | max |")
    print("| --- | --- | --- | --- | --- | --- |")
    for stage in ("embed", "search", "rerank", "total"):
        s = report["latency_ms"][stage]
        print(f"| {stage} | {s['mean']:.1f} | {s['p50']:.1f} | {s['p90']:.1f} | "
              f"{s['p95']:.1f} | {s['max']:.1f} |")

    if report["errors"]:
        print(f"\n{report['errors']} query/queries errored.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate KB retrieval quality + latency.")
    parser.add_argument("--kb", default=_DEFAULT_KB, help="KB id to evaluate")
    parser.add_argument("--dataset", default=None, help="path to dataset JSONL (default eval/datasets/<kb>.jsonl)")
    parser.add_argument("--limit", type=int, default=5, help="top-k retrieved per query (default 5)")
    parser.add_argument("--top-k", default="", help="comma-separated k values for metrics (default derived from limit)")
    parser.add_argument("--repeat", type=int, default=1, help="repeat each query N times for latency stability")
    parser.add_argument("--gen", choices=("llm", "extractive"), default="llm", help="dataset generator")
    parser.add_argument("--n", type=int, default=30, help="number of queries to build")
    parser.add_argument("--seed", type=int, default=13, help="sampling seed")
    parser.add_argument("--rebuild-dataset", action="store_true", help="force rebuild the dataset")
    parser.add_argument("--out", default=None, help="result JSON path")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())