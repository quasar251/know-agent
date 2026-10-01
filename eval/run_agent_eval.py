"""Agent end-to-end evaluation CLI.

Runs the real LangGraph agent (in-process) against the same 50-question dataset
the retrieval harness uses, then scores each answer with an LLM judge.

    python -m eval.run_agent_eval
    python -m eval.run_agent_eval --kb <kb_id> --max-items 10
    python -m eval.run_agent_eval --model glm-4-flash

Measures, per turn:
  latency  TTFT / 检索段 / 生成段 / 总耗时
  cost     cost_usd + in/out tokens
  citation sources 是否命中标准答案文档
  quality  LLM 评委的 faithfulness / correctness
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow `python eval/run_agent_eval.py` in addition to `python -m eval.run_agent_eval`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import metrics as M
from eval.agent_eval import (
    AgentOutcome,
    AgentRunner,
    load_agent_context,
    load_judge_client,
    warmup,
)
from eval.dataset import EvalItem, build_dataset, load_dataset, save_dataset
from eval.judge import judge_answer

_DEFAULT_KB = "c5a60b1d-c4a2-475a-b921-1242d93f51c2"
_EVAL_DIR = Path(__file__).resolve().parent
_CORRECT_THRESHOLD = 0.6  # correctness >= 0.6 (i.e. judge score >= 3/5) counts as a correct answer


async def _ensure_dataset(
    kb_id: str, dataset_path: Path, *, gen: str, n: int, seed: int, rebuild: bool
) -> list[EvalItem]:
    """Load the shared dataset, building it from KB chunks if it doesn't exist yet."""
    if dataset_path.exists() and not rebuild:
        items = load_dataset(dataset_path)
        print(f"dataset: {dataset_path} ({len(items)} items)")
        return items

    from eval.retrieval import load_pipeline

    print(f"building dataset ({gen}, n={n}) from KB chunks ...")
    pipeline = await load_pipeline(kb_id)
    items = await build_dataset(pipeline, n, generator=gen, seed=seed)
    if not items:
        return []
    save_dataset(dataset_path, items)
    print(f"dataset: {dataset_path} ({len(items)} items)")
    return items


async def _run(args: argparse.Namespace) -> int:
    ctx = await load_agent_context(args.kb, model_override=args.model)
    print(f"KB: {ctx.kb.name}  ({ctx.kb.id})")
    print(f"collection: {ctx.kb.collection_name}  reranker={ctx.reranker_active}  "
          f"grouping={ctx.kb.grouping_enabled}")
    if ctx.user_email:
        print(f"owner: {ctx.user_email}")

    runner = AgentRunner(ctx)
    judge_client, judge_model = load_judge_client()
    print(f"judge: {judge_model or '(unavailable — answers will not be scored)'}\n")

    dataset_path = (
        Path(args.dataset) if args.dataset else _EVAL_DIR / "datasets" / f"{args.kb}.jsonl"
    )
    items = await _ensure_dataset(
        args.kb,
        dataset_path,
        gen=args.gen,
        n=args.n,
        seed=args.seed,
        rebuild=args.rebuild_dataset,
    )
    if not items:
        print("error: dataset is empty (run `python -m eval.run_eval` first).")
        return 1
    if args.max_items and args.max_items < len(items):
        items = items[: args.max_items]
        print(f"using first {len(items)} items (--max-items)\n")

    await warmup(ctx)

    rows: list[dict[str, Any]] = []
    ttft: list[float] = []
    last_token: list[float] = []
    stream: list[float] = []
    tail: list[float] = []
    retrieve: list[float] = []
    generate: list[float] = []
    total: list[float] = []
    costs: list[float] = []
    cite_hits: list[float] = []
    cite_rr: list[float] = []
    cited_counts: list[int] = []
    faith: list[float] = []
    correct: list[float] = []
    tool_errors = 0
    agent_errors = 0
    judge_errors = 0

    for i, item in enumerate(items, start=1):
        outcome: AgentOutcome = await runner.run(item.query)
        if outcome.error:
            agent_errors += 1
            rows.append({"query": item.query, "relevant": item.relevant, "error": outcome.error})
            print(f"[{i}/{len(items)}] AGENT ERROR: {outcome.error}")
            continue

        tool_errors += outcome.tool_errors
        total.append(outcome.total_ms)
        costs.append(outcome.cost_usd)
        if outcome.ttft_ms is not None:
            ttft.append(outcome.ttft_ms)
        if outcome.last_token_ms is not None:
            last_token.append(outcome.last_token_ms)
        if outcome.stream_ms is not None:
            stream.append(outcome.stream_ms)
        if outcome.tail_ms is not None:
            tail.append(outcome.tail_ms)
        if outcome.retrieve_ms is not None:
            retrieve.append(outcome.retrieve_ms)
        if outcome.generate_ms is not None:
            generate.append(outcome.generate_ms)

        rel = set(item.relevant)
        cite_hit = M.hit_at_k(outcome.sources, rel, len(outcome.sources) or 1) if outcome.sources else 0.0
        cite_hits.append(cite_hit)
        cite_rr.append(M.reciprocal_rank(outcome.sources, rel))
        cited_counts.append(len(outcome.sources))

        score = await judge_answer(
            judge_client,
            judge_model,
            query=item.query,
            answer=outcome.answer,
            contexts=outcome.contexts,
        ) if judge_client is not None else None
        if score is not None:
            if score.error:
                judge_errors += 1
            if score.faithfulness is not None:
                faith.append(score.faithfulness)
            if score.correctness is not None:
                correct.append(score.correctness)

        rows.append(
            {
                "query": item.query,
                "relevant": item.relevant,
                "answer": outcome.answer,
                # Persisted so answers can be re-judged offline (e.g. by a
                # different judge model) without re-running the agent —
                # without this a re-judge would score against empty context.
                "contexts": outcome.contexts,
                "sources": outcome.sources,
                "cite_hit": cite_hit,
                "ttft_ms": round(outcome.ttft_ms, 1) if outcome.ttft_ms is not None else None,
                "last_token_ms": round(outcome.last_token_ms, 1) if outcome.last_token_ms is not None else None,
                "stream_ms": round(outcome.stream_ms, 1) if outcome.stream_ms is not None else None,
                "tail_ms": round(outcome.tail_ms, 1) if outcome.tail_ms is not None else None,
                "retrieve_ms": round(outcome.retrieve_ms, 1) if outcome.retrieve_ms is not None else None,
                "generate_ms": round(outcome.generate_ms, 1) if outcome.generate_ms is not None else None,
                "total_ms": round(outcome.total_ms, 1),
                "cost_usd": round(outcome.cost_usd, 6),
                "in_tokens": outcome.in_tokens,
                "out_tokens": outcome.out_tokens,
                "iterations": outcome.iterations,
                "tool_calls": outcome.tool_calls,
                "faithfulness": score.faithfulness if score else None,
                "correctness": score.correctness if score else None,
                "judge_reason": score.reason if score else "",
                "error": None,
            }
        )
        faith_txt = f"{score.faithfulness:.2f}" if score and score.faithfulness is not None else "-"
        corr_txt = f"{score.correctness:.2f}" if score and score.correctness is not None else "-"
        stream_txt = f"{outcome.stream_ms:.0f}" if outcome.stream_ms is not None else "-"
        print(
            f"[{i}/{len(items)}] cite={'✓' if cite_hit else '✗'} "
            f"faith={faith_txt} corr={corr_txt}  "
            f"ttft={outcome.ttft_ms or 0:.0f}ms stream={stream_txt}ms "
            f"out={outcome.out_tokens}tok total={outcome.total_ms:.0f}ms"
        )

    if judge_client is not None:
        await _aclose(judge_client)

    report = {
        "mode": "agent",
        "kb_id": ctx.kb.id,
        "kb_name": ctx.kb.name,
        "collection_name": ctx.kb.collection_name,
        "reranker_active": ctx.reranker_active,
        "grouping_enabled": ctx.kb.grouping_enabled,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_items": len(rows),
            "n_scored": len(correct),
            "model_override": args.model,
            "judge_model": judge_model,
            "dataset_path": str(dataset_path),
            "correct_threshold": _CORRECT_THRESHOLD,
        },
        "latency_ms": {
            "ttft": M.latency_stats(ttft),
            "last_token": M.latency_stats(last_token),
            "stream": M.latency_stats(stream),
            "tail": M.latency_stats(tail),
            "retrieve": M.latency_stats(retrieve),
            "generate": M.latency_stats(generate),
            "total": M.latency_stats(total),
        },
        "cost": {
            "total_usd": round(sum(costs), 6),
            "mean_usd": round(sum(costs) / len(costs), 6) if costs else 0.0,
            "in_tokens": sum(r.get("in_tokens", 0) for r in rows),
            "out_tokens": sum(r.get("out_tokens", 0) for r in rows),
        },
        "citation": {
            "hit_rate": _mean(cite_hits),
            "mrr": _mean(cite_rr),
            "mean_cited": _mean(cited_counts),
        },
        "quality": {
            "faithfulness": _mean(faith),
            "correctness": _mean(correct),
            "correct_rate": (
                sum(1 for v in correct if v >= _CORRECT_THRESHOLD) / len(correct)
                if correct
                else 0.0
            ),
        },
        "reliability": {
            "agent_errors": agent_errors,
            "tool_errors": tool_errors,
            "judge_errors": judge_errors,
        },
        "per_query": rows,
    }

    _print_report(report)

    out_path = Path(args.out) if args.out else (
        _EVAL_DIR / "results" / f"{args.kb}_agent_{datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nresults written to {out_path}")
    return 0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_report(report: dict[str, Any]) -> None:
    cfg = report["config"]
    print(f"\n## 端到端指标（{cfg['n_items']} 轮，评委打分 {cfg['n_scored']} 轮）\n")

    print("### 答案质量（LLM 评委，0-1）\n")
    print("| 指标 | 得分 |")
    print("| --- | --- |")
    print(f"| 忠实度 faithfulness | {report['quality']['faithfulness']:.3f} |")
    print(f"| 正确性 correctness | {report['quality']['correctness']:.3f} |")
    print(f"| 正确率 (≥{cfg['correct_threshold']}) | {report['quality']['correct_rate']:.3f} |")

    print("\n### 引用准确性\n")
    print("| 指标 | 值 |")
    print("| --- | --- |")
    print(f"| 引用命中率 | {report['citation']['hit_rate']:.3f} |")
    print(f"| 引用 MRR | {report['citation']['mrr']:.3f} |")
    print(f"| 平均引用数 | {report['citation']['mean_cited']:.2f} |")

    print("\n### 延迟（ms）\n")
    print("| 阶段 | mean | p50 | p90 | p95 | max |")
    print("| --- | --- | --- | --- | --- | --- |")
    for stage, label in (
        ("ttft", "首 token"),
        ("last_token", "末 token"),
        ("stream", "流式生成 (首→末)"),
        ("tail", "流式后收尾"),
        ("retrieve", "检索段"),
        ("generate", "生成段"),
        ("total", "总耗时"),
    ):
        s = report["latency_ms"][stage]
        print(f"| {label} | {s['mean']:.1f} | {s['p50']:.1f} | {s['p90']:.1f} | "
              f"{s['p95']:.1f} | {s['max']:.1f} |")

    c = report["cost"]
    print("\n### 成本\n")
    print(f"- 总成本: ${c['total_usd']:.6f}（平均 ${c['mean_usd']:.6f}/轮）")
    print(f"- tokens: in {c['in_tokens']} / out {c['out_tokens']}")

    r = report["reliability"]
    print(
        f"\n### 可靠性\n\n- agent 失败 {r['agent_errors']} · 工具失败 {r['tool_errors']} "
        f"· 评委失败 {r['judge_errors']}"
    )


async def _aclose(client: Any) -> None:
    """Best-effort close of the judge client; failure to close is not a run failure."""
    closer = getattr(client, "close", None) or getattr(client, "aclose", None)
    if closer is None:
        return
    try:
        res = closer()
        if asyncio.iscoroutine(res):
            await res
    except Exception as exc:  # noqa: BLE001
        print(f"warning: closing judge client failed (ignored): {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the LangGraph agent end-to-end and score answers with an LLM judge."
    )
    parser.add_argument("--kb", default=_DEFAULT_KB, help="KB id to evaluate")
    parser.add_argument("--dataset", default=None, help="dataset JSONL (default eval/datasets/<kb>.jsonl)")
    parser.add_argument("--max-items", type=int, default=0, help="only run the first N items")
    parser.add_argument("--model", default=None, help="override the agent's LLM model")
    parser.add_argument("--gen", choices=("llm", "extractive"), default="llm")
    parser.add_argument("--n", type=int, default=50, help="dataset size when building from scratch")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--out", default=None, help="result JSON path")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())