"""Short-term memory effect evaluation CLI.

    python -m eval.run_memory_eval                 # 5 cases x 3 arms
    python -m eval.run_memory_eval --cases 3
    python -m eval.run_memory_eval --rebuild-cases

See ``eval/memory_eval.py`` for the experiment design (A/B/C arms, why the facts
are random codes, and how to read the numbers).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.agent_eval import load_agent_context  # noqa: E402
from eval.memory_eval import (  # noqa: E402
    ARMS,
    CASE_SCHEMA_VERSION,
    FACT_CHOICES,
    ArmOutcome,
    build_cases,
    load_cases,
    run_arm,
    save_cases,
    summarise,
)

_DEFAULT_KB = "c5a60b1d-c4a2-475a-b921-1242d93f51c2"
_EVAL_DIR = Path(__file__).resolve().parent


async def _owner_memory_count(user_id: str) -> int:
    from sqlalchemy import func, select

    from src.conversations.models import UserMemory
    from src.infra.database import get_session_factory

    async with get_session_factory()() as sess:
        return int(
            (
                await sess.execute(
                    select(func.count()).select_from(UserMemory).where(
                        UserMemory.user_id == user_id,
                        UserMemory.deleted_at.is_(None),
                    )
                )
            ).scalar()
            or 0
        )


async def _run(args: argparse.Namespace) -> int:
    from src.infra.database import init_db

    await init_db()
    ctx = await load_agent_context(args.kb, model_override=args.model)
    print(f"KB: {ctx.kb.name}  ({ctx.kb.id})")
    print(f"owner: {ctx.user_email or '(none)'}  user_id={ctx.user_id}")

    if ctx.user_id is None:
        print("error: KB owner has no user row; memory layers need a user_id.")
        return 1

    # Short-term-only comparison: if L1/L2 already hold memories, arm A's
    # advantage could come from them instead of L4 and the experiment is void.
    n_mem = await _owner_memory_count(ctx.user_id)
    print(f"owner long-term memories: {n_mem}")
    if n_mem and not args.allow_existing_memories:
        print(
            "error: owner already has long-term memories, so a short-term-only "
            "comparison would be contaminated (L1/L2 would inject them into arm A "
            "and B alike). Purge them, use a fresh user, or pass "
            "--allow-existing-memories to override."
        )
        return 1

    # The dataset name encodes the shape, so a run can never silently reuse a
    # dataset built with a different fact/filler count.
    cases_path = Path(args.cases_file) if args.cases_file else (
        _EVAL_DIR / "datasets" /
        f"memory_short_term_v{CASE_SCHEMA_VERSION}_n{args.cases}"
        f"_f{args.facts_per_case}_fl{args.fillers}_s{args.seed}.jsonl"
    )
    if args.rebuild_cases or not cases_path.exists():
        cases = build_cases(
            args.cases, seed=args.seed,
            facts_per_case=args.facts_per_case, n_fillers=args.fillers,
        )
        save_cases(cases_path, cases)
        print(f"cases built: {cases_path} ({len(cases)})")
    else:
        cases = load_cases(cases_path)
        print(f"cases loaded: {cases_path} ({len(cases)})")
    if cases:
        print(f"per case: {len(cases[0].facts)} facts, "
              f"{len(cases[0].filler_turns)} fillers, "
              f"{1 + len(cases[0].filler_turns) + 2} turns × 3 arms "
              f"= {(1 + len(cases[0].filler_turns) + 2) * 3 * len(cases)} agent calls")

    print("\narms:")
    for name, spec in ARMS.items():
        print(f"  {name:<14} window={spec['window']} batch={spec['batch']} "
              f"send_ids={spec['send_ids']}")

    selected = [a for a in ARMS if a in set(args.arms)] if args.arms else list(ARMS)
    if not selected:
        print(f"error: --arms matched nothing; valid arms: {list(ARMS)}")
        return 1
    if len(selected) != len(ARMS):
        print(f"\n⚠️ 只跑 {selected} —— 对照不完整，仅用于校准（如测天花板），"
              "不要用这份结果下结论。")

    all_results: list[ArmOutcome] = []
    for arm in selected:
        print(f"\n--- running {arm} ---")
        rows = await run_arm(ctx, cases, arm)
        for r in rows:
            facts = "ERR" if r.error else f"{r.facts_found}/{r.facts_total}"
            pre = "kept" if r.constraint_kept else "LOST"
            cite = "kept" if r.citation_kept else "LOST"
            l4 = sorted(k for k in r.layers if not k.startswith("_"))
            print(f"  {r.case_id}  facts={facts:<5} prefix={pre:<5} citation={cite:<5} "
                  f"l4_kept={r.l4_facts_kept}/{r.facts_total}  layers={l4}  "
                  f"ttft={r.ttft_ms or 0:.0f}ms")
        all_results.extend(rows)

    agg = summarise(all_results)
    _print_report(agg, cases)

    report = {
        "mode": "short_term_memory",
        "kb_id": ctx.kb.id,
        "kb_name": ctx.kb.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_cases": len(cases),
            "seed": args.seed,
            "model_override": args.model,
            "cases_path": str(cases_path),
            "arms": ARMS,
            "auto_extract": False,
        },
        "arms": agg,
        "per_case": [r.__dict__ | {"retention": r.retention} for r in all_results],
    }

    out_path = Path(args.out) if args.out else (
        _EVAL_DIR / "results" /
        f"{args.kb}_memory_{datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nresults written to {out_path}")
    return 0


def _print_report(agg: dict[str, Any], cases: list[Any]) -> None:
    print(f"\n## 短期记忆效果评测（{len(cases)} 个 case × {len(agg)} 组）\n")

    # summarise() emits an entry per arm in ARMS, so "arm missing" shows up as
    # cases == 0 rather than a missing key.
    if any(row["cases"] == 0 for row in agg.values()):
        # Calibration view: no A/B/C triple, so no conclusions may be drawn.
        # Used to measure the ceiling of a probe before paying for a full run.
        print("> ⚠️ **仅校准**：不是完整的 A/B/C 对照，下面的数字只能用于判断"
              "「这个探针的天花板够不够高」，不能作为记忆效果的结论。\n")
        print("| 组 | 约束①前缀 | 95% CI | 约束②尾注 | 95% CI | 事实保留 | 有 L4 |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        for name, row in agg.items():
            if row["cases"] == 0:
                continue
            lo, hi = row["constraint_rate_ci95"]
            clo, chi = row["citation_rate_ci95"]
            print(f"| {name} | {row['constraint_rate']:.3f} | [{lo:.2f}, {hi:.2f}] | "
                  f"{row['citation_rate']:.3f} | [{clo:.2f}, {chi:.2f}] | "
                  f"{row['retention']:.3f} | {row['cases_with_l4']}/{row['cases']} |")
        return

    a, b, c = agg["A_memory_on"], agg["B_memory_off"], agg["C_full_window"]

    print("### 通道一：约束遵守率（两种约束对比，同轮同成本植入）\n")
    print("| 组 | 配置 | ① 前缀「【合规】」 | 95% CI | ② 尾注「引用来源:」 | 95% CI | 有 L4 |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for name, label, spec in (
        ("A_memory_on", "A 记忆开启", "window=2 batch=2 + 传 ids"),
        ("B_memory_off", "B 记忆关闭", "window=2 batch=2，不传 ids"),
        ("C_full_window", "C 全窗口", "window=0（不裁剪），不传 ids"),
    ):
        row = agg[name]
        lo, hi = row["constraint_rate_ci95"]
        clo, chi = row["citation_rate_ci95"]
        print(f"| {label} | {spec} | **{row['constraint_rate']:.3f}** "
              f"({row['constraint_kept']}/{row['cases']}) | [{lo:.2f}, {hi:.2f}] | "
              f"**{row['citation_rate']:.3f}** ({row['citation_kept']}/{row['cases']}) | "
              f"[{clo:.2f}, {chi:.2f}] | {row['cases_with_l4']}/{row['cases']} |")
    print("\n> ①是**无理由的格式仪式**，②是**任务对齐**的（KB 提示词本来就要求引用文件名），"
          "两者同轮植入、零额外成本。差多少就是「仪式 vs 任务」的差距。")
    print("> C 组的 ② 遵守率 = **该指标的天花板**：指令逐字在上下文里时模型能做到多少。")

    print("\n### 通道二：埋点事实保留率\n")
    print("| 组 | 事实保留率 | 95% CI | 命中 | case 级全对率 | 95% CI | 判「不属于 KB」 |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for name, row in agg.items():
        lo, hi = row["retention_ci95"]
        clo, chi = row["case_all_facts_ci95"]
        print(f"| {name} | {row['retention']:.3f} | [{lo:.2f}, {hi:.2f}] | "
              f"{row['facts_found']}/{row['facts_total']} | "
              f"{row['case_all_facts_rate']:.3f} ({row['cases_all_facts']}/{row['cases']}) | "
              f"[{clo:.2f}, {chi:.2f}] | {row['refused']}/{row['cases']} |")
    print("\n> 事实在同一个 case 内是**聚集**的（模型要么整轮用上记忆、要么整轮拒答），"
          "所以 CI 以 **case 数** 为分母更诚实；「事实保留率」列只是分辨率更高的原始率。")

    print("\n### L4 压缩保真度与污染（A 组）\n")
    print(f"- 事实保真度：**{a['l4_fidelity']:.3f}**（{a['l4_facts_kept']}/{a['facts_total']}）")
    print(f"- L4 中记录到助手「我无法记忆」类免责声明的 case："
          f"**{a['l4_records_refusal']}/{a['cases']}**"
          f"（这类声明会被每轮重新注入，令模型继续拒答）")
    if a["l4_facts_kept"] and a["retention"] < a["l4_fidelity"]:
        print(f"- ⚠️ 信息在 L4 里（{a['l4_fidelity']:.2f}）却没被用上（{a['retention']:.2f}）："
              "瓶颈在注入后的使用，不在压缩本身。")

    print("\n### 探针轮首包延迟（ms, p50）\n")
    for name, row in agg.items():
        print(f"- {name}: {row['probe_ttft_p50']}")

    print("\n### 判读（基于 95% CI 是否重叠，不做点估计比较）\n")

    def _sep(x: dict, y: dict, key: str) -> bool:
        """True when x and y have disjoint 95% CIs (the only claim we may make)."""
        xl, xh = x[f"{key}_ci95"]
        yl, yh = y[f"{key}_ci95"]
        return xh < yl or yh < xl

    # --- constraint channels: pick the one with the usable ceiling -----------
    # The ceiling test decides which constraint is worth concluding from: an
    # instruction even a full-context model honours only half the time carries
    # no usable signal about what memory lost.
    print("- **约束探针天花板**（C 组，指令逐字可见）："
          f"① 前缀 {c['constraint_rate']:.2f} / ② 尾注 {c['citation_rate']:.2f}")
    primary = "citation_rate" if c["citation_rate"] > c["constraint_rate"] else "constraint_rate"
    plabel = "②尾注" if primary == "citation_rate" else "①前缀"
    print(f"- 采用 **{plabel}** 作为主判据（天花板较高）")

    if _sep(c, b, primary):
        print(f"- **约束通道有效性成立** ✓：C({c[primary]:.2f}) 与 B({b[primary]:.2f}) "
              "的 CI 不重叠 —— 该指令确实只在历史可见时才生效。")
    else:
        print(f"- ⚠️ **约束通道有效性未成立**：C({c[primary]:.2f}) 与 B({b[primary]:.2f}) "
              f"的 CI 重叠，不能归因于记忆（case 数 {a['cases']}）。")

    if _sep(c, a, primary):
        print(f"- **记忆在该通道上有损**：A({a[primary]:.2f}) 与 C({c[primary]:.2f}) "
              f"的 CI 不重叠。对照 L4 保真度 {a['l4_fidelity']:.2f} 判断是压缩丢信息还是模型没用上。")
    else:
        print(f"- 记忆在该通道上的缺口**不可区分**："
              f"A({a[primary]:.2f}) 与 C({c[primary]:.2f}) 的 CI 重叠。")
    print(f"- 净增益：A − B = **{a[primary] - b[primary]:+.3f}**（{plabel}遵守率）")

    # --- fact channel ---
    print()
    if _sep(c, b, "retention"):
        print(f"- 事实通道有效性成立 ✓：C({c['retention']:.2f}) vs B({b['retention']:.2f}) 的 CI 不重叠。")
    if _sep(c, a, "retention"):
        print(f"- **记忆在事实通道上有损**：A({a['retention']:.2f}) vs C({c['retention']:.2f}) "
              f"CI 不重叠；而 A 组 L4 保真度为 {a['l4_fidelity']:.2f} —— "
              "信息在摘要里却没被用上，瓶颈在注入后的使用，不在压缩。")
    else:
        print(f"- 事实通道 A({a['retention']:.2f}) vs C({c['retention']:.2f})：CI 重叠，"
              "当前样本量下不可区分。")
    print(f"- 净增益：A − B = **{a['retention'] - b['retention']:+.3f}**（事实保留率）")

    if a["errors"] or b["errors"] or c["errors"]:
        print(f"\n⚠️ 失败 case 数：A={a['errors']} B={b['errors']} C={c['errors']}")


def main() -> int:
    p = argparse.ArgumentParser(description="Short-term memory effect evaluation.")
    p.add_argument("--kb", default=_DEFAULT_KB)
    p.add_argument("--model", default=None)
    p.add_argument("--cases", type=int, default=20, help="number of cases")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--facts-per-case", type=int, default=FACT_CHOICES,
                   help="planted facts per case (free precision knob: same turn)")
    p.add_argument("--fillers", type=int, default=3,
                   help="filler turns before the probe (2 is the minimum that "
                        "clears the window=2/batch=2 compression trigger)")
    p.add_argument("--arms", default="",
                   help="comma-separated subset of arms (e.g. C_full_window). "
                        "Incomplete sets are calibration only — no conclusions.")
    p.add_argument("--cases-file", default=None)
    p.add_argument("--rebuild-cases", action="store_true")
    p.add_argument("--allow-existing-memories", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    args.arms = [a.strip() for a in (args.arms or "").split(",") if a.strip()]
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
