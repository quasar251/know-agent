"""Short-term memory effect evaluation (multi-turn, v3-M2).

The 93 memory unit tests assert the *mechanism* (does the window trim, does the
summarizer fire, is L4 injected). None of them asserts the *effect* — that an
answer which was impossible without short-term memory becomes possible with it.
This harness measures exactly that, with a controlled three-arm experiment.

Design
------
Each case is a 6-round conversation whose FIRST round plants three facts that
cannot be guessed or derived (randomly generated codes, not domain knowledge),
followed by five neutral KB questions that push the planted round out of the L5
window, and finally one probe turn that asks for the planted facts.

    window=2 / batch=2  ->  L5 keeps only the last round verbatim, rounds 1-4
                            are folded into L4 (context_summary)

Three arms, same cases, same settings, differing only in what the agent sees:

  A  memory on   window=2, batch=2, conversation_id+user_id passed
                 -> planted round is reachable via L4
  B  memory off  window=2, batch=2, ids NOT passed
                 -> planted round is gone; this is the REPORT.md baseline
  C  full window window=0, batch=0, ids NOT passed
                 -> nothing is trimmed, the planted round is verbatim in L5
                 -> VALIDITY ARM: an upper bound that proves the case is
                    answerable at all. If C fails, the case is broken and
                    A/B prove nothing.

reading the result
------------------
  C >> B   case is valid: the fact is only reachable if the history survives
  A ~  C   short-term memory recovers essentially all of what an untrimmed
           history would give (compression is not lossy in practice)
  A <  C   compression lost something -> look at l4_fidelity per fact

Fact retention is measured by deterministic substring match on the normalised
answer, not by an LLM judge, so the number is reproducible and free.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Letters that survive OCR/typing confusion are irrelevant here, but avoid
# I/O/1/0 so a code can never be confused with a number when eyeballing output.
_CODE_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"

FILLER_QUESTIONS = [
    "这个知识库大概涵盖哪些方面的内容？",
    "研究生延期毕业一般最多能延多久？",
    "优秀研究生奖学金评审的对象是哪些研究生？",
    "学位论文的双盲评审一般由谁组织？",
    "研究生学籍管理规定适用于哪些学生？",
]

# --- constraint channel -------------------------------------------------
# Two standing instructions are planted in the same turn so one run measures
# both kinds. The comparison is the point: if only the task-aligned one survives,
# the earlier low ceiling was about the instruction being an arbitrary ritual;
# if BOTH decay, the ceiling is a property of cross-turn instructions generally.
#
# 1) `CONSTRAINT_TOKEN` — an arbitrary formatting ritual. Cheap to check, but it
#    demands something with no reason attached, so drift is expected.
# 2) `CITATION_MARKER` — task-aligned: the KB prompt already asks for filename
#    citations ("必要时引用 chunk 来源（filename）方便追溯"), so honouring it is
#    the same work the model is already inclined to do. Same cost to check.
CONSTRAINT_TOKEN = "【合规】"
CITATION_MARKER = "引用来源"
_CITATION_RE = re.compile(r"引用来源\s*[:：]")

# Bump whenever the case *content* changes (plant wording, constraint set, probe
# wording). It is part of the dataset filename so a stale JSONL can never be
# silently reused after a schema change — the failure mode that would otherwise
# be invisible, because the run would look perfectly healthy.
CASE_SCHEMA_VERSION = 2

# Facts planted per case. Deliberately 2 — NOT "as many as fit".
#
# More facts ride along in the same turn at zero extra LLM cost, which makes them
# look like a free precision knob. Measured: at 4 facts the L4 summary collapses
# to ~「（编号1等）」and fidelity drops from 1.00 to 0.125, because _compress_rounds
# targets a ≤150-char summary and simply omits what does not fit. Measuring that
# is measuring the summary's *capacity*, not short-term memory — a different
# experiment with a different conclusion. 2 keeps the planted content inside the
# budget so the metric reflects the memory path itself; sample size is bought
# with more cases instead.
FACT_CHOICES = 2


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — the right CI for a proportion at small n.

    The normal approximation is useless here (it happily returns bounds below 0
    or above 1 at n=5); Wilson stays inside [0, 1] and behaves at extremes,
    which is exactly the 0.00 / 1.00 cases this harness produces.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

# Markers that mean "the model declined because the question is out of KB scope".
_REFUSAL_MARKERS = ("kb 中没有相关内容", "不属于", "无法给出", "没有相关内容")

# Markers that mean "the assistant told the user it cannot remember".
# When the compressor folds one of these into L4, the refusal is re-injected on
# every later turn and the model keeps refusing — memory sealing itself shut.
_MEMORY_REFUSAL_MARKERS = (
    "拒绝记忆", "拒绝记录", "无法记忆", "无法跨", "无法持久", "无法记录",
    "不能记忆", "不记忆", "仅能基于知识库", "无法保存",
    # the same disclaimer phrased as "noted it down as unremembered"
    "未记录", "不予执行", "无法长期", "不保存",
)


def _norm(text: str) -> str:
    """Lowercase + strip everything but alphanumerics and CJK.

    Makes matching robust to how the model renders a code: ``ZEPHYR-4821``,
    ``zephyr 4821`` and ``ZEPHYR4821`` all normalise to ``zephyr4821``.
    """
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", (text or "").lower())


@dataclass
class MemoryCase:
    case_id: str
    plant_turn: str
    filler_turns: list[str]
    probe_fact: str           # asks for the planted facts
    probe_kb: str             # ordinary KB question, carries the constraint
    facts: list[str]          # display form, for the report
    facts_norm: list[str]     # normalised form, for matching
    constraint_token: str     # leading marker the user demanded in round 1
    citation_marker: str = CITATION_MARKER  # trailing citation the user demanded


@dataclass
class ArmOutcome:
    arm: str
    case_id: str
    answer: str = ""
    probe_kb_answer: str = ""
    # --- fact channel: expected to be ~0 everywhere, see module docstring ---
    facts_found: int = 0
    facts_total: int = 0
    refused: bool = False
    # --- constraint channel: the metrics short-term memory can actually move ---
    constraint_kept: bool = False   # arbitrary ritual: leading 【合规】
    citation_kept: bool = False     # task-aligned: trailing 「引用来源: <file>」
    l4_summary: str = ""
    l4_facts_kept: int = 0
    l4_records_refusal: bool = False
    layers: dict[str, int] = field(default_factory=dict)
    ttft_ms: float | None = None
    total_ms: float = 0.0
    error: str | None = None

    @property
    def retention(self) -> float:
        return self.facts_found / self.facts_total if self.facts_total else 0.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def build_cases(
    n: int = 5,
    seed: int = 17,
    *,
    facts_per_case: int = FACT_CHOICES,
    n_fillers: int = 3,
) -> list[MemoryCase]:
    """Deterministic case set — same seed => same planted codes.

    The codes are random by construction: the point is that no amount of
    pretrained knowledge or domain reasoning can produce ``ZEPHYR-4821``, so a
    correct answer can only come from the conversation history.

    ``facts_per_case`` is the cheapest precision knob available: extra planted
    facts ride along in the *same* plant turn, so they add observations at zero
    extra LLM cost. They are **not** independent samples though — a model that
    refuses the whole case fails all of them together — which is why the report
    also prints cluster-level (per-case) rates and CI on the case count.

    ``n_fillers`` only has to clear the compression trigger
    (``(rounds - compressed) - window >= batch`` → 2 rounds suffice at
    window=2/batch=2). 3 keeps margin; each one costs a full agent turn.
    """
    rng = random.Random(seed)
    cases: list[MemoryCase] = []
    for i in range(n):
        facts: list[str] = []
        for _ in range(max(1, facts_per_case)):
            code = "".join(rng.choice(_CODE_LETTERS) for _ in range(6))
            facts.append(f"{code}-{rng.randint(1000, 9999)}")
        listed = "\n".join(f"{j + 1}) 内部编号 {j + 1} 是 {f}；" for j, f in enumerate(facts))
        plant = (
            f"先同步 {len(facts)} 条内部编号，之后我会问你，请记住：\n"
            f"{listed}\n"
            f"另外定两条我们组的规矩，从现在起无论我问什么都必须遵守：\n"
            f"① 回答的第一行必须以「{CONSTRAINT_TOKEN}」这三个字开头；\n"
            f"② 回答的最后必须单独一行写出「{CITATION_MARKER}: <用到的文件名>」。\n"
            f"这两条是硬性要求，任何情况下都不能省略。"
        )
        probe_fact = (
            f"刚才那 {len(facts)} 条内部编号分别是什么？只列编号，不要展开。"
        )
        cases.append(
            MemoryCase(
                case_id=f"st-{i + 1:02d}",
                plant_turn=plant,
                filler_turns=list(FILLER_QUESTIONS[: max(1, n_fillers)]),
                probe_fact=probe_fact,
                probe_kb="研究生国家助学金的资助标准是怎么定的？",
                facts=facts,
                facts_norm=[_norm(f) for f in facts],
                constraint_token=CONSTRAINT_TOKEN,
                citation_marker=CITATION_MARKER,
            )
        )
    return cases


def save_cases(path: str | Path, cases: list[MemoryCase]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_cases(path: str | Path) -> list[MemoryCase]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[MemoryCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        out.append(
            MemoryCase(
                case_id=obj["case_id"],
                plant_turn=obj["plant_turn"],
                filler_turns=list(obj["filler_turns"]),
                probe_fact=obj["probe_fact"],
                probe_kb=obj["probe_kb"],
                facts=list(obj["facts"]),
                facts_norm=list(obj["facts_norm"]),
                constraint_token=obj.get("constraint_token") or CONSTRAINT_TOKEN,
                citation_marker=obj.get("citation_marker") or CITATION_MARKER,
            )
        )
    return out


def facts_present(text: str, facts_norm: list[str]) -> list[bool]:
    """Which planted facts appear verbatim (after normalisation) in *text*."""
    hay = _norm(text)
    return [f in hay for f in facts_norm]


# ---------------------------------------------------------------------------
# Settings per arm
# ---------------------------------------------------------------------------
def apply_memory_settings(*, window: int, batch: int, auto_extract: bool = False) -> None:
    """Vary the two short-term knobs for one arm.

    get_settings() is lru_cached and pydantic-settings prefers real env vars over
    the .env file, so setting os.environ + clearing the cache is enough to swap
    the configuration without spawning a subprocess per arm.

    ``auto_extract`` defaults to False so long-term memory (L1/L2) stays empty
    and cannot contaminate a short-term-only measurement.
    """
    from src.settings import get_settings

    os.environ["MEMORY_WINDOW_SIZE"] = str(window)
    os.environ["MEMORY_COMPRESSION_BATCH"] = str(batch)
    os.environ["MEMORY_AUTO_EXTRACT"] = "true" if auto_extract else "false"
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Conversation simulation
# ---------------------------------------------------------------------------
class ConversationSim:
    """Drives one conversation through the real persistence + agent path.

    Mirrors what the frontend does around POST /api/chat:
      * every turn appends a Message row (PG) and fires the short-term memory
        bookkeeping — exactly ``routes.append_message``'s two side effects,
        except ``_memory_update`` is awaited instead of fire-and-forget so the
        window/compression state is deterministic at the next turn;
      * the full thread is replayed as ``history`` on every turn, which is what
        makes the L5 window trimming observable.
    """

    def __init__(self, ctx: Any, *, title: str) -> None:
        self.ctx = ctx
        self.title = title
        self.conversation_id = str(uuid.uuid4())
        self.history: list[dict[str, str]] = []
        self._runner = None

    async def __aenter__(self) -> "ConversationSim":
        from src.auth.models import User
        from src.conversations.models import Conversation
        from src.conversations.short_term_memory import resolve_session_llm
        from src.infra.database import get_session_factory

        from eval.agent_eval import AgentRunner

        factory = get_session_factory()
        async with factory() as sess:
            sess.add(
                Conversation(
                    id=self.conversation_id,
                    user_id=self.ctx.user_id,
                    title=self.title,
                    kb_id=self.ctx.kb.id,
                )
            )
            await sess.commit()
            user = await sess.get(User, self.ctx.user_id)
            conv = await sess.get(Conversation, self.conversation_id)
            self.llm_cfg = resolve_session_llm(user, conv)

        self._runner = AgentRunner(self.ctx, capture_layers=True)
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.cleanup()

    async def _append(self, role: str, content: str) -> None:
        from src.conversations.models import Message
        from src.infra.database import get_session_factory

        async with get_session_factory()() as sess:
            sess.add(
                Message(
                    id=str(uuid.uuid4()),
                    conversation_id=self.conversation_id,
                    role=role,
                    content=content,
                )
            )
            await sess.commit()

    async def _memory_update(self, role: str, content: str) -> None:
        from src.conversations.short_term_memory import _memory_update

        await _memory_update(self.ctx.user_id, self.conversation_id, role, content, self.llm_cfg)

    async def say(self, text: str, *, send_ids: bool) -> Any:
        """One full user turn: persist -> memory bookkeeping -> agent -> persist."""
        await self._append("user", text)
        await self._memory_update("user", text)

        outcome = await self._runner.run(
            text,
            history=list(self.history),
            conversation_id=self.conversation_id if send_ids else None,
            user_id=self.ctx.user_id if send_ids else None,
        )
        answer = outcome.answer or ""

        await self._append("assistant", answer)
        await self._memory_update("assistant", answer)

        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": answer})
        return outcome

    async def summary(self) -> str:
        from src.conversations.short_term_memory import get_context_summary

        return await get_context_summary(self.ctx.user_id, self.conversation_id) or ""

    async def cleanup(self) -> None:
        from sqlalchemy import delete

        from src.conversations.models import Conversation, Message
        from src.conversations.short_term_memory import clear_conversation_hot_state
        from src.infra.database import get_session_factory

        await clear_conversation_hot_state(self.ctx.user_id, self.conversation_id)
        async with get_session_factory()() as sess:
            await sess.execute(delete(Message).where(Message.conversation_id == self.conversation_id))
            await sess.execute(delete(Conversation).where(Conversation.id == self.conversation_id))
            await sess.commit()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
ARMS: dict[str, dict[str, Any]] = {
    # arm -> settings + whether the graph receives conversation/user identity
    "A_memory_on": {"window": 2, "batch": 2, "send_ids": True},
    "B_memory_off": {"window": 2, "batch": 2, "send_ids": False},
    "C_full_window": {"window": 0, "batch": 0, "send_ids": False},
}


async def run_arm(ctx: Any, cases: list[MemoryCase], arm: str) -> list[ArmOutcome]:
    spec = ARMS[arm]
    apply_memory_settings(window=spec["window"], batch=spec["batch"])

    from eval.agent_eval import warmup

    await warmup(ctx)
    results: list[ArmOutcome] = []

    for case in cases:
        async with ConversationSim(ctx, title=f"[mem-eval] {arm} {case.case_id}") as sim:
            try:
                await sim.say(case.plant_turn, send_ids=spec["send_ids"])
                for filler in case.filler_turns:
                    await sim.say(filler, send_ids=spec["send_ids"])
                probe = await sim.say(case.probe_fact, send_ids=spec["send_ids"])
                probe_kb = await sim.say(case.probe_kb, send_ids=spec["send_ids"])
                summary = await sim.summary()
            except Exception as exc:  # noqa: BLE001
                log.warning("case %s failed: %s", case.case_id, exc)
                results.append(
                    ArmOutcome(arm=arm, case_id=case.case_id, error=str(exc)[:200],
                               facts_total=len(case.facts))
                )
                continue

        answer = probe.answer or ""
        kb_answer = probe_kb.answer or ""
        found = facts_present(answer, case.facts_norm)
        kept = facts_present(summary, case.facts_norm)
        low = _norm(kb_answer)
        results.append(
            ArmOutcome(
                arm=arm,
                case_id=case.case_id,
                answer=answer,
                probe_kb_answer=kb_answer,
                facts_found=sum(found),
                facts_total=len(case.facts),
                refused=any(m in low for m in _REFUSAL_MARKERS),
                constraint_kept=kb_answer.strip().startswith(case.constraint_token),
                citation_kept=_CITATION_RE.search(kb_answer) is not None,
                l4_summary=summary,
                l4_facts_kept=sum(kept),
                l4_records_refusal=any(m in summary for m in _MEMORY_REFUSAL_MARKERS),

                layers=dict(probe_kb.layers),
                ttft_ms=probe_kb.ttft_ms,
                total_ms=probe_kb.total_ms,
                error=probe_kb.error,
            )
        )
        log.info(
            "arm=%s case=%s facts=%d/%d prefix=%s citation=%s l4_kept=%d/%d layers=%s",
            arm, case.case_id, sum(found), len(case.facts),
            "kept" if kb_answer.strip().startswith(case.constraint_token) else "LOST",
            "kept" if _CITATION_RE.search(kb_answer) else "LOST",
            sum(kept), len(case.facts),
            sorted(k for k in probe_kb.layers if not k.startswith("_")),
        )
    return results


def summarise(results: list[ArmOutcome]) -> dict[str, Any]:
    """Aggregate both channels per arm + L4 fidelity + probe latency."""
    import statistics

    out: dict[str, Any] = {}
    for arm in ARMS:
        rows = [r for r in results if r.arm == arm and not r.error]
        total_facts = sum(r.facts_total for r in rows)
        found = sum(r.facts_found for r in rows)
        kept = sum(r.l4_facts_kept for r in rows)
        ttfts = [r.ttft_ms for r in rows if r.ttft_ms is not None]
        n_cases = len(rows)
        cons_kept = sum(1 for r in rows if r.constraint_kept)
        cite_kept = sum(1 for r in rows if r.citation_kept)
        cite_lo, cite_hi = wilson_ci(cite_kept, n_cases)
        # Cluster-level: a case counts as "recovered" only when EVERY planted
        # fact came back. Facts inside one case are not independent, so this is
        # the honest denominator for a CI.
        cases_all_facts = sum(1 for r in rows if r.facts_total and r.facts_found == r.facts_total)
        ret_lo, ret_hi = wilson_ci(found, total_facts)
        con_lo, con_hi = wilson_ci(cons_kept, n_cases)
        cluster_lo, cluster_hi = wilson_ci(cases_all_facts, n_cases)
        out[arm] = {
            "cases": n_cases,
            "errors": len([r for r in results if r.arm == arm and r.error]),
            "facts_found": found,
            "facts_total": total_facts,
            "retention": round(found / total_facts, 3) if total_facts else 0.0,
            "retention_ci95": [round(ret_lo, 3), round(ret_hi, 3)],
            "refused": sum(1 for r in rows if r.refused),
            "constraint_kept": cons_kept,
            "constraint_rate": round(cons_kept / n_cases, 3) if n_cases else 0.0,
            "constraint_rate_ci95": [round(con_lo, 3), round(con_hi, 3)],
            "citation_kept": cite_kept,
            "citation_rate": round(cite_kept / n_cases, 3) if n_cases else 0.0,
            "citation_rate_ci95": [round(cite_lo, 3), round(cite_hi, 3)],
            "cases_all_facts": cases_all_facts,
            "case_all_facts_rate": (
                round(cases_all_facts / n_cases, 3) if n_cases else 0.0
            ),
            "case_all_facts_ci95": [round(cluster_lo, 3), round(cluster_hi, 3)],
            "l4_facts_kept": kept,
            "l4_fidelity": round(kept / total_facts, 3) if total_facts else 0.0,
            # How many L4 summaries carry the assistant's own "I can't remember"
            # disclaimer — the self-sealing-refusal signal.
            "l4_records_refusal": sum(1 for r in rows if r.l4_records_refusal),
            "cases_with_l4": sum(1 for r in rows if r.layers.get("early_summary")),
            "probe_ttft_p50": round(statistics.median(ttfts), 1) if ttfts else None,
        }
    return out
