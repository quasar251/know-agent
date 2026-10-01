"""LLM-as-judge for agent answers (end-to-end evaluation).

The retrieval harness measures *whether the right chunks came back*. This module
measures *whether the answer built on top of them is any good*, which is the only
signal that captures the agent's own contribution (prompt adherence, grounding,
refusal to hallucinate).

Two dimensions, each scored 1-5 by the judge and normalized to [0, 1]:

  * faithfulness — every factual claim in the answer must be traceable to the
    retrieved context. Low = hallucination.
  * correctness  — does the answer actually resolve the user's question.

The judge runs on the same env LLM as the agent (see ``resolve_env_llm``). Judge
failures never abort a run — they surface as ``None`` scores and are excluded
from the averages and counted separately.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# Judge answers are 1-5 integers (models are far more consistent with a small
# integer scale than with free-form floats); we normalize by this divisor.
_SCALE = 5.0

_PROMPT = """你是一名严格的 RAG 答案评审员。请依据【检索到的资料】评判【模型回答】。

【用户问题】
{query}

【检索到的资料】
{contexts}

【模型回答】
{answer}

请从两个维度打分，均为 1-5 的整数：
- faithfulness（忠实度）：回答中的事实性陈述是否都能在【检索到的资料】中找到依据。
  5 = 全部有依据；3 = 有少量发挥但主体有依据；1 = 大量编造或与资料矛盾。
- correctness（正确性）：回答是否准确且完整地解答了【用户问题】。
  5 = 完全正确；3 = 部分正确；1 = 完全错误或答非所问。

只输出一个 JSON 对象，不要输出任何其他文字、解释或代码块标记：
{{"faithfulness": <1-5>, "correctness": <1-5>, "reason": "<一句话理由>"}}"""


@dataclass
class JudgeScore:
    faithfulness: float | None  # normalized to [0, 1], None if the judge failed
    correctness: float | None
    reason: str = ""
    error: str | None = None


def _parse(raw: str) -> tuple[int, int, str]:
    """Extract the two integer scores from a judge reply.

    Tolerates ```json fences and surrounding prose by grabbing the first
    brace-balanced object. Raises ValueError when no usable score is found.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in judge reply: {raw[:120]!r}")
    obj = json.loads(match.group(0))
    faith = int(obj["faithfulness"])
    correct = int(obj["correctness"])
    reason = str(obj.get("reason") or "")
    if not (1 <= faith <= 5 and 1 <= correct <= 5):
        raise ValueError(f"scores out of 1-5 range: {obj}")
    return faith, correct, reason


async def judge_answer(
    client: Any,
    model: str,
    *,
    query: str,
    answer: str,
    contexts: str,
    max_context_chars: int = 6000,
) -> JudgeScore:
    """Score one answer. Never raises — failures come back as ``JudgeScore(None, None, error=...)``."""
    if not answer or not answer.strip():
        return JudgeScore(None, None, error="empty answer")
    if not contexts:
        # Nothing was retrieved: faithfulness is undefined, correctness still is.
        contexts = "(未检索到任何资料)"

    prompt = _PROMPT.format(
        query=query,
        contexts=contexts[:max_context_chars],
        answer=answer[:4000],
    )
    # Mirror the agent's own request config (LLM_EXTRA_BODY). Reasoning models
    # in thinking mode spend the whole max_tokens budget on hidden reasoning and
    # return empty `content`, which would make every score unparsable — and the
    # judge should in any case run under the same contract as the system it is
    # judging.
    from src.infra.llm import with_extra_body

    try:
        resp = await client.chat.completions.create(
            **with_extra_body(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 256,
                }
            )
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("judge call failed: %s", exc)
        return JudgeScore(None, None, error=f"judge call failed: {exc}")

    try:
        faith, correct, reason = _parse(raw)
    except Exception as exc:  # noqa: BLE001
        log.warning("judge reply unparsable: %s", exc)
        return JudgeScore(None, None, error=f"unparsable judge reply: {exc}")

    return JudgeScore(
        faithfulness=faith / _SCALE,
        correctness=correct / _SCALE,
        reason=reason,
    )