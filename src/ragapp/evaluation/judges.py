"""Judges that score one RAG answer.

Three metrics, all in [0, 1]:

- faithfulness        share of the answer's factual claims supported by the retrieved
                      context (the Ragas definition: decompose into claims, verify each).
                      An answer that makes no claims (a clean abstention) scores 1.0.
- answer_correctness  agreement with the reference answer, graded 1-5 and rescaled.
- context_recall      share of the reference answer's statements found in the retrieved
                      context. Measures the retriever, independent of the generator.
                      None for unanswerable questions (there is nothing to retrieve).

`LLMJudge` uses Claude with schema-constrained output. `HeuristicJudge` is a
lexical-overlap stand-in for offline runs and unit tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from ..config import Settings
from ..embeddings import content_tokens
from ..llm import LLM, Usage, build_llm


@dataclass
class JudgeScores:
    faithfulness: float
    answer_correctness: float
    context_recall: float | None
    details: dict = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    model: str = ""


class Judge(Protocol):
    name: str

    def evaluate(
        self, question: str, answer: str, contexts: list[str], ground_truth: str, answerable: bool
    ) -> JudgeScores: ...


# --------------------------------------------------------------------------- LLM judge


class ClaimVerdict(BaseModel):
    claim: str = Field(description="One atomic factual claim made by the answer.")
    reason: str = Field(description="Where the context supports it, or what is missing.")
    supported: bool


class FaithfulnessOutput(BaseModel):
    claims: list[ClaimVerdict]


class CorrectnessOutput(BaseModel):
    reasoning: str
    score: Literal[1, 2, 3, 4, 5]


class StatementVerdict(BaseModel):
    statement: str = Field(description="One atomic statement from the reference answer.")
    reason: str
    in_context: bool


class RecallOutput(BaseModel):
    statements: list[StatementVerdict]


FAITHFULNESS_SYSTEM = """You are auditing a retrieval-augmented question answering system for unsupported claims. \
Your verdicts gate a production release, so be exact rather than generous.

You are given the retrieved context, the user's question, and the system's answer. Do two things:

1. Break the answer into its atomic factual claims: each claim states one fact and can be checked on its own. \
Resolve pronouns so each claim stands alone. Skip citation markers like [1], and skip sentences that make no \
factual claim about the subject matter, such as a statement that the documents do not cover something or an \
offer to help further. If the answer makes no factual claims at all, return an empty list.

2. For each claim, decide whether the context supports it. A claim is supported only if the context states it or \
it follows directly from what the context states (including simple arithmetic or applying a stated rule to the \
case in the question). A claim that is true in the real world but absent from the context is not supported. \
A claim that contradicts the context is not supported.

Write the reason before the verdict."""

CORRECTNESS_SYSTEM = """You are grading a question answering system against a reference answer written by a subject expert.

Score how well the system's answer agrees with the reference:
5 - conveys everything in the reference, with no contradicting or incorrect additions
4 - conveys the key fact(s) of the reference; minor omissions or harmless extra detail
3 - partly correct: some key facts present, others missing or vague
2 - mostly wrong or missing, with a small correct element
1 - contradicts the reference, or does not answer

When the reference says the documentation does not cover the question, the correct behaviour is to say so: \
an answer that declines for that reason scores 5, and an answer that supplies facts anyway scores 1.

Wording and length do not matter, only the facts conveyed. Write your reasoning before the score."""

RECALL_SYSTEM = """You are measuring retrieval quality for a question answering system.

You are given the retrieved context and a reference answer written by a subject expert. Break the reference \
answer into its atomic statements, and for each one decide whether the retrieved context contains the information \
needed to make that statement. Judge only the context, not your own knowledge. Write the reason before the verdict."""


def _context_block(contexts: list[str]) -> str:
    body = "\n".join(
        f'<excerpt index="{i}">\n{c}\n</excerpt>' for i, c in enumerate(contexts, start=1)
    )
    return f"<context>\n{body}\n</context>"


class LLMJudge:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.name = f"llm:{llm.model}"

    def evaluate(
        self, question: str, answer: str, contexts: list[str], ground_truth: str, answerable: bool
    ) -> JudgeScores:
        context = _context_block(contexts)
        usage = Usage()

        faith, r = self.llm.structured(
            FAITHFULNESS_SYSTEM,
            f"{context}\n\n<question>\n{question}\n</question>\n\n<answer>\n{answer}\n</answer>",
            FaithfulnessOutput,
        )
        usage += r.usage
        model = r.model
        faithfulness = (
            sum(c.supported for c in faith.claims) / len(faith.claims) if faith.claims else 1.0
        )

        correct, r = self.llm.structured(
            CORRECTNESS_SYSTEM,
            f"<question>\n{question}\n</question>\n\n<reference_answer>\n{ground_truth}\n"
            f"</reference_answer>\n\n<system_answer>\n{answer}\n</system_answer>",
            CorrectnessOutput,
        )
        usage += r.usage

        recall_value = None
        recall_details: list[dict] = []
        if answerable:
            recall, r = self.llm.structured(
                RECALL_SYSTEM,
                f"{context}\n\n<reference_answer>\n{ground_truth}\n</reference_answer>",
                RecallOutput,
            )
            usage += r.usage
            recall_details = [s.model_dump() for s in recall.statements]
            recall_value = (
                sum(s.in_context for s in recall.statements) / len(recall.statements)
                if recall.statements
                else 1.0
            )

        return JudgeScores(
            faithfulness=faithfulness,
            answer_correctness=(correct.score - 1) / 4,
            context_recall=recall_value,
            details={
                "claims": [c.model_dump() for c in faith.claims],
                "correctness_reasoning": correct.reasoning,
                "reference_statements": recall_details,
            },
            usage=usage,
            model=model,
        )


# --------------------------------------------------------------------- heuristic judge

_ABSTAIN = re.compile(r"don't know|do not know|does not cover|doesn't cover|not covered|do not say")


def _recall(reference: str, candidate: str) -> float:
    ref = content_tokens(reference)
    return len(ref & content_tokens(candidate)) / len(ref) if ref else 1.0


class HeuristicJudge:
    """Token-overlap approximations of the three metrics. No model calls.

    Useful for exercising the eval and gate plumbing without an API key. It cannot
    detect paraphrased or subtly wrong claims, so never gate a real release on it.
    """

    name = "heuristic"

    def __init__(self, support_threshold: float = 0.6):
        self.support_threshold = support_threshold

    def evaluate(
        self, question: str, answer: str, contexts: list[str], ground_truth: str, answerable: bool
    ) -> JudgeScores:
        context_tokens = content_tokens(" ".join(contexts))
        abstained = bool(_ABSTAIN.search(answer.lower()))
        claims = []
        for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"\[\d+\]", "", answer)):
            tokens = content_tokens(sentence)
            if not tokens or _ABSTAIN.search(sentence.lower()):
                continue
            support = len(tokens & context_tokens) / len(tokens)
            claims.append(
                {
                    "claim": sentence.strip(),
                    "supported": support >= self.support_threshold,
                    "reason": f"{support:.0%} of content tokens appear in the context",
                }
            )
        faithfulness = sum(c["supported"] for c in claims) / len(claims) if claims else 1.0
        if answerable:
            correctness = _recall(ground_truth, answer)
            recall = _recall(ground_truth, " ".join(contexts))
        else:
            correctness = 1.0 if abstained and not claims else 0.0
            recall = None
        return JudgeScores(
            faithfulness=faithfulness,
            answer_correctness=correctness,
            context_recall=recall,
            details={"claims": claims},
            model=self.name,
        )


def build_judge(settings: Settings) -> Judge:
    if settings.llm_provider == "fake":
        return HeuristicJudge()
    return LLMJudge(build_llm(settings, role="judge"))
