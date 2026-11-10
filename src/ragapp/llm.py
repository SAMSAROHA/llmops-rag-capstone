"""LLM backends. `AnthropicLLM` calls Claude; `FakeLLM` is an offline extractive stand-in."""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel

from .config import Settings
from .embeddings import content_tokens

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(*(a + b for a, b in zip(asdict(self).values(), asdict(other).values())))

    @property
    def total_tokens(self) -> int:
        return sum(asdict(self).values())


@dataclass
class LLMResult:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end_turn"
    request_id: str | None = None


class LLMError(RuntimeError):
    """The model call completed but produced no usable output (refusal, truncation)."""


class LLM(Protocol):
    model: str

    def generate(self, system: str, user: str) -> LLMResult: ...

    def structured(self, system: str, user: str, schema: type[T]) -> tuple[T, LLMResult]: ...


def _usage(raw) -> Usage:
    return Usage(
        input_tokens=raw.input_tokens or 0,
        output_tokens=raw.output_tokens or 0,
        cache_read_input_tokens=getattr(raw, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(raw, "cache_creation_input_tokens", 0) or 0,
    )


class AnthropicLLM:
    def __init__(
        self,
        model: str,
        effort: str = "medium",
        max_tokens: int = 16000,
        refusal_fallback: bool = True,
        client=None,
    ):
        import anthropic

        # Zero-arg client resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login`.
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.refusal_fallback = refusal_fallback

    def generate(self, system: str, user: str) -> LLMResult:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
        )
        if self.refusal_fallback:
            response = self.client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **kwargs
            )
        else:
            response = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in response.content if b.type == "text")
        return LLMResult(
            text=text.strip(),
            model=response.model,  # the model that actually served (may be a fallback)
            usage=_usage(response.usage),
            stop_reason=response.stop_reason,
            request_id=response._request_id,
        )

    def structured(self, system: str, user: str, schema: type[T]) -> tuple[T, LLMResult]:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
            output_format=schema,
        )
        result = LLMResult(
            text="",
            model=response.model,
            usage=_usage(response.usage),
            stop_reason=response.stop_reason,
            request_id=response._request_id,
        )
        if response.stop_reason in ("refusal", "max_tokens") or response.parsed_output is None:
            raise LLMError(f"structured call ended with stop_reason={response.stop_reason}")
        return response.parsed_output, result


_DOC = re.compile(r'<document index="(\d+)"[^>]*>\n(.*?)\n</document>', re.S)
_QUESTION = re.compile(r"<question>\n?(.*?)\n?</question>", re.S)
class FakeLLM:
    """Offline stand-in: answers with the retrieved sentences that best overlap the question.

    It exists so the pipeline, cost accounting, tracing and gate plumbing can be
    exercised with no API key. `hallucinate=True` makes it invent unsupported
    claims, which is how the tests prove the release gate actually fails.
    """

    def __init__(self, model: str = "fake-extractive", hallucinate: bool = False):
        self.model = model
        self.hallucinate = hallucinate

    def generate(self, system: str, user: str) -> LLMResult:
        question = (_QUESTION.search(user) or [None, user])[1]
        sentences = []
        for index, body in _DOC.findall(user):
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
                sentence = sentence.strip().lstrip("#-* ").strip()
                if len(sentence) > 20 and sentence not in {x[0] for x in sentences}:
                    sentences.append((sentence, index, content_tokens(sentence)))
        # IDF over the retrieved sentences: a query term found nowhere gets the
        # maximum weight, so a question about an absent topic scores low.
        n = len(sentences) or 1
        idf = {
            t: math.log(1 + n / (1 + sum(t in toks for _, _, toks in sentences)))
            for t in content_tokens(question)
        }
        total = sum(idf.values()) or 1.0
        scored = sorted(
            ((sum(idf[t] for t in idf if t in toks) / total, s, i) for s, i, toks in sentences),
            key=lambda x: -x[0],
        )
        if self.hallucinate:
            text = (
                "The platform was founded on the moon in 1887 by a consortium of dolphins. "
                "Every customer receives a complimentary gold-plated server each quarter."
            )
        elif not scored or scored[0][0] < 0.4:
            text = "I don't know based on the provided documents."
        else:
            keep = [x for x in scored[:2] if x[0] >= 0.75 * scored[0][0]]
            text = " ".join(f"{s} [{i}]" for _, s, i in keep)
        usage = Usage(input_tokens=len(system + user) // 4, output_tokens=len(text) // 4)
        return LLMResult(
            text=text, model=self.model, usage=usage, request_id=f"fake_{uuid.uuid4().hex[:12]}"
        )

    def structured(self, system: str, user: str, schema: type[T]) -> tuple[T, LLMResult]:
        raise NotImplementedError("FakeLLM has no judge; use HeuristicJudge offline.")


def build_llm(settings: Settings, role: str = "generator") -> LLM:
    if settings.llm_provider == "fake":
        return FakeLLM()
    if role == "judge":
        return AnthropicLLM(
            settings.judge_model, settings.judge_effort, settings.llm_max_tokens, False
        )
    return AnthropicLLM(
        settings.llm_model,
        settings.llm_effort,
        settings.llm_max_tokens,
        settings.llm_refusal_fallback,
    )
