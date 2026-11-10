"""Embedding backends: local FastEmbed model, plus a hashing embedder for offline use."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from .config import Settings

_TOKEN = re.compile(r"[a-z0-9]+")


STOPWORDS = frozenset(
    "a an and are as at be by do does for from how in is it of on or that the to what when "
    "which who why with can i my your you our we if".split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> set[str]:
    """Lower-cased tokens minus stopwords, with a trailing plural 's' removed."""
    return {
        t[:-1] if len(t) > 3 and t.endswith("s") else t
        for t in tokenize(text)
        if t not in STOPWORDS
    }


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Feature-hashing bag of content words, L2-normalised.

    Deterministic and dependency-free. Good enough for tests and offline smoke
    runs; use FastEmbed for real retrieval quality.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _index(self, feature: str) -> tuple[int, float]:
        digest = hashlib.md5(feature.encode()).digest()
        bucket = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return bucket, sign

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            # Binary term presence over content words: long chunks are not
            # rewarded for repeating a term, and stopwords carry no weight.
            for feat in sorted(content_tokens(text)):
                bucket, sign = self._index(feat)
                vec[bucket] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class FastEmbedEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)
        self.name = model_name
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hash":
        return HashEmbedder()
    return FastEmbedEmbedder(settings.embedding_model)
