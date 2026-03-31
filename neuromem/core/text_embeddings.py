"""
Shared text embedding backends for NeuroMem.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


class TextEmbedder:
    """Minimal embedder interface."""

    def encode(self, text: str) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class LexicalHashingEmbedder(TextEmbedder):
    """
    Deterministic lexical embedding using token and bigram feature hashing.

    This is a significant improvement over hashing the whole string because
    texts with overlapping tokens map to overlapping feature dimensions.
    """

    dimension: int = 512
    use_bigrams: bool = True

    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            return vector

        features: List[Tuple[str, float]] = [(token, 1.0) for token in tokens]
        if self.use_bigrams and len(tokens) > 1:
            features.extend(
                (f"{left}__{right}", 0.75)
                for left, right in zip(tokens, tokens[1:])
            )

        for feature, weight in features:
            index = self._stable_hash(feature) % self.dimension
            vector[index] += weight

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    @staticmethod
    def _stable_hash(value: str) -> int:
        return int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16)


class TfidfEmbedder(TextEmbedder):
    """Corpus-fitted TF-IDF embedder used for rigorous benchmark workloads."""

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    @classmethod
    def fit(
        cls,
        texts: Sequence[str],
        *,
        max_features: int = 4096,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
    ) -> "TfidfEmbedder":
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "TfidfEmbedder requires scikit-learn. Install it to use the tfidf backend."
            ) from exc

        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z0-9_]+\b",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            norm="l2",
        )
        vectorizer.fit(list(texts))
        return cls(vectorizer)

    def encode(self, text: str) -> np.ndarray:
        vector = self.vectorizer.transform([text]).toarray()[0]
        return np.asarray(vector, dtype=np.float32)
