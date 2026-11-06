"""Minimal BM25 search implementation without a third-party search package."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import log

from bayyina.normalize import tokenize
from bayyina.types import Chunk


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._frequencies: dict[str, Counter[str]] = {}
        self._lengths: dict[str, int] = {}
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._size = 0
        self._avgdl = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        """Index a collection (the task calls this once per index)."""
        self._frequencies.clear()
        self._lengths.clear()
        self._postings.clear()
        for chunk in chunks:
            tf = Counter(tokenize(chunk.text))
            self._frequencies[chunk.chunk_id] = tf
            self._lengths[chunk.chunk_id] = sum(tf.values())
            for term, freq in tf.items():
                self._postings[term][chunk.chunk_id] = freq
        self._size = len(self._frequencies)
        self._avgdl = (sum(self._lengths.values()) / self._size
                       if self._size else 0.0)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._size or top_k <= 0:
            return []
        terms = set(tokenize(query))
        if not terms:
            return []
        scores: dict[str, float] = defaultdict(float)
        for term in sorted(terms):
            postings = self._postings.get(term, {})
            if not postings:
                continue
            df = len(postings)
            idf = log(1.0 + (self._size - df + 0.5) / (df + 0.5))
            for chunk_id, freq in postings.items():
                dl = self._lengths[chunk_id]
                length_factor = dl / self._avgdl if self._avgdl else 0.0
                denominator = freq + self.k1 * (1.0 - self.b + self.b * length_factor)
                scores[chunk_id] += idf * freq * (self.k1 + 1.0) / denominator
        ordered = sorted(((id_, score) for id_, score in scores.items() if score > 0),
                         key=lambda item: (-item[1], item[0]))
        return ordered[:top_k]
