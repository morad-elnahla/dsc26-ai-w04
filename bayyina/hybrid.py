"""Reciprocal-rank fusion of BM25 and an interchangeable dense embedder."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from bayyina.bm25 import BM25Index
from bayyina.types import Chunk, Embedder, Hit, Reranker


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id not in seen:
                scores[chunk_id] += 1.0 / (k + rank)
                seen.add(chunk_id)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, *,
                 k1: float = 1.5, b: float = 0.75,
                 reranker: Reranker | None = None):
        self.chunks = list(chunks)
        self._by_id = {c.chunk_id: c for c in self.chunks}
        self._embedder = embedder
        self._reranker = reranker
        self._bm25 = BM25Index(k1=k1, b=b)
        self._bm25.add(self.chunks)
        self._dense = (np.asarray(embedder.embed([c.text for c in self.chunks]))
                       if self.chunks else None)

    def search(self, query: str, top_k: int = 5,
               candidates: int = 50) -> list[Hit]:
        if not query.strip() or top_k <= 0 or candidates <= 0:
            return []

        sparse_ids = [chunk_id for chunk_id, _ in
                      self._bm25.search(query, top_k=candidates)]
        dense_ids: list[str] = []
        if self.chunks:
            query_vector = np.asarray(self._embedder.embed([query]))[0]
            similarities = self._dense @ query_vector
            dense_ids = [self.chunks[i].chunk_id for i in sorted(
                range(len(self.chunks)),
                key=lambda i: (-float(similarities[i]), self.chunks[i].chunk_id)
            )[:candidates]]

        fused = rrf([sparse_ids, dense_ids])[:candidates]
        hits = [Hit(chunk_id=chunk_id, doc_id=self._by_id[chunk_id].doc_id,
                    start=self._by_id[chunk_id].start,
                    end=self._by_id[chunk_id].end,
                    text=self._by_id[chunk_id].text,
                    score=score) for chunk_id, score in fused]

        if self._reranker is not None:
            scores = self._reranker(query, hits)
            hits = [Hit(chunk_id=h.chunk_id, doc_id=h.doc_id, start=h.start,
                        end=h.end, text=h.text, score=float(score))
                    for h, score in zip(hits, scores)]
            hits.sort(key=lambda h: (-h.score, h.chunk_id))
        return hits[:top_k]
