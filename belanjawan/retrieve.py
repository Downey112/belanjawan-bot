"""Search: vector (dense), keyword (BM25), hybrid (RRF fusion of both) and hybrid + rerank."""

import re
from dataclasses import dataclass

import numpy as np

from belanjawan.corpus import Chunk
from belanjawan.index import Index, embed_query

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
POOL = 30  # candidates each retriever contributes before fusion / reranking
RRF_K = 60  # standard RRF constant: damps the advantage of being ranked 1st vs 2nd
RERANK_MODEL = "ms-marco-TinyBERT-L-2-v2"
_ranker = None


@dataclass
class Hit:
    """One search result. `score` means different things per mode (cosine, BM25, RRF or reranker)."""

    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    """Lowercase words for BM25. Joins digit groups so 'RM1,000' and 'RM1000' match."""
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    return re.findall(r"[a-z0-9]+", text)


def _allowed(index: Index, year: int | None) -> np.ndarray:
    """Boolean mask of chunks that pass the optional year filter."""
    if year is None:
        return np.ones(len(index.chunks), dtype=bool)
    return np.array([c.year == year for c in index.chunks])


def _top(scores: np.ndarray, mask: np.ndarray, k: int) -> list[int]:
    """Indices of the k highest scores among allowed chunks."""
    scores = np.where(mask, scores, -np.inf)
    order = np.argsort(-scores, kind="stable")[:k]
    return [int(i) for i in order if np.isfinite(scores[i])]


def dense_search(index: Index, question: str, k: int, year: int | None = None) -> list[Hit]:
    """Cosine similarity between the question vector and every chunk vector."""
    scores = index.embeddings @ embed_query(question)
    return [Hit(index.chunks[i], float(scores[i])) for i in _top(scores, _allowed(index, year), k)]


def bm25_search(index: Index, question: str, k: int, year: int | None = None) -> list[Hit]:
    """Keyword search: good at exact numbers, acronyms and names that vectors blur."""
    from rank_bm25 import BM25Okapi

    if index.bm25 is None:
        index.bm25 = BM25Okapi([tokenize(c.text) for c in index.chunks])
    tokens = tokenize(question)
    if not tokens:
        return []
    scores = np.asarray(index.bm25.get_scores(tokens))
    mask = _allowed(index, year) & (scores > 0)  # a chunk sharing no word with the question is not a hit
    return [Hit(index.chunks[i], float(scores[i])) for i in _top(scores, mask, k)]


def rrf_merge(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(id) = sum over lists of 1 / (k + rank), rank starting at 1.

    Uses only positions, so a cosine score and a BM25 score never need to be put on one scale.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def get_ranker():
    """Load the FlashRank cross-encoder once (about 4 MB, CPU)."""
    global _ranker
    if _ranker is None:
        from flashrank import Ranker

        _ranker = Ranker(model_name=RERANK_MODEL, cache_dir=".cache/flashrank")
    return _ranker


def rerank(question: str, chunks: list[Chunk], k: int) -> list[Hit]:
    """Re-score candidates by reading question and chunk together. Scores are 0..1."""
    from flashrank import RerankRequest

    if not chunks:
        return []
    by_id = {c.id: c for c in chunks}
    request = RerankRequest(query=question, passages=[{"id": c.id, "text": c.text} for c in chunks])
    results = get_ranker().rerank(request)
    return [Hit(by_id[r["id"]], float(r["score"])) for r in results[:k]]


def search(index: Index, question: str, mode: str = "hybrid_rerank", k: int = 5,
           year: int | None = None) -> list[Hit]:
    """Run one of the four search modes and return the top k hits."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not index.chunks or not question.strip():
        return []
    if mode == "dense":
        return dense_search(index, question, k, year)
    if mode == "bm25":
        return bm25_search(index, question, k, year)

    dense_ids = [h.chunk.id for h in dense_search(index, question, POOL, year)]
    bm25_ids = [h.chunk.id for h in bm25_search(index, question, POOL, year)]
    fused = rrf_merge([dense_ids, bm25_ids])
    by_id = {c.id: c for c in index.chunks}
    if mode == "hybrid":
        return [Hit(by_id[cid], score) for cid, score in fused[:k]]
    return rerank(question, [by_id[cid] for cid, _ in fused[:POOL]], k)
