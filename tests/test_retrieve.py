import numpy as np
import pytest

from belanjawan import retrieve
from belanjawan.retrieve import rrf_merge, search, tokenize


def test_rrf_scores_and_order():
    fused = rrf_merge([["A", "B"], ["B", "C"], ["B", "A"]], k=60)
    assert [cid for cid, _ in fused] == ["B", "A", "C"]
    scores = dict(fused)
    assert scores["B"] == pytest.approx(1 / 62 + 1 / 61 + 1 / 61)
    assert scores["C"] == pytest.approx(1 / 62)


def test_tokenizer_keeps_money_and_acronyms():
    tokens = tokenize("RM1,000 for B40 under SST, 2027.")
    assert {"rm1000", "b40", "sst", "2027"} <= set(tokens)
    assert tokenize("RM1000") == tokenize("rm1,000")


def test_dense_search_and_year_filter(toy_index, monkeypatch):
    monkeypatch.setattr(retrieve, "embed_query", lambda q: np.array([0, 0, 1, 0], dtype=np.float32))
    assert search(toy_index, "cash aid", "dense", k=1)[0].chunk.id == "c"
    hits = search(toy_index, "cash aid", "dense", k=4, year=2026)
    assert {h.chunk.year for h in hits} == {2026}


def test_bm25_finds_exact_amount(toy_index):
    hits = search(toy_index, "RM1000 relief", "bm25", k=2)
    assert hits[0].chunk.id == "a"
    assert all(h.score > 0 for h in hits)


def test_hybrid_combines_both(toy_index, monkeypatch):
    # dense prefers d, keywords prefer a: both must appear in the fused top 2
    monkeypatch.setattr(retrieve, "embed_query", lambda q: np.array([0, 0, 0, 1], dtype=np.float32))
    ids = {h.chunk.id for h in search(toy_index, "RM1000 relief", "hybrid", k=2)}
    assert ids == {"a", "d"}


def test_unknown_mode_is_rejected(toy_index):
    with pytest.raises(ValueError):
        search(toy_index, "x", "magic")
