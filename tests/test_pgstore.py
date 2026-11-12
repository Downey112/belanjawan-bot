"""Runs only when DATABASE_URL points at a Postgres with pgvector (CI starts one as a service)."""

import os

import numpy as np
import pytest

from belanjawan import retrieve

pytestmark = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture
def conn(toy_index):
    from belanjawan import pgstore

    # the toy index has 4-d vectors; pad them to 384-d so they fit the real column type
    padded = np.zeros((4, 384), dtype=np.float32)
    padded[:, :4] = toy_index.embeddings
    toy_index.embeddings = padded
    c = pgstore.connect()
    c.execute("DROP TABLE IF EXISTS chunks")
    pgstore.load(c, toy_index)
    yield c
    c.execute("DROP TABLE IF EXISTS chunks")
    c.close()


def test_dense_matches_in_memory_search(conn, toy_index, monkeypatch):
    from belanjawan import pgstore

    query = np.zeros(384, dtype=np.float32)
    query[2] = 1.0
    monkeypatch.setattr(retrieve, "embed_query", lambda q: query)
    expected = [h.chunk.id for h in retrieve.search(toy_index, "cash aid", "dense", k=2)]
    got = [h.chunk.id for h in pgstore.dense_search(conn, "cash aid", k=2, vector=query)]
    assert got[0] == expected[0] == "c"


def test_keyword_search_and_year_filter(conn):
    from belanjawan import pgstore

    assert pgstore.keyword_search(conn, "hospital clinic", k=1)[0].chunk.id == "d"
    assert all(h.chunk.year == 2026 for h in pgstore.keyword_search(conn, "relief vehicle", k=4, year=2026))
