"""Optional Postgres + pgvector backend: the same dense and keyword search, done inside the database.

The in-memory index is fine for 3,500 chunks. This module shows how the same design moves to a
real database when the corpus grows: vectors in a `vector(384)` column with an HNSW index, and
keyword search with Postgres full-text search instead of BM25.

    export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
    python -m belanjawan.pgstore load      # copy index/ into Postgres
"""

import os
import sys

import numpy as np

from belanjawan.index import Index, embed_query, load_index
from belanjawan.retrieve import Hit

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id        text PRIMARY KEY,
    doc       text NOT NULL,
    year      int  NOT NULL,
    page      int  NOT NULL,
    text      text NOT NULL,
    embedding vector(384) NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
"""


def connect(url: str | None = None):
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(url or os.environ["DATABASE_URL"], autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def load(conn, index: Index) -> int:
    """Create the table if needed and upsert every chunk with its vector."""
    conn.execute(SCHEMA)
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (id, doc, year, page, text, embedding) VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding",
            [(c.id, c.doc, c.year, c.page, c.text, np.asarray(v)) for c, v in zip(index.chunks, index.embeddings)],
        )
    return len(index.chunks)


def _hits(rows) -> list[Hit]:
    from belanjawan.corpus import Chunk

    return [Hit(Chunk(id=r[0], doc=r[1], year=r[2], page=r[3], text=r[4]), float(r[5])) for r in rows]


def dense_search(conn, question: str, k: int, year: int | None = None, vector=None) -> list[Hit]:
    """Cosine similarity via pgvector's <=> (cosine distance) operator."""
    vector = embed_query(question) if vector is None else vector
    rows = conn.execute(
        "SELECT id, doc, year, page, text, 1 - (embedding <=> %s) AS score FROM chunks "
        "WHERE %s::int IS NULL OR year = %s ORDER BY embedding <=> %s LIMIT %s",
        (vector, year, year, vector, k),
    ).fetchall()
    return _hits(rows)


def keyword_search(conn, question: str, k: int, year: int | None = None) -> list[Hit]:
    """Postgres full-text search, OR-ing the question's words, ranked by ts_rank_cd."""
    rows = conn.execute(
        "WITH q AS (SELECT replace(plainto_tsquery('english', %s)::text, '&', '|')::tsquery AS query) "
        "SELECT id, doc, year, page, text, ts_rank_cd(tsv, q.query) AS score FROM chunks, q "
        "WHERE tsv @@ q.query AND (%s::int IS NULL OR year = %s) ORDER BY score DESC LIMIT %s",
        (question, year, year, k),
    ).fetchall()
    return _hits(rows)


if __name__ == "__main__":
    if sys.argv[1:] == ["load"]:
        print(f"Loaded {load(connect(), load_index())} chunks into Postgres")
    else:
        print(__doc__)
