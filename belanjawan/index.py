"""Index: embed every chunk once, save to disk, and load it back for searching."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from belanjawan.corpus import Chunk

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, runs on CPU via ONNX (fastembed)
INDEX_DIR = Path("index")
_model = None


def get_model():
    """Load the embedding model once and reuse it (loading takes a few seconds)."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding  # imported here so tests without the model stay fast

        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, so a dot product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-12)).astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed one question into a unit-length vector."""
    vector = next(iter(get_model().query_embed(text)))
    return _normalise(np.asarray(vector, dtype=np.float32))


@dataclass
class Index:
    """Everything search needs: the chunks and one embedding row per chunk."""

    chunks: list[Chunk]
    embeddings: np.ndarray
    bm25: Any = field(default=None, repr=False)  # built on first keyword search


def build_index(chunks: list[Chunk], out_dir: Path = INDEX_DIR) -> None:
    """Embed all chunks and write chunks.jsonl, embeddings.npy and meta.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    vectors = np.array(list(get_model().embed([c.text for c in chunks], batch_size=64)))
    embeddings = _normalise(vectors)
    np.save(out_dir / "embeddings.npy", embeddings)

    meta = {
        "model": MODEL_NAME,
        "chunks": len(chunks),
        "dimension": int(embeddings.shape[1]),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_index(out_dir: Path = INDEX_DIR) -> Index:
    """Load a saved index. Raises FileNotFoundError if ingest.py has not been run."""
    out_dir = Path(out_dir)
    with open(out_dir / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
    embeddings = np.load(out_dir / "embeddings.npy")
    if len(chunks) != len(embeddings):
        raise ValueError(f"index is inconsistent: {len(chunks)} chunks vs {len(embeddings)} vectors")
    return Index(chunks=chunks, embeddings=embeddings)
