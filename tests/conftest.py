import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from belanjawan.corpus import Chunk  # noqa: E402
from belanjawan.index import Index  # noqa: E402


@pytest.fixture
def toy_index() -> Index:
    """Four chunks with hand-made 4-d unit vectors, so dense scores are predictable."""
    chunks = [
        Chunk("a", "budget_speech.pdf", 2026, 1, "Income tax relief of RM1,000 for B40 households"),
        Chunk("b", "budget_speech.pdf", 2026, 2, "Electric vehicle charging stations nationwide"),
        Chunk("c", "budget2027_speech.pdf", 2027, 3, "SARA cash aid raised to RM100 a month"),
        Chunk("d", "budget2027_speech.pdf", 2027, 4, "Hospital and clinic upgrades for rural areas"),
    ]
    return Index(chunks=chunks, embeddings=np.eye(4, dtype=np.float32))
