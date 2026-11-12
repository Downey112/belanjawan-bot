"""HTTP API.  Run:  uvicorn belanjawan.api:app --reload   then open http://127.0.0.1:8000/docs"""

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from belanjawan import generate
from belanjawan.index import MODEL_NAME, load_index
from belanjawan.retrieve import search

Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the index once when the server starts."""
    if not hasattr(app.state, "index"):  # tests put a small index here before starting
        app.state.index = load_index()
    yield


app = FastAPI(title="Belanjawan Bot API", version="2.0.0", lifespan=lifespan)


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Mode = "hybrid_rerank"
    k: int = Field(default=5, ge=1, le=20)
    year: Literal[2026, 2027] | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    year: Literal[2026, 2027] | None = None


class HitOut(BaseModel):
    doc: str
    page: int
    score: float
    snippet: str


class QueryResponse(generate.Answer):
    hits: list[HitOut]


def _hits_out(hits) -> list[HitOut]:
    return [HitOut(doc=h.chunk.doc, page=h.chunk.page, score=round(h.score, 4),
                   snippet=h.chunk.text[:300]) for h in hits]


@app.get("/v1/health")
def health(request: Request) -> dict:
    return {"status": "ok", "chunks": len(request.app.state.index.chunks), "embedding_model": MODEL_NAME}


@app.post("/v1/search")
def search_endpoint(body: SearchRequest, request: Request) -> list[HitOut]:
    """Retrieval only: no LLM call, no key needed."""
    return _hits_out(search(request.app.state.index, body.question, body.mode, body.k, body.year))


@app.post("/v1/query")
def query_endpoint(body: QueryRequest, request: Request) -> QueryResponse:
    """Retrieve with hybrid_rerank, then answer with citations (or refuse)."""
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not set on the server")
    hits = search(request.app.state.index, body.question, "hybrid_rerank", 5, body.year)
    result = generate.answer(body.question, hits, min_score=generate.MIN_RERANK_SCORE)
    return QueryResponse(**result.model_dump(), hits=_hits_out(hits))
