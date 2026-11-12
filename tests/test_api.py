from functools import partial
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from belanjawan import api, generate
from belanjawan.retrieve import Hit


class FakeLLM:
    """Stands in for the Groq client and returns a fixed reply."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


@pytest.fixture
def client(toy_index, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    api.app.state.index = toy_index
    with TestClient(api.app) as c:
        yield c
    del api.app.state.index


def use(monkeypatch, hits, reply="unused"):
    """Make the API see these hits and this LLM reply."""
    llm = FakeLLM(reply)
    monkeypatch.setattr(api, "search", lambda *a, **k: hits)
    monkeypatch.setattr(generate, "answer", partial(generate.answer, client=llm))
    return llm


def test_health(client):
    body = client.get("/v1/health").json()
    assert body["status"] == "ok" and body["chunks"] == 4


def test_rejects_short_question_and_bad_year(client):
    assert client.post("/v1/search", json={"question": "hi"}).status_code == 422
    assert client.post("/v1/query", json={"question": "tax relief", "year": 2019}).status_code == 422


def test_query_needs_key(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    assert client.post("/v1/query", json={"question": "tax relief"}).status_code == 503


def test_weak_hits_refuse_without_calling_llm(client, toy_index, monkeypatch):
    llm = use(monkeypatch, [Hit(toy_index.chunks[0], 0.001)])
    body = client.post("/v1/query", json={"question": "Singapore budget?"}).json()
    assert body["refused"] is True and llm.calls == 0


def test_invented_citation_is_dropped(client, toy_index, monkeypatch):
    reply = "Relief is RM1,000 [budget_speech.pdf p.1]. Also free cars [made_up.pdf p.99]."
    use(monkeypatch, [Hit(toy_index.chunks[0], 0.9)], reply)
    body = client.post("/v1/query", json={"question": "What tax relief?"}).json()
    assert body["refused"] is False
    assert body["citations"] == [{"doc": "budget_speech.pdf", "page": 1}]


def test_answer_without_any_valid_citation_is_refused(client, toy_index, monkeypatch):
    use(monkeypatch, [Hit(toy_index.chunks[0], 0.9)], "Relief is RM5,000 [other.pdf p.7].")
    assert client.post("/v1/query", json={"question": "What tax relief?"}).json()["refused"] is True


def test_model_out_of_scope_token_refuses(client, toy_index, monkeypatch):
    use(monkeypatch, [Hit(toy_index.chunks[0], 0.9)], "OUT_OF_SCOPE")
    assert client.post("/v1/query", json={"question": "Best nasi lemak?"}).json()["refused"] is True
