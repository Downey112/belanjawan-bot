import pytest

from belanjawan.generate import valid_citations
from belanjawan.retrieve import Hit


@pytest.fixture
def hits(toy_index):
    return [Hit(toy_index.chunks[2], 0.9)]  # budget2027_speech.pdf page 3


@pytest.mark.parametrize("reply", [
    "SARA is RM100 [budget2027_speech.pdf p.3].",
    "SARA is RM100 【budget2027_speech.pdf p.3】.",
    "SARA is RM100 (budget2027_speech.pdf, page 3).",
    "SARA is RM100 [budget2027_speech.pdf p. 3-4].",
    "SARA is RM100 [budget2027_speech.pdf, p.3]",
    "SARA is RM100 — budget2027_speech.pdf pg 3",
])
def test_citation_formats_the_model_actually_uses(reply, hits):
    assert [(c.doc, c.page) for c in valid_citations(reply, hits)] == [("budget2027_speech.pdf", 3)]


def test_citation_to_a_page_we_did_not_retrieve_is_dropped(hits):
    assert valid_citations("[budget2027_speech.pdf p.30] and [budget_speech.pdf p.3]", hits) == []
