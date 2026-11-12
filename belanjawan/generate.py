"""Answer: send the retrieved excerpts to the LLM, then check what comes back."""

import os
import re

from pydantic import BaseModel, Field

from belanjawan.retrieve import Hit

DEFAULT_MODEL = "openai/gpt-oss-120b"  # on Groq's free tier
OUT_OF_SCOPE = "OUT_OF_SCOPE"
# Reranker score below which we refuse without calling the LLM. Calibrated on the golden set
# (see EVAL.md: unrelated questions score < 0.003, the weakest answerable one 0.008); only meaningful for hybrid_rerank, whose scores are 0..1.
MIN_RERANK_SCORE = 0.005
REFUSAL = "I can't answer that from the Budget documents I have."

SYSTEM_PROMPT = f"""You are Belanjawan Bot. You answer questions about Malaysia's Budget 2026 and \
Budget 2027 using ONLY the document excerpts provided.
- After every fact, cite its excerpt exactly as it is labelled, e.g. [budget2027_speech.pdf p.12].
- Do not use outside knowledge. Do not guess numbers.
- If the excerpts do not answer the question, reply with exactly: {OUT_OF_SCOPE}
- Keep answers short and factual. Keep Malaysian terms (e.g. SARA, STR, KWSP) as written."""

# Matches the file name followed by a page number, whatever the brackets or separators:
# [a.pdf p.12], 【a.pdf p.12】, (a.pdf, page 12), a.pdf p. 12-13 (takes the first page).
CITATION = re.compile(r"([\w.\-]+\.pdf)\W{0,3}(?:p\.?|pp\.?|page|pg\.?)\s*(\d+)", re.IGNORECASE)


class Citation(BaseModel):
    doc: str
    page: int


class Answer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    # Why it was refused: "no_hits", "low_score", "model_out_of_scope", "no_valid_citation", "empty_reply"
    reason: str | None = None


def build_context(hits: list[Hit]) -> str:
    """Label each excerpt with its document and page so the model can cite it."""
    return "\n\n---\n\n".join(f"[{h.chunk.doc} p.{h.chunk.page}]\n{h.chunk.text}" for h in hits)


def valid_citations(text: str, hits: list[Hit]) -> list[Citation]:
    """Citations in the model's text that point at an excerpt we actually gave it.

    Anything else is invented by the model and dropped.
    """
    retrieved = {(h.chunk.doc, h.chunk.page) for h in hits}
    found = []
    for doc, page in CITATION.findall(text):
        pair = (doc, int(page))
        if pair in retrieved and Citation(doc=doc, page=int(page)) not in found:
            found.append(Citation(doc=doc, page=int(page)))
    return found


def answer(question: str, hits: list[Hit], client=None, model: str = DEFAULT_MODEL,
           min_score: float | None = None) -> Answer:
    """Answer `question` from `hits`. Pass min_score to refuse when the best hit is too weak."""
    if not hits:
        return Answer(answer=REFUSAL, refused=True, reason="no_hits")
    if min_score is not None and max(h.score for h in hits) < min_score:
        return Answer(answer=REFUSAL, refused=True, reason="low_score")

    if client is None:
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n\n{build_context(hits)}\n\nQuestion: {question}"},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        return Answer(answer=REFUSAL, refused=True, reason="empty_reply")
    if text.startswith(OUT_OF_SCOPE) or text == OUT_OF_SCOPE:
        return Answer(answer=REFUSAL, refused=True, reason="model_out_of_scope")
    citations = valid_citations(text, hits)
    if not citations:  # an answer that cites nothing we retrieved is not grounded
        return Answer(answer=REFUSAL, refused=True, reason="no_valid_citation")
    return Answer(answer=text, citations=citations)
