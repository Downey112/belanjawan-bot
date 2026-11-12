# Design decisions (be ready to explain these)

Short answers to the questions an interviewer is most likely to ask. Rewrite them in your own words.

## 1. Chunking

**What:** each page is cut into ~800-character chunks with 150 characters of overlap. The cut is
moved back to the last newline (or sentence end) within 200 characters of the window edge.
Fragments under 40 characters are dropped. Every chunk keeps its document and page number.

**Why:** citations need page numbers, so chunks never cross pages. Cutting at a line end keeps a
sentence like "SARA … up to 150 ringgit per month" in one piece. The overlap means a fact sitting
on a border still appears whole in one of the two chunks.

**Known weakness:** two-column pages (Economic Outlook Chapter 1, pages 3–15) come out of
pdfplumber with the columns interleaved, so those chunks mix two articles. A layout-aware
extractor would fix it.

## 2. Hybrid search and Reciprocal Rank Fusion

**What:** vector search and BM25 each return their top 30. RRF gives every chunk
`sum(1 / (60 + rank))` over the two lists and sorts by that.

**Why RRF:** cosine similarity (0–1) and BM25 scores (unbounded) are on different scales.
RRF only looks at positions, so no score normalisation or weight tuning is needed. A chunk that both
retrievers rank highly wins; a chunk only one of them finds can still make it in.

**What the eval showed:** vector-only found the right page in the top 5 for 79% of questions, BM25
91%, hybrid 94%, hybrid + rerank 97% (see EVAL.md). Hybrid alone had a *lower* hit@1 than BM25
(0.47 vs 0.56) because fusion pulls in vector results that are near but not exact; the reranker
fixes the ordering.

## 3. Reranking and the refusal threshold

**What:** FlashRank (TinyBERT cross-encoder, 4 MB, CPU) reads question and chunk together and scores
0–1. If the best score is under `MIN_RERANK_SCORE = 0.005`, the API refuses without calling the LLM.

**How the number was chosen:** on the golden set, unrelated questions (recipe, stock price,
football, weather) scored below 0.003; the weakest answerable question scored 0.008. Near-domain
questions (Singapore's budget, a 2029 Malaysian budget) score above 0.94, so **no threshold can stop
them**: they are left to the LLM's OUT_OF_SCOPE rule and the citation check.

## 4. Treating the LLM's answer as untrusted

- The prompt labels each excerpt `[file.pdf p.N]` and asks for citations in that form.
- `valid_citations()` keeps only citations matching a retrieved (document, page); invented ones
  are dropped.
- An answer with no valid citation becomes a refusal. So does the token `OUT_OF_SCOPE`.
- `Answer.reason` records which rule fired, and the app shows it.

**A real bug this caught (issue #15):** after deploying, the live app refused "How much SARA aid do
STR recipients get in 2027?" although search ranked the right page first. The model
(gpt-oss-120b) writes citations as `【budget2027_speech.pdf p.25】`, not `[...]`, so the strict
pattern found no citation and the guardrail refused a correct answer. Fix: match the file name and
page whatever the brackets, test six real formats, and log the refusal reason. Lesson: a guardrail
that is too strict fails silently as "the bot is unhelpful", so record *why* it fired.

## 5. Evaluation

- **Retrieval eval** (`eval/run_retrieval.py`): no LLM, deterministic, runs in CI in under a
  minute. A result counts only if it is the expected document *and* page.
- **Answer eval** (`eval/run_answers.py`): full pipeline; checks the expected figure appears in
  the reply, the right page is cited, and off-topic questions are refused. Deterministic string
  checks, not an LLM judge, so reruns are comparable.
- **Regression gate:** CI fails if any retrieval metric falls more than 0.03 (one question) below
  `eval/baseline.json`.
- **Golden-set bias:** the first draft of questions mostly reuses the documents' own wording,
  which favours BM25. Add paraphrased and Malay questions before trusting small differences.

## 6. Why fastembed instead of sentence-transformers

Same model (all-MiniLM-L6-v2), but run through ONNX Runtime: no torch (~200 MB+), faster start on
Streamlit Cloud and in Docker. The vectors are normalised once at build time, so cosine similarity
is a single matrix–vector product over 3,531 × 384 floats (~25 ms per query).

## 7. In-memory index vs pgvector

For 3,531 chunks a numpy array is simpler and fast enough. `pgstore.py` shows the move to Postgres:
`vector(384)` with an HNSW index for cosine search, and Postgres full-text search in place of BM25.
Same results on the test fixtures (CI checks this against a real pgvector container).
