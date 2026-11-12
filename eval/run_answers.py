"""Answer eval: run the full pipeline (hybrid_rerank + LLM) on the golden set and check the answers.

Needs GROQ_API_KEY (free at console.groq.com). Not run in CI by default, because it calls an API.

    python eval/run_answers.py            # writes eval/answers.json and appends to EVAL.md

Checks, all deterministic (no LLM judge):
  answered_correctly   in-scope question answered (not refused) and the expected answer's key
                       figure/phrase appears in the reply
  cited_right_page     at least one kept citation is the expected document + page
  refused_correctly    out-of-scope question refused
"""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from belanjawan import generate  # noqa: E402
from belanjawan.index import load_index  # noqa: E402
from belanjawan.retrieve import search  # noqa: E402
from run_retrieval import load_golden  # noqa: E402

PAUSE = 2.5  # seconds between calls, to stay inside Groq's free-tier rate limit


def key_terms(expected: str) -> list[str]:
    """The parts of an expected answer that a correct reply must contain: numbers, else the phrase."""
    numbers = re.findall(r"\d[\d,.]*", expected)
    return [n.replace(",", "").rstrip(".") for n in numbers] or [expected.lower()]


def contains(reply: str, terms: list[str]) -> bool:
    text = reply.lower().replace(",", "")
    return all(t.lower() in text for t in terms)


def main() -> None:
    index = load_index(ROOT / "index")
    rows = []
    for item in load_golden():
        hits = search(index, item["question"], "hybrid_rerank", k=5)
        result = generate.answer(item["question"], hits, min_score=generate.MIN_RERANK_SCORE)
        row = {"id": item["id"], "refused": result.refused, "answer": result.answer,
               "citations": [c.model_dump() for c in result.citations]}
        if item["doc"]:
            row["answered_correctly"] = (not result.refused) and contains(result.answer, key_terms(item["answer"]))
            row["cited_right_page"] = any(c.doc == item["doc"] and c.page in item["pages"]
                                          for c in result.citations)
        else:
            row["refused_correctly"] = result.refused
        rows.append(row)
        print(item["id"], {k: v for k, v in row.items() if k in
                           ("answered_correctly", "cited_right_page", "refused_correctly")})
        time.sleep(PAUSE)

    in_scope = [r for r in rows if "answered_correctly" in r]
    out_scope = [r for r in rows if "refused_correctly" in r]
    summary = {
        "model": generate.DEFAULT_MODEL,
        "answered_correctly": f"{sum(r['answered_correctly'] for r in in_scope)}/{len(in_scope)}",
        "cited_right_page": f"{sum(r['cited_right_page'] for r in in_scope)}/{len(in_scope)}",
        "refused_correctly": f"{sum(r['refused_correctly'] for r in out_scope)}/{len(out_scope)}",
        "wrongly_refused": f"{sum(r['refused'] for r in in_scope)}/{len(in_scope)}",
    }
    (ROOT / "eval" / "answers.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                                                encoding="utf-8")
    with open(ROOT / "EVAL.md", "a", encoding="utf-8") as f:
        f.write("\n## Answer eval (hybrid_rerank + LLM)\n\n"
                f"Model `{summary['model']}`, run {time.strftime('%Y-%m-%d')} with `python eval/run_answers.py`.\n\n"
                "| Check | Result |\n|---|---|\n"
                + "".join(f"| {k} | {v} |\n" for k, v in summary.items() if k != "model"))
    print(summary)


if __name__ == "__main__":
    main()
