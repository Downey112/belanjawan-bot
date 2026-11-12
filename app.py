"""Belanjawan Bot: ask about Malaysia's Budget 2026 and 2027, get answers with page citations.

Local dev:
    pip install -r requirements.txt
    python ingest.py                 # only needed if index/ is missing
    export GROQ_API_KEY="gsk_..."
    streamlit run app.py

On Streamlit Community Cloud, set GROQ_API_KEY in the app's Secrets.
"""

import os

import streamlit as st
from groq import Groq

from belanjawan import generate
from belanjawan.index import load_index
from belanjawan.retrieve import MODES, search


@st.cache_resource
def get_index():
    return load_index()


def get_client() -> Groq:
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        st.error("GROQ_API_KEY not set. Add it to Streamlit secrets or your environment "
                 "(free key at console.groq.com).")
        st.stop()
    return Groq(api_key=api_key)


def main():
    st.set_page_config(page_title="Belanjawan Bot", page_icon="🇲🇾")
    st.title("🇲🇾 Belanjawan Bot")
    st.caption("Answers about Malaysia's Budget 2026 and 2027, grounded in the official Ministry "
               "of Finance documents, with page citations.")

    with st.sidebar:
        year = st.selectbox("Budget year", ["Both", 2027, 2026])
        mode = st.selectbox("Search mode", MODES, index=MODES.index("hybrid_rerank"),
                            help="hybrid_rerank = keyword + vector search, fused, then reranked. "
                                 "See EVAL.md for how each mode scores.")

    index, client = get_index(), get_client()
    st.session_state.setdefault("messages", [])
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    question = st.chat_input("e.g. How much is the SARA cash aid in 2027?")
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the Budget documents..."):
            hits = search(index, question, mode, k=5, year=None if year == "Both" else year)
            min_score = generate.MIN_RERANK_SCORE if mode == "hybrid_rerank" else None
            result = generate.answer(question, hits, client=client, min_score=min_score)
        st.markdown(result.answer)
        if result.refused:
            st.caption(f"Refused: {result.reason}")
        with st.expander("Sources"):
            for c in result.citations:
                st.markdown(f"- {c.doc}, page {c.page}")
            if not result.citations:
                st.markdown("_No source cited._")
    st.session_state.messages.append({"role": "assistant", "content": result.answer})


if __name__ == "__main__":
    main()
