"""
Belanjawan Bot — ask questions about Malaysia's Budget 2026 and get answers
grounded in the actual government documents, with page citations.

Local dev:
    pip install -r requirements.txt
    python download_docs.py
    python ingest.py
    export GROQ_API_KEY="gsk_..."
    streamlit run app.py

On Streamlit Community Cloud: set GROQ_API_KEY in the app's Secrets instead
of an environment variable (Settings -> Secrets, as a TOML entry:
GROQ_API_KEY = "gsk_...").
"""

import os
import pathlib

import chromadb
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

CHROMA_DIR = pathlib.Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "belanjawan_2026"
TOP_K = 5
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are Belanjawan Bot, an assistant that answers questions about \
Malaysia's Budget 2026 (Belanjawan 2026) using ONLY the provided document excerpts. \
If the excerpts don't contain enough information to answer, say so directly — don't \
guess or use outside knowledge. Always mention which document/page an answer is \
based on when possible. Keep answers concise and factual."""


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_chroma_collection():
    if not CHROMA_DIR.exists():
        st.error("No Chroma database found — run `python ingest.py` first (locally, "
                 "before deploying) and commit the chroma_db/ folder, or wire up a "
                 "build step that runs ingestion before the app starts.")
        st.stop()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def get_groq_client() -> Groq:
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        st.error("GROQ_API_KEY not set — add it to Streamlit secrets or your local "
                 "environment. Get a free key at console.groq.com (no card required).")
        st.stop()
    return Groq(api_key=api_key)


def retrieve(query: str, model, collection, k: int = TOP_K):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    return list(zip(chunks, metadatas))


def build_context(retrieved: list[tuple[str, dict]]) -> str:
    parts = []
    for text, meta in retrieved:
        parts.append(f"[Source: {meta['source']}, page {meta['page']}]\n{text}")
    return "\n\n---\n\n".join(parts)


def main():
    st.set_page_config(page_title="Belanjawan Bot", page_icon="🇲🇾")
    st.title("🇲🇾 Belanjawan Bot")
    st.caption("Ask questions about Malaysia's Budget 2026 — answers are grounded in "
               "the official Ministry of Finance documents, with page citations.")

    embedding_model = load_embedding_model()
    collection = load_chroma_collection()
    groq_client = get_groq_client()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("e.g. What tax changes were announced for 2026?")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the Budget documents..."):
                retrieved = retrieve(question, embedding_model, collection)
                context = build_context(retrieved)

            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Document excerpts:\n\n{context}\n\n"
                                                 f"Question: {question}"},
                ],
            )
            answer = response.choices[0].message.content
            st.markdown(answer)

            with st.expander("Sources used"):
                seen = set()
                for _, meta in retrieved:
                    key = (meta["source"], meta["page"])
                    if key not in seen:
                        st.markdown(f"- {meta['source']}, page {meta['page']}")
                        seen.add(key)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
