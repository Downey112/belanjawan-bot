"""
Ingestion pipeline for Belanjawan Bot.

Extracts text from the downloaded Budget 2026 PDFs (page by page, so we can
cite page numbers later), splits it into overlapping chunks, embeds each
chunk with a local (free, no API) sentence-transformers model, and stores
everything in a persistent local Chroma database.

Usage:
    pip install -r requirements.txt
    python ingest.py
"""

import pathlib

import chromadb
import pdfplumber
from sentence_transformers import SentenceTransformer

DATA_DIR = pathlib.Path(__file__).parent / "data"
CHROMA_DIR = pathlib.Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "belanjawan_2026"

CHUNK_SIZE = 800  # characters per chunk — small enough for precise retrieval,
CHUNK_OVERLAP = 150  # large enough to keep sentences mostly intact


def extract_pages(pdf_path: pathlib.Path) -> list[tuple[int, str]]:
    """Returns a list of (page_number, page_text) tuples."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
    return pages


def chunk_text(text: str) -> list[str]:
    """Simple sliding-window chunking by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def main() -> None:
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDFs found in {DATA_DIR} — run download_docs.py first.")

    print("Loading local embedding model (first run downloads ~90MB, cached after)...", flush=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Fresh collection each ingest run, so re-running doesn't duplicate chunks
    client.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [
        c.name for c in client.list_collections()
    ] else None
    collection = client.create_collection(COLLECTION_NAME)

    all_chunks, all_metadatas, all_ids = [], [], []

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name} ...", flush=True)
        pages = extract_pages(pdf_path)
        for page_num, page_text in pages:
            for chunk_idx, chunk in enumerate(chunk_text(page_text)):
                chunk_id = f"{pdf_path.stem}_p{page_num}_c{chunk_idx}"
                all_chunks.append(chunk)
                all_metadatas.append({"source": pdf_path.name, "page": page_num})
                all_ids.append(chunk_id)

        print(f"  {len(pages)} pages processed", flush=True)

    print(f"\nEmbedding {len(all_chunks)} chunks locally (no API calls)...", flush=True)
    embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=32).tolist()

    print("Writing to Chroma...", flush=True)
    # Chroma has a batch size limit — insert in batches to be safe
    BATCH = 500
    for i in range(0, len(all_chunks), BATCH):
        collection.add(
            documents=all_chunks[i : i + BATCH],
            embeddings=embeddings[i : i + BATCH],
            metadatas=all_metadatas[i : i + BATCH],
            ids=all_ids[i : i + BATCH],
        )
        print(f"  ...{min(i + BATCH, len(all_chunks))}/{len(all_chunks)} written", flush=True)

    print(f"\nDone. {len(all_chunks)} chunks indexed into '{COLLECTION_NAME}' at {CHROMA_DIR}", flush=True)


if __name__ == "__main__":
    main()
