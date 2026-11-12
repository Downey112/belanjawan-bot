"""Build the search index: download the Budget PDFs, chunk every page, embed the chunks.

Usage:
    pip install -r requirements.txt
    python ingest.py              # downloads missing PDFs into data/, writes index/
    python ingest.py --force      # re-download every PDF first
"""

import argparse

from belanjawan.corpus import DOCUMENTS, build_chunks, download
from belanjawan.index import INDEX_DIR, build_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download every PDF")
    args = parser.parse_args()

    download(force=args.force)
    chunks = build_chunks()
    for doc in DOCUMENTS:
        print(f"  {doc}: {sum(c.doc == doc for c in chunks)} chunks")
    print(f"Embedding {len(chunks)} chunks locally (no API calls)...", flush=True)
    build_index(chunks)
    print(f"Done. Index written to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
