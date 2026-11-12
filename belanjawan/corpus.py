"""Corpus: which Budget documents we use, how to download them, and how to cut them into chunks."""

import dataclasses
from pathlib import Path

import pdfplumber
import requests

# Official Ministry of Finance documents (belanjawan.mof.gov.my), keyed by local filename.
DOCUMENTS: dict[str, dict] = {
    "budget2027_speech.pdf": {
        "url": "https://belanjawan.mof.gov.my/pdf/belanjawan2027/ucapan/bs27.pdf",
        "year": 2027,
        "title": "Budget 2027 Speech",
    },
    "budget2027_economic_outlook.pdf": {
        "url": "https://belanjawan.mof.gov.my/pdf/belanjawan2027/economy/economic-2027.pdf",
        "year": 2027,
        "title": "Economic Outlook 2027",
    },
    "budget2027_economic_ch1.pdf": {
        "url": "https://belanjawan.mof.gov.my/pdf/belanjawan2027/economy/Chapter-1.pdf",
        "year": 2027,
        "title": "Economic Outlook 2027, Chapter 1",
    },
    "budget_speech.pdf": {
        "url": "https://belanjawan.mof.gov.my/pdf/belanjawan2026/ucapan/bs26.pdf",
        "year": 2026,
        "title": "Budget 2026 Speech",
    },
    "economic_outlook.pdf": {
        "url": "https://belanjawan.mof.gov.my/pdf/belanjawan2026/economy/economic-2026.pdf",
        "year": 2026,
        "title": "Economic Outlook 2026",
    },
    "economic_ch1_management.pdf": {
        "url": "https://belanjawan.mof.gov.my/pdf/belanjawan2026/economy/Chapter-1.pdf",
        "year": 2026,
        "title": "Economic Outlook 2026, Chapter 1",
    },
}

DATA_DIR = Path("data")
CHUNK_SIZE = 800  # characters: small enough for precise retrieval
CHUNK_OVERLAP = 150  # characters shared by neighbouring chunks, so a sentence cut at a border is not lost
BREAK_ZONE = 200  # look this far back from the window end for a clean place to cut
MIN_CHUNK = 40  # drop fragments shorter than this (page numbers, headers)


@dataclasses.dataclass
class Chunk:
    """A piece of one page, with where it came from."""

    id: str
    doc: str
    year: int
    page: int
    text: str


def download(force: bool = False, data_dir: Path = DATA_DIR) -> None:
    """Download any missing Budget PDFs (or all of them when force=True)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, meta in DOCUMENTS.items():
        out_path = data_dir / filename
        if out_path.exists() and out_path.stat().st_size > 0 and not force:
            continue
        print(f"Downloading {filename} ...", flush=True)
        resp = requests.get(meta["url"], timeout=120, headers={"User-Agent": "belanjawan-bot"})
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        print(f"  saved {len(resp.content):,} bytes", flush=True)


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return (page_number, text) for every page that has text. Page numbers start at 1."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((number, text))
    return pages


def _find_cut(text: str, start: int, end: int) -> int:
    """Pick where a chunk should end: the last newline, else the last sentence end, near `end`."""
    zone_start = max(start, end - BREAK_ZONE)
    zone = text[zone_start:end]
    newline = zone.rfind("\n")
    if newline != -1:
        return zone_start + newline + 1
    sentence = max(zone.rfind(". "), zone.rfind("? "), zone.rfind("! "))
    if sentence != -1:
        return zone_start + sentence + 2
    return end  # no clean break: hard cut


def chunk_page(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split one page into overlapping chunks that end at a newline or sentence where possible."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        cut = end if end == len(text) else _find_cut(text, start, end)
        piece = text[start:cut].strip()
        if len(piece) >= MIN_CHUNK:
            chunks.append(piece)
        if cut >= len(text):
            break
        start = max(start + 1, cut - overlap)  # always move forward
    return chunks


def build_chunks(data_dir: Path = DATA_DIR) -> list[Chunk]:
    """Read every document in DOCUMENTS and return all chunks with their source and page."""
    chunks = []
    for filename, meta in DOCUMENTS.items():
        pdf_path = data_dir / filename
        if not pdf_path.exists():
            raise FileNotFoundError(f"{pdf_path} is missing: run `python ingest.py` to download it")
        for page_number, text in extract_pages(pdf_path):
            for i, piece in enumerate(chunk_page(text)):
                chunks.append(Chunk(
                    id=f"{pdf_path.stem}_p{page_number}_c{i}",
                    doc=filename,
                    year=meta["year"],
                    page=page_number,
                    text=piece,
                ))
    return chunks
