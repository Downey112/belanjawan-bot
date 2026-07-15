"""
Downloads the real Budget 2026 documents from Malaysia's official Budget
Portal (belanjawan.mof.gov.my). Run this once before ingest.py.

Usage:
    pip install requests
    python download_docs.py
"""

import pathlib
import requests

BASE = "https://belanjawan.mof.gov.my/pdf/belanjawan2026"

# A focused, manageable set for a few-day project — expand later if you want
# a bigger corpus (e.g. add BNM Monetary Policy Statements as a v2).
DOCUMENTS = {
    "budget_speech.pdf": f"{BASE}/ucapan/bs26.pdf",
    "economic_outlook.pdf": f"{BASE}/economy/economic-2026.pdf",
    "economic_ch1_management.pdf": f"{BASE}/economy/Chapter-1.pdf",
}

OUT_DIR = pathlib.Path(__file__).parent / "data"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for filename, url in DOCUMENTS.items():
        out_path = OUT_DIR / filename
        print(f"Downloading {filename} ...", flush=True)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            print(f"  saved {len(resp.content):,} bytes to {out_path}", flush=True)
        except requests.RequestException as e:
            print(f"  FAILED: {e} — the Budget Portal's exact file paths can shift "
                  f"between years; check https://belanjawan.mof.gov.my/en/speech "
                  f"for the current direct link if this 404s.", flush=True)

    print("\nDone. Check data/ for downloaded PDFs before running ingest.py.", flush=True)


if __name__ == "__main__":
    main()
