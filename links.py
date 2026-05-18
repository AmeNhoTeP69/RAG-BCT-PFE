"""
links.py — Download all BCT PDFs from links.txt using a browser-like session.

Usage:
    python links.py links.txt
    python links.py links.txt bct_documents   (custom output folder)
"""

import os
import re
import sys
import time
import logging
from pathlib import Path
from urllib.parse import urlparse

import requests

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_URL   = "https://www.bct.gov.tn"
REFERER    = "https://www.bct.gov.tn/bct/siteprod/page.jsp?id=226"
DELAY      = 1.5   # seconds between downloads
MAX_RETRY  = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Full Chrome-like headers — makes the server treat us as a real browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/pdf,application/octet-stream,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         REFERER,
    "Connection":      "keep-alive",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def safe_filename(url: str) -> str:
    """Derive a clean filename from the URL's path component."""
    name = urlparse(url).path.split("/")[-1]
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    return name.strip(" .") or "file.pdf"


def is_valid_pdf(filepath: Path) -> bool:
    """Return True only if file exists, is > 5 KB, and starts with %PDF."""
    if not filepath.exists() or filepath.stat().st_size < 5000:
        return False
    with open(filepath, "rb") as f:
        return f.read(4) == b"%PDF"


def make_session() -> requests.Session:
    """Create an authenticated-looking session by visiting BCT first."""
    session = requests.Session()
    session.headers.update(HEADERS)
    log.info("Warming up session — visiting BCT homepage for cookies...")
    try:
        session.get(BASE_URL, timeout=20)
        time.sleep(1)
        session.get(REFERER, timeout=20)
        time.sleep(1)
        log.info("Session ready ✓")
    except Exception as e:
        log.warning(f"Warm-up partial failure (continuing anyway): {e}")
    return session


# ─── Core download ────────────────────────────────────────────────────────────

def download(session: requests.Session, url: str, out_dir: Path) -> str:
    """
    Download one PDF. Returns 'ok', 'skipped', or 'failed'.
    """
    filename = safe_filename(url)
    filepath = out_dir / filename

    if is_valid_pdf(filepath):
        log.info(f"  ↷ Already valid — skipping: {filename}")
        return "skipped"

    # Remove corrupted leftover if any
    if filepath.exists():
        filepath.unlink()

    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = session.get(url, timeout=60, stream=True)

            if resp.status_code == 503:
                wait = 10 * attempt
                log.warning(f"  ⚠ 503 on attempt {attempt} — waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            content = resp.content

            if not content.startswith(b"%PDF"):
                log.warning(f"  ⚠ Not a real PDF ({len(content)} bytes). Retrying...")
                time.sleep(5 * attempt)
                continue

            filepath.write_bytes(content)
            log.info(f"  ✓ {filename}  ({len(content):,} bytes)")
            return "ok"

        except requests.exceptions.Timeout:
            log.warning(f"  ⚠ Timeout on attempt {attempt}")
            time.sleep(5 * attempt)
        except Exception as e:
            log.error(f"  ✗ Error: {e}")
            time.sleep(3)

    log.error(f"  ✗ Gave up: {filename}")
    return "failed"


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python links.py links.txt [output_folder]")
        sys.exit(1)

    input_file  = Path(sys.argv[1])
    output_dir  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("bct_documents")
    output_dir.mkdir(exist_ok=True)

    # Read URLs, skip blanks and comments
    urls = [
        line.strip()
        for line in input_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not urls:
        log.error("No URLs found in the input file.")
        sys.exit(1)

    log.info(f"Found {len(urls)} URLs → saving to '{output_dir}/'")

    session = make_session()

    ok, skipped, failed = 0, 0, 0
    for i, url in enumerate(urls, start=1):
        log.info(f"[{i}/{len(urls)}] {url.split('/')[-1]}")
        result = download(session, url, output_dir)
        if result == "ok":       ok      += 1
        elif result == "skipped": skipped += 1
        else:                    failed  += 1
        time.sleep(DELAY)

    log.info("=" * 60)
    log.info(f"✓ Downloaded: {ok}  |  ↷ Skipped: {skipped}  |  ✗ Failed: {failed}")
    log.info(f"Total valid PDFs in '{output_dir}': "
             f"{sum(1 for f in output_dir.glob('*.pdf') if is_valid_pdf(f))}")


if __name__ == "__main__":
    main()