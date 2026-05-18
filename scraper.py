"""
scraper.py — BCT Document Scraper with Browser Session
─────────────────────────────────────────────────────────
The BCT website blocks bare requests.get() calls with 503 errors.
Fix: use a persistent requests.Session that:
  1. First visits the main site to collect real cookies
  2. Sends full browser-like headers (User-Agent, Accept, Referer)
  3. Retries with exponential backoff on failure
  4. Detects and deletes any corrupted 503 "fake" PDFs
"""

import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import time
import re
import logging

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_URL = "https://www.bct.gov.tn"
PAGE_URL = "https://www.bct.gov.tn/bct/siteprod/page.jsp?id=226"

SAVE_DIR = "bct_documents"
os.makedirs(SAVE_DIR, exist_ok=True)

DELAY_BETWEEN_DOWNLOADS = 2   # seconds between each file (be polite)
MAX_RETRIES = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─── Browser-like headers ─────────────────────────────────────────────────────

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def make_session() -> requests.Session:
    """
    Create a requests Session that looks like a real browser.

    Warm-up strategy:
      - Visit the BCT homepage first → server sets session cookies
      - Visit the circulars page → picks up any page-specific cookies
      - All subsequent PDF downloads carry these cookies automatically
    """
    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)

    log.info("Warming up session — visiting BCT homepage...")
    try:
        session.get(BASE_URL, timeout=30)
        time.sleep(1)
        session.get(PAGE_URL, timeout=30)
        time.sleep(1)
        log.info("Session ready ✓")
    except Exception as e:
        log.warning(f"Session warm-up partial failure (continuing anyway): {e}")

    return session


# ─── Helpers ──────────────────────────────────────────────────────────────────

def clean_filename(text: str) -> str:
    """Sanitize text for a safe Windows filename."""
    text = text.strip().replace("\n", " ")
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    return text[:150]


def is_corrupted(filepath: str) -> bool:
    """Return True if the file is a saved 503 error page, not a real PDF."""
    if not os.path.exists(filepath):
        return False
    size = os.path.getsize(filepath)
    if size > 5000:
        return False
    with open(filepath, "rb") as f:
        header = f.read(512)
    # Real PDFs start with %PDF; error pages contain HTML keywords
    return b"%PDF" not in header


# ─── Core Logic ───────────────────────────────────────────────────────────────

def get_all_documents(session: requests.Session) -> list[tuple[str, str]]:
    """
    Scrape the BCT circulars page and return a deduplicated list of
    (name, url) tuples for every PDF link found.
    """
    log.info(f"Fetching document list from: {PAGE_URL}")
    try:
        resp = session.get(PAGE_URL, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        docs = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ".pdf" not in href.lower():
                continue
            full_url = urljoin(BASE_URL, href)
            name = clean_filename(a.get_text(strip=True)) or full_url.split("/")[-1]
            docs.append((name, full_url))

        # Deduplicate by URL while preserving order
        seen: set[str] = set()
        unique = [(n, u) for n, u in docs if not (u in seen or seen.add(u))]
        log.info(f"Found {len(unique)} unique PDF links")
        return unique

    except Exception as e:
        log.error(f"Failed to fetch document list: {e}")
        return []


def download_file(session: requests.Session, name: str, url: str) -> bool:
    """
    Download a single PDF file using the browser session.

    Returns True on success, False if all retries are exhausted.
    """
    filename = name if name.endswith(".pdf") else name + ".pdf"
    filepath = os.path.join(SAVE_DIR, filename)

    # Skip if already downloaded and valid
    if os.path.exists(filepath):
        if is_corrupted(filepath):
            log.warning(f"Removing corrupted cached file: {filename}")
            os.remove(filepath)
        else:
            log.info(f"Already exists (valid): {filename}")
            return True

    # Add a Referer header so the server thinks we clicked a link on the page
    headers = {"Referer": PAGE_URL}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"  ↓  {filename[:80]}  (attempt {attempt}/{MAX_RETRIES})")
            resp = session.get(url, headers=headers, timeout=90, stream=True)

            if resp.status_code == 503:
                wait = 10 * attempt
                log.warning(f"    503 received — waiting {wait}s before retry...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            content = resp.content

            # Sanity check: must start with %PDF
            if not content.startswith(b"%PDF"):
                log.warning(f"    Content is not a real PDF (got {len(content)} bytes). Retrying...")
                time.sleep(8 * attempt)
                continue

            with open(filepath, "wb") as f:
                f.write(content)

            log.info(f"  ✓  Saved {len(content):,} bytes → {filename[:70]}")
            return True

        except requests.exceptions.Timeout:
            log.warning(f"    Timeout on attempt {attempt}. Retrying...")
            time.sleep(5 * attempt)
        except Exception as e:
            log.error(f"    Error: {e}")
            time.sleep(5 * attempt)

    log.error(f"  ✗  Gave up after {MAX_RETRIES} attempts: {filename}")
    return False


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    session = make_session()
    docs = get_all_documents(session)

    if not docs:
        log.error("No documents found. Exiting.")
        return

    success, failed = 0, 0
    for i, (name, url) in enumerate(docs, start=1):
        log.info(f"[{i}/{len(docs)}] Processing: {name[:60]}")
        ok = download_file(session, name, url)
        if ok:
            success += 1
        else:
            failed += 1
        time.sleep(DELAY_BETWEEN_DOWNLOADS)

    log.info(f"\n{'='*60}")
    log.info(f"Done! Success: {success} | Failed: {failed} | Total: {len(docs)}")


if __name__ == "__main__":
    main()