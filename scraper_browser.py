"""
scraper_browser.py — BCT Document Scraper using a Real Browser (Playwright)
─────────────────────────────────────────────────────────────────────────────
Uses a real headless Chromium browser to download PDFs from the BCT website.
This bypasses all server-side anti-scraping protections because it IS a real
browser with real cookies, JS execution, and fingerprinting.

Strategy: navigate to each PDF URL with the browser, intercept the response,
and save the binary content directly. No download dialog needed.

Usage:
    python scraper_browser.py
"""

import os
import re
import time
import logging
from pathlib import Path
from playwright.sync_api import sync_playwright, Route, Request

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_URL = "https://www.bct.gov.tn"
PAGE_URL = "https://www.bct.gov.tn/bct/siteprod/page.jsp?id=226"
SAVE_DIR = Path("bct_documents")
SAVE_DIR.mkdir(exist_ok=True)

DELAY = 2.0  # seconds between downloads (polite)
TIMEOUT = 60000  # ms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def clean_filename(text: str) -> str:
    text = text.strip().replace("\n", " ")
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    return text[:150]


def is_valid_pdf(filepath: Path) -> bool:
    """Return True if file exists and starts with PDF magic bytes."""
    if not filepath.exists() or filepath.stat().st_size < 5000:
        return False
    with open(filepath, "rb") as f:
        return f.read(4) == b"%PDF"


# ─── Main ─────────────────────────────────────────────────────────────────────

def scrape():
    with sync_playwright() as p:
        log.info("Launching headless Chromium browser...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            locale="fr-FR",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        # ── Step 1: Warm up session with real cookies ──────────────────────
        log.info("Visiting BCT homepage to warm up session...")
        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=TIMEOUT)
        time.sleep(1.5)

        # ── Step 2: Navigate to circulars listing page ─────────────────────
        log.info("Loading circulars listing page...")
        page.goto(PAGE_URL, wait_until="domcontentloaded", timeout=TIMEOUT)
        time.sleep(1.5)

        # ── Step 3: Collect all PDF links from the page ────────────────────
        links = page.evaluate("""() => {
            return Array.from(document.querySelectorAll('a[href]'))
                .filter(a => a.href.toLowerCase().includes('.pdf'))
                .map(a => ({ href: a.href, text: a.innerText.trim() }));
        }""")

        # Deduplicate by URL
        seen: set[str] = set()
        unique_links = []
        for link in links:
            if link["href"] not in seen:
                seen.add(link["href"])
                unique_links.append(link)

        log.info(f"Found {len(unique_links)} unique PDF links")

        # ── Step 4: Download each PDF by navigating to it ─────────────────
        success, skipped, failed = 0, 0, 0

        for i, link in enumerate(unique_links, start=1):
            url  = link["href"]
            name = clean_filename(link["text"]) or url.split("/")[-1].replace(".pdf", "")
            filename = name if name.endswith(".pdf") else name + ".pdf"
            filepath = SAVE_DIR / filename

            log.info(f"[{i}/{len(unique_links)}] {filename[:70]}")

            # Skip if already valid
            if is_valid_pdf(filepath):
                log.info("  → Already exists (valid), skipping.")
                skipped += 1
                continue

            # Remove any corrupted leftover
            if filepath.exists():
                filepath.unlink()

            try:
                # Use the API to fetch the PDF with all browser cookies intact
                pdf_page = context.new_page()
                response = pdf_page.goto(url, wait_until="commit", timeout=TIMEOUT)

                if response is None:
                    log.error("  ✗ No response received")
                    pdf_page.close()
                    failed += 1
                    continue

                if response.status == 503:
                    log.warning(f"  ✗ Server returned 503 — skipping for now")
                    pdf_page.close()
                    failed += 1
                    time.sleep(5)
                    continue

                content = response.body()
                pdf_page.close()

                if not content.startswith(b"%PDF"):
                    log.warning(f"  ✗ Not a valid PDF ({len(content)} bytes), got: {content[:80]}")
                    failed += 1
                    continue

                filepath.write_bytes(content)
                log.info(f"  ✓ Saved {len(content):,} bytes")
                success += 1

            except Exception as e:
                log.error(f"  ✗ Error: {e}")
                if filepath.exists():
                    filepath.unlink()
                failed += 1

            time.sleep(DELAY)

        browser.close()

        log.info("=" * 60)
        log.info(f"✓ Downloaded: {success} | ↷ Skipped: {skipped} | ✗ Failed: {failed}")
        log.info(f"Total valid PDFs in folder: {len([f for f in SAVE_DIR.glob('*.pdf') if is_valid_pdf(f)])}")


if __name__ == "__main__":
    scrape()
