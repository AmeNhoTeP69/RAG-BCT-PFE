"""
step1_extract.py
────────────────
STEP 1 — PDF Text Extraction for BCT (Banque Centrale de Tunisie) documents.

Design decisions:
  - PyMuPDF (fitz) is used over pdfplumber/pypdf because it:
      * Handles embedded fonts and ligatures better (critical for French text)
      * Preserves reading order more reliably in multi-column layouts
      * Is significantly faster on large batches
  - We extract text block-by-block (not page-by-page) to preserve logical
    paragraph boundaries — this pays dividends in the chunking step.
  - Output is structured JSON, not raw .txt, so all downstream steps can
    access metadata without re-parsing filenames.
  - The "mentions légales" file is filtered out automatically by content heuristic
    rather than hardcoded filename — more robust.
"""

import re
import json
import logging
from pathlib import Path
from datetime import datetime

import fitz  # PyMuPDF — install: pip install pymupdf
import pytesseract
from langdetect import detect, DetectorFactory
from pdf2image import convert_from_path

# Ensure reproducible language detection
DetectorFactory.seed = 0

# ─── Configuration ────────────────────────────────────────────────────────────

INPUT_DIR  = Path("bct_documents")        # folder containing raw PDFs
OUTPUT_DIR = Path("data/extracted")       # where to write extracted JSON
OUTPUT_FILE = OUTPUT_DIR / "documents.json"

# Minimum characters a document must contain to be considered valid content.
# Filters out scanned-only PDFs, covers, and the "mentions légales" file.
MIN_CONTENT_LENGTH = 200

# BCT document types — used to classify each PDF from its filename/content
DOC_TYPES = {
    "circulaire": ["circulaire", "circ"],
    "note":       ["note", "note aux", "note de", "note circulaire"],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Core extraction helpers ───────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> dict:
    """
    Extract structured text from a single PDF using PyMuPDF.

    Returns a dict with raw text, page count, and per-page breakdown.
    Returns None if the PDF has no extractable text (scanned image-only).
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    full_text_parts = []
    extraction_method = "fitz"

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks", sort=True)
        page_text_parts = []

        for block in blocks:
            if block[6] != 0:      # skip image blocks
                continue
            raw = block[4]
            cleaned = clean_block_text(raw)
            if cleaned:
                page_text_parts.append(cleaned)

        page_text = "\n".join(page_text_parts)
        pages.append({"page": page_num, "text": page_text})
        full_text_parts.append(page_text)

    full_text = "\n\n".join(full_text_parts).strip()
    
    # ── OCR Fallback ──
    if len(full_text) < MIN_CONTENT_LENGTH:
        log.info(f"  → Text too short ({len(full_text)} chars), attempting OCR...")
        try:
            # pdf2image converts PDF pages to PIL images
            images = convert_from_path(str(pdf_path))
            ocr_text_parts = []
            for i, image in enumerate(images):
                # We can specify 'fra+ara' for bilingual BCT docs
                page_ocr = pytesseract.image_to_string(image, lang='fra+ara')
                ocr_text_parts.append(page_ocr)
                if i < len(pages):
                    pages[i]["text"] = page_ocr  # update per-page text
                else:
                    pages.append({"page": i+1, "text": page_ocr})
            
            full_text = "\n\n".join(ocr_text_parts).strip()
            extraction_method = "tesseract_ocr"
            log.info(f"  → OCR successful: {len(full_text)} characters extracted.")
        except Exception as e:
            log.error(f"  ! OCR failed for {pdf_path.name}: {e}")
            # If OCR fails or Tesseract isn't installed, we keep the short text (likely empty)

    doc.close()

    if len(full_text) < MIN_CONTENT_LENGTH:
        return None  # still too short

    return {
        "full_text": full_text,
        "page_count": len(pages),
        "pages": pages,
        "extraction_method": extraction_method
    }


def clean_block_text(raw: str) -> str:
    """
    Minimal cleaning of a raw text block from PyMuPDF.

    We intentionally keep this light here — heavy normalization is done
    in step2_chunk.py. Here we only fix extraction artefacts:
      - PyMuPDF sometimes inserts stray hyphens at line breaks (soft-hyphens)
      - Occasional null bytes or control characters from encoding issues
      - Excessive internal whitespace within a block
    """
    # Remove soft hyphens (U+00AD) used for line-break hyphenation
    text = raw.replace("\u00ad", "")

    # Remove null bytes and other non-printable ASCII control chars
    # (keep \n and \t which carry structural meaning)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse multiple spaces within a line (but preserve newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # Remove lines that are pure whitespace
    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip()]

    return "\n".join(lines).strip()


# ─── Metadata inference ────────────────────────────────────────────────────────

def infer_doc_type(filename: str, text_preview: str) -> str:
    """
    Classify a document as 'circulaire', 'note', or 'unknown'.

    Strategy:
      1. Check filename tokens (extremely reliable for BCT files)
      2. Check first few meaningful lines of text for formal headers
    """
    fn = filename.lower()
    if any(kw in fn for kw in ["circ", "circulaire"]):
        return "circulaire"
    if any(kw in fn for kw in ["note", "nb"]):
        return "note"

    # Fallback to text preview
    preview = text_preview.lower()
    if "circulaire" in preview[:300]:
        return "circulaire"
    if "note" in preview[:300]:
        return "note"

    return "unknown"


def infer_year(filename: str, text_preview: str) -> str | None:
    """
    Extract the most likely publication year from filename or document header.

    Strategy:
      1. Filename: BCT filenames usually have _YYYY_ or CirYYYY format.
      2. Header: Search for "Tunis, le DD/MM/YYYY" or "تونس في YYYY/MM/DD"
      3. Fallback: Most frequent 4-digit year in the first 1000 chars.
    """
    # Improved pattern for BCT years (handles underscores, dashes, and parentheses)
    # Looks for 4 digits 1980-2029 preceded and followed by non-digits or boundaries
    year_pattern = r"(?:^|[\s_\-\(\)])(?P<year>19[89]\d|20[0-2]\d)(?:$|[\s_\-\(\)\.])"

    # 1. High Priority: Filename (Cleanest source of truth)
    fn_match = re.search(year_pattern, filename)
    if fn_match:
        return fn_match.group("year")

    # 2. Medium Priority: Formal Date Headers (Tunis, le 24 MAI 2017)
    # This regex looks for French months or digits followed by a 4-digit year
    header_pattern = r"(?i)(?:le|في)\s+\d{1,2}\s+(?:[a-zéû\u0621-\u064A\s]+)?\s+(?P<year>20[0-2]\d|199\d)"
    header_match = re.search(header_pattern, text_preview[:500])
    if header_match:
        return header_match.group("year")

    # 3. Low Priority: Most frequent year in text
    txt_matches = re.findall(year_pattern, text_preview[:1000])
    if txt_matches:
        # Avoid picking up "1996" or "1981" if other recent years exist
        # (Since documents often mention old laws)
        recent_years = [y for y in txt_matches if int(y) >= 2010]
        if recent_years:
            return max(set(recent_years), key=recent_years.count)
        return max(set(txt_matches), key=txt_matches.count)

    return None


def infer_title(filename: str, text_preview: str) -> str:
    """
    Derive a human-readable title from the filename or first meaningful line.

    BCT filenames are already fairly descriptive; we clean them up.
    Fallback: first non-empty line of the document text.
    """
    # Try filename first: strip extension, replace separators, title-case
    stem = Path(filename).stem
    title_from_file = re.sub(r"[_\-]+", " ", stem).strip()

    # If the filename is just a number or very short, use text
    if len(title_from_file) < 5 or title_from_file.isdigit():
        first_line = next(
            (ln.strip() for ln in text_preview.splitlines() if len(ln.strip()) > 10),
            title_from_file
        )
        return first_line[:120]  # cap at 120 chars

    return title_from_file[:120]


def is_irrelevant(filename: str, text: str) -> bool:
    """
    Detect non-regulatory documents that should be excluded.

    Current heuristic: the 'mentions légales' file and very short documents.
    Extend this list as needed without touching the rest of the pipeline.
    """
    irrelevant_keywords = [
        "mentions légales",
        "mentions legales",
        "conditions générales d'utilisation",
        "politique de confidentialité",
    ]
    combined = (filename.lower() + " " + text[:300].lower())
    return any(kw in combined for kw in irrelevant_keywords)


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def process_pdf(pdf_path: Path, doc_id: str) -> dict | None:
    """
    Full extraction + metadata inference for one PDF.

    Returns a structured document dict or None if the file should be skipped.
    """
    log.info(f"Processing: {pdf_path.name}")

    extraction = extract_text_from_pdf(pdf_path)
    if extraction is None:
        log.warning(f"  → Skipped (no extractable text): {pdf_path.name}")
        return None

    text = extraction["full_text"]
    preview = text[:1000]  # used for heuristics — avoids scanning entire text

    if is_irrelevant(pdf_path.name, preview):
        log.info(f"  → Skipped (irrelevant document): {pdf_path.name}")
        return None

    # Detect primary language
    try:
        lang = detect(preview) if preview.strip() else "unknown"
    except:
        lang = "unknown"

    return {
        "doc_id":      doc_id,
        "filename":    pdf_path.name,
        "title":       infer_title(pdf_path.name, preview),
        "type":        infer_doc_type(pdf_path.name, preview),
        "year":        infer_year(pdf_path.name, preview),
        "language":    lang,
        "extraction_method": extraction.get("extraction_method", "fitz"),
        "page_count":  extraction["page_count"],
        "text":        text,
        # Keep per-page breakdown — useful for debugging and future citation features
        "pages":       extraction["pages"],
        "char_count":  len(text),
    }


def extract_all(input_dir: Path = INPUT_DIR, output_file: Path = OUTPUT_FILE) -> list[dict]:
    """
    Extract all PDFs from input_dir and save structured JSON to output_file.

    Returns the list of document dicts (also persisted to disk).
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {input_dir.resolve()}")

    log.info(f"Found {len(pdf_files)} PDF files in '{input_dir}'")

    documents = []
    skipped = 0

    for idx, pdf_path in enumerate(pdf_files):
        doc_id = f"bct_{idx:04d}"   # e.g. "bct_0001"
        doc = process_pdf(pdf_path, doc_id)
        if doc:
            documents.append(doc)
        else:
            skipped += 1

    # Save to disk
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    log.info(
        f"\n✓ Extraction complete: {len(documents)} documents saved, "
        f"{skipped} skipped → {output_file}"
    )

    # Print a summary breakdown by type and year
    _print_summary(documents)

    return documents


def _print_summary(documents: list[dict]) -> None:
    """Log a quick summary table after extraction."""
    types  = {}
    years  = {}
    for doc in documents:
        types[doc["type"]] = types.get(doc["type"], 0) + 1
        yr = doc["year"] or "unknown"
        years[yr] = years.get(yr, 0) + 1

    log.info("Document types: " + ", ".join(f"{t}: {c}" for t, c in sorted(types.items())))
    recent_years = sorted(
        ((yr, c) for yr, c in years.items() if yr != "unknown"),
        reverse=True
    )[:8]
    log.info("Recent years: " + ", ".join(f"{yr}: {c}" for yr, c in recent_years))


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    documents = extract_all()
    # Quick sanity check — print the first document's metadata (no full text)
    if documents:
        sample = {k: v for k, v in documents[0].items() if k != "text" and k != "pages"}
        print("\nSample document metadata:")
        print(json.dumps(sample, ensure_ascii=False, indent=2))