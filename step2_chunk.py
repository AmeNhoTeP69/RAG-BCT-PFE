"""
step2_chunk.py
──────────────
STEP 2 — Smart Section-Based Chunking for BCT Regulatory Documents.

Why section-based over naive sliding window?
────────────────────────────────────────────
BCT circulaires and notes are structured around numbered articles:
  "Article 1 – Champ d'application"
  "Art. 2 – Définitions"
  "CHAPITRE I – DISPOSITIONS GÉNÉRALES"

Splitting on these natural boundaries means each chunk is semantically
self-contained — a retrieval hit on "Article 3" will return the *entire*
Article 3, not half of it mixed with Article 4.

Fallback strategy:
  If a document has no detectable article headers (some notes are free-form
  prose), we fall back to a sentence-aware sliding window so nothing is lost.

Chunk size policy:
  - Target: 300–500 words per chunk
  - If an article is very long (> 600 words), it is split further at
    paragraph/sentence boundaries — never mid-sentence.
  - If an article is very short (< 50 words), it is merged with the next one.
  - Overlap: 50-word sentence-boundary overlap between merged/split chunks
    (preserves context across boundaries).

Each chunk carries full metadata so any downstream step can filter by
document type, year, or source article without re-parsing.
"""

import re
import json
import uuid
import logging
from pathlib import Path
from typing import Optional

from utils.text_utils import normalize_french_text, remove_boilerplate, count_words, sentence_split_french

# ─── Configuration ────────────────────────────────────────────────────────────

INPUT_FILE  = Path("data/extracted/documents.json")
OUTPUT_DIR  = Path("data/chunks")
OUTPUT_FILE = OUTPUT_DIR / "chunks.json"

# Word-count thresholds
CHUNK_TARGET_WORDS  = 400   # ideal chunk size
CHUNK_MAX_WORDS     = 600   # split if article exceeds this
CHUNK_MIN_WORDS     = 50    # merge with next if article is shorter than this
OVERLAP_WORDS       = 75    # words of overlap between split sub-chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Section detection patterns ───────────────────────────────────────────────
#
# These regex patterns cover the main structural markers in BCT documents.
# Priority order matters: more specific patterns first.
#
# Examples matched:
#   "Article 1er – Champ d'application"
#   "Art. 2 - Définitions"
#   "ARTICLE 3 :"
#   "CHAPITRE I – DISPOSITIONS GÉNÉRALES"
#   "TITRE II – DES OBLIGATIONS"
#   "SECTION 1 – ..."
#   "I. Contexte"       (Roman numerals, used in free-form notes)
#   "1. Objet"          (Simple numbered sections)

SECTION_PATTERNS = [
    # Full "Article" keyword (most common in circulaires)
    re.compile(
        r"(?im)"
        r"^(?P<header>"
        r"(?:article|art\.?)\s+"          # "Article" or "Art."
        r"(?:\d+(?:er|ème|e)?|premier)"   # number: 1, 1er, 2ème, premier
        r"(?:\s*[.:\-–—]\s*(?P<title>[^\n]{0,80}))?"  # optional title
        r")\s*$"
    ),
    # Chapter / Title / Section headings
    re.compile(
        r"(?im)"
        r"^(?P<header>"
        r"(?:chapitre|titre|section|paragraphe)\s+"
        r"(?:[IVXivx]+|\d+)"              # Roman or Arabic numeral
        r"(?:\s*[.:\-–—]\s*(?P<title>[^\n]{0,80}))?"
        r")\s*$"
    ),
    # Numbered headings like "1. Objet" or "I. Contexte" (free-form notes)
    re.compile(
        r"(?im)"
        r"^(?P<header>"
        r"(?:[IVXivx]{1,6}|\d{1,2})\."   # "I." or "1."
        r"\s+(?P<title>[A-ZÀ-Ö][^\n]{2,60})"  # title starting with capital
        r")\s*$"
    ),
]


def detect_sections(text: str) -> list[dict]:
    """
    Identify all section boundaries in a document's text.

    Returns a list of dicts:
      { "start": int, "end": int, "header": str, "title": str }
    sorted by position.

    "end" of each section = "start" of the next (or end of document).
    """
    matches = []
    for pattern in SECTION_PATTERNS:
        for m in pattern.finditer(text):
            header = m.group("header").strip()
            title_group = m.groupdict().get("title")
            title = title_group.strip() if title_group else ""
            matches.append({
                "start":  m.start(),
                "header": header,
                "title":  title,
            })

    if not matches:
        return []

    # Sort by position, deduplicate overlapping matches
    matches.sort(key=lambda x: x["start"])
    deduped = [matches[0]]
    for m in matches[1:]:
        # Skip if within 10 chars of previous match (same boundary detected twice)
        if m["start"] - deduped[-1]["start"] > 10:
            deduped.append(m)

    # Add "end" positions
    for i, section in enumerate(deduped):
        section["end"] = deduped[i + 1]["start"] if i + 1 < len(deduped) else len(text)

    return deduped


# ─── Chunk builders ───────────────────────────────────────────────────────────

def build_chunk(text: str, metadata: dict, section_header: str = "") -> dict:
    """
    Package a text segment + metadata into a chunk dict.

    Generates a stable UUID from the doc_id + section_header so re-running
    the pipeline produces the same chunk IDs (idempotent).
    """
    # Deterministic ID: reproducible across runs
    seed = f"{metadata['doc_id']}::{section_header}::{text[:50]}"
    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))

    return {
        "chunk_id": chunk_id,
        "text":     text.strip(),
        "metadata": {
            "doc_id":         metadata["doc_id"],
            "title":          metadata["title"],
            "type":           metadata["type"],
            "year":           metadata["year"],
            "filename":       metadata["filename"],
            "section_header": section_header,
            "word_count":     count_words(text),
        }
    }


def split_long_text(text: str, max_words: int, overlap_words: int) -> list[str]:
    """
    Split a text that exceeds max_words into overlapping sub-chunks.

    Uses sentence boundaries for splits — never cuts mid-sentence.
    Overlap is achieved by appending the last N words of the previous
    chunk to the start of the next one.

    Example with max=300, overlap=75:
      chunk1 = sentences 1-18  (≈300 words)
      chunk2 = last 75w of chunk1 + sentences 19-36  (≈300 words)
      ...
    """
    sentences = sentence_split_french(text)
    if not sentences:
        return [text]

    chunks = []
    current_sentences = []
    current_word_count = 0
    overlap_buffer: list[str] = []   # sentences carried over for overlap

    for sent in sentences:
        sent_words = count_words(sent)

        if current_word_count + sent_words > max_words and current_sentences:
            # Flush current chunk
            chunks.append(" ".join(current_sentences))

            # Build overlap buffer: keep last sentences summing to ~overlap_words
            overlap_buffer = []
            overlap_count = 0
            for s in reversed(current_sentences):
                w = count_words(s)
                if overlap_count + w > overlap_words:
                    break
                overlap_buffer.insert(0, s)
                overlap_count += w

            current_sentences = overlap_buffer.copy()
            current_word_count = overlap_count

        current_sentences.append(sent)
        current_word_count += sent_words

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def merge_short_sections(sections: list[dict], min_words: int) -> list[dict]:
    """
    Merge consecutive short sections into a single logical chunk.

    A section under min_words is appended to the previous one (or buffered
    until the buffer reaches min_words). This avoids one-liner chunks like:
      "Article 5 – Entrée en vigueur\nLa présente circulaire entre en vigueur..."
    which are too short to embed meaningfully.
    """
    if not sections:
        return []

    merged = []
    buffer_text = ""
    buffer_header = ""

    for sec in sections:
        wc = count_words(sec["text"])

        if wc < min_words:
            # Accumulate into buffer
            sep = "\n\n" if buffer_text else ""
            buffer_text   += sep + (f"{sec['header']}\n{sec['text']}" if sec.get("header") else sec["text"])
            buffer_header  = buffer_header or sec.get("header", "")
        else:
            # Flush buffer if we have one
            if buffer_text:
                merged.append({
                    "text":   buffer_text,
                    "header": buffer_header,
                })
                buffer_text   = ""
                buffer_header = ""
            merged.append(sec)

    # Don't forget trailing buffer
    if buffer_text:
        if merged:
            # Append to last chunk
            merged[-1]["text"] += "\n\n" + buffer_text
        else:
            merged.append({"text": buffer_text, "header": buffer_header})

    return merged


# ─── Document-level chunking ──────────────────────────────────────────────────

def chunk_document(doc: dict) -> list[dict]:
    """
    Produce all chunks for one document.

    Strategy:
      1. Clean the text
      2. Detect article/section headers
      3. If headers found → section-based chunking
         If not found     → sentence-aware sliding window (fallback)
      4. Merge short sections, split long ones
      5. Wrap each chunk with full metadata
    """
    doc_id   = doc["doc_id"]
    text     = doc["text"]
    metadata = {
        "doc_id":   doc_id,
        "title":    doc.get("title", ""),
        "type":     doc.get("type", "unknown"),
        "year":     doc.get("year"),
        "filename": doc.get("filename", ""),
    }

    # Step 1: Clean text
    text = normalize_french_text(text)
    text = remove_boilerplate(text)

    # Step 2: Detect sections
    sections = detect_sections(text)

    if sections:
        # ── Section-based path ──────────────────────────────────────────────
        log.debug(f"  {doc_id}: {len(sections)} sections detected")

        # Extract preamble (text before the first section)
        preamble = text[:sections[0]["start"]].strip()
        raw_sections = []

        if count_words(preamble) >= CHUNK_MIN_WORDS:
            raw_sections.append({"text": preamble, "header": "Préambule"})

        for sec in sections:
            section_text = text[sec["start"]:sec["end"]].strip()
            # Remove the header line from the body text to avoid duplication
            header_line = sec["header"]
            if section_text.startswith(header_line):
                section_text = section_text[len(header_line):].strip()

            if section_text:
                raw_sections.append({
                    "text":   section_text,
                    "header": header_line,
                })

        # Step 3: Merge short, split long
        processed = merge_short_sections(raw_sections, CHUNK_MIN_WORDS)
        chunks = []

        for sec in processed:
            wc = count_words(sec["text"])
            if wc > CHUNK_MAX_WORDS:
                sub_texts = split_long_text(sec["text"], CHUNK_TARGET_WORDS, OVERLAP_WORDS)
                for i, sub in enumerate(sub_texts):
                    header = f"{sec['header']} (part {i+1})" if len(sub_texts) > 1 else sec["header"]
                    chunks.append(build_chunk(sub, metadata, header))
            else:
                chunks.append(build_chunk(sec["text"], metadata, sec["header"]))

    else:
        # ── Fallback: sentence-aware sliding window ─────────────────────────
        log.debug(f"  {doc_id}: no sections found, using sliding window fallback")
        sub_texts = split_long_text(text, CHUNK_TARGET_WORDS, OVERLAP_WORDS)
        chunks = [
            build_chunk(sub, metadata, f"segment_{i+1}")
            for i, sub in enumerate(sub_texts)
        ]

    log.info(f"  {doc_id} → {len(chunks)} chunks "
             f"(avg {sum(c['metadata']['word_count'] for c in chunks)//max(len(chunks),1)} words/chunk)")
    return chunks


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def chunk_all(
    input_file:  Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
) -> list[dict]:
    """
    Load extracted documents, chunk them all, and save to disk.

    Returns the flat list of all chunks.
    """
    input_file  = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, encoding="utf-8") as f:
        documents = json.load(f)

    log.info(f"Chunking {len(documents)} documents …")

    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    _print_chunk_summary(all_chunks)
    log.info(f"\n✓ Chunking complete: {len(all_chunks)} chunks → {output_file}")
    return all_chunks


def _print_chunk_summary(chunks: list[dict]) -> None:
    word_counts = [c["metadata"]["word_count"] for c in chunks]
    avg  = sum(word_counts) // len(word_counts) if word_counts else 0
    lo   = min(word_counts, default=0)
    hi   = max(word_counts, default=0)

    by_type: dict[str, int] = {}
    for c in chunks:
        t = c["metadata"]["type"]
        by_type[t] = by_type.get(t, 0) + 1

    log.info(f"Chunk word counts — min: {lo}, avg: {avg}, max: {hi}")
    log.info("Chunks by doc type: " + ", ".join(f"{t}: {n}" for t, n in sorted(by_type.items())))


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chunks = chunk_all()

    # Inspect a few sample chunks
    print("\n── Sample chunks ───────────────────────────────────────────")
    for chunk in chunks[:3]:
        print(f"\nchunk_id : {chunk['chunk_id']}")
        print(f"section  : {chunk['metadata']['section_header']}")
        print(f"doc      : {chunk['metadata']['title']} ({chunk['metadata']['year']})")
        print(f"words    : {chunk['metadata']['word_count']}")
        print(f"text     : {chunk['text'][:200]} …")
        print("─" * 60)