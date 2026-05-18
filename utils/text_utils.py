"""
utils/text_utils.py
───────────────────
Shared text helpers used across multiple pipeline steps.
Kept here to avoid duplication between step1, step2, etc.
"""

import re
import unicodedata


def normalize_french_text(text: str) -> str:
    """
    Normalize French text for consistent processing.

    What we do (and why):
      - NFC normalization: ensures accented characters like 'é' are stored as
        a single code point, not as 'e' + combining accent. This prevents
        mismatches in string comparisons and tokenization.
      - Curly quotes → straight quotes: some PDFs embed typographic quotes
        that confuse tokenizers and regex patterns.
      - Em/en dashes → hyphen: standardizes punctuation for French legal text
        where both styles appear interchangeably.
      - Non-breaking spaces → regular spaces: common in French typography
        (used before «, », !, ?, :, ;) but break tokenization.

    We intentionally do NOT lowercase or remove punctuation here — that
    would harm downstream NER and semantic search quality.
    """
    # Unicode normalization (NFC: composed form)
    text = unicodedata.normalize("NFC", text)

    # Typographic quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')   # " "
    text = text.replace("\u2018", "'").replace("\u2019", "'")   # ' '
    text = text.replace("\u00ab", '"').replace("\u00bb", '"')   # « »

    # Dashes
    text = text.replace("\u2013", "-").replace("\u2014", "-")   # – —

    # Non-breaking and thin spaces
    text = text.replace("\u00a0", " ").replace("\u202f", " ")

    # Collapse multiple whitespace (preserve single newlines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def remove_boilerplate(text: str) -> str:
    """
    Remove recurring BCT document boilerplate that adds noise without meaning.

    Examples of boilerplate found in BCT documents:
      - Page headers/footers with "Banque Centrale de Tunisie", page numbers
      - "www.bct.gov.tn" URLs
      - Repeated section dividers (lines of dashes/dots)
    """
    # Page number patterns (e.g. "Page 3 / 12", "- 3 -", "3 sur 12")
    text = re.sub(r"(?m)^[-–]\s*\d+\s*[-–]$", "", text)
    text = re.sub(r"(?m)^Page\s+\d+\s*[/sur]+\s*\d+$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\d+\s*/\s*\d+$", "", text)

    # BCT website and address boilerplate
    text = re.sub(r"www\.bct\.gov\.tn", "", text, flags=re.IGNORECASE)
    text = re.sub(r"bct@bct\.gov\.tn", "", text, flags=re.IGNORECASE)

    # Lines that are pure punctuation/decoration (e.g. "------", ".........")
    text = re.sub(r"(?m)^[.\-_=*]{4,}$", "", text)

    # Collapse blank lines left by removals
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def count_words(text: str) -> int:
    """Count whitespace-separated tokens — fast proxy for word count."""
    return len(text.split())


def sentence_split_french(text: str) -> list[str]:
    """
    Split text into sentences, handling French-specific patterns.

    Why not use spaCy sentence splitting here?
    - Too slow to run on every text block during chunking
    - We only need approximate sentence boundaries for overlap calculation

    French-specific considerations:
      - Abbreviations: M., Mme., art., al., etc. should not split sentences
      - Numbers: "1.500 dinars" should not split at the dot
      - Decimal commas are standard in French: "3,5%" — no false split risk there
    """
    # Protect common French abbreviations from triggering splits
    ABBREVS = [
        "M.", "Mme.", "Dr.", "Prof.", "art.", "al.", "ibid.", "op.", "cit.",
        "cf.", "etc.", "ex.", "fig.", "vol.", "n°", "N°",
    ]
    placeholder = text
    abbrev_map = {}
    for i, abbrev in enumerate(ABBREVS):
        token = f"__ABBREV{i}__"
        abbrev_map[token] = abbrev
        placeholder = placeholder.replace(abbrev, token)

    # Also protect decimal numbers like "1.500" or "art.12"
    placeholder = re.sub(r"(\d)\.(\d)", r"\1__DOT__\2", placeholder)

    # Split on sentence-ending punctuation followed by space + capital
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ö])", placeholder)

    # Restore protected tokens
    result = []
    for s in sentences:
        s = s.replace("__DOT__", ".")
        for token, abbrev in abbrev_map.items():
            s = s.replace(token, abbrev)
        s = s.strip()
        if s:
            result.append(s)

    return result