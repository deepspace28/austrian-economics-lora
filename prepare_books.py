"""
Build a plain-prose training corpus from the PDFs in books/ (override: SRC_DIR).

Everything here is mechanical extraction from the books themselves -- no
generated or synthetic text.

The core trick is font size. These are typeset books, so running heads, page
numbers, footnotes and marginalia are all set smaller than the body text.
Measuring the modal body size per book and keeping only spans near it removes
that furniture far more reliably than regex heuristics can.

Then: drop front/back matter (contents, index, bibliography), de-hyphenate
across line breaks, and reflow wrapped lines back into paragraphs.

Usage:
    python prepare_books.py                 # all books
    python prepare_books.py "THE CULTURAL WAR.pdf"   # one book (testing)
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
import re
import sys
import unicodedata
from collections import Counter

import fitz  # PyMuPDF

SRC_DIR = os.environ.get("SRC_DIR", os.path.join(_ROOT, "books"))
OUT_DIR = os.path.join(_ROOT, "datasets", "books_clean")
STATS_PATH = os.path.join(_ROOT, "datasets", "books_stats.json")

# Not usable as prose style data:
#   Hayek  -- scanned page images, only ~1,300 real text chars in 13 pages.
#             Would need OCR; it is 0.4% of the corpus.
#   list   -- a reading list, not continuous prose.
SKIP = {
    "Hayek_Use_of_Knowledge_in_Society.pdf",
    "Youth liberty prize reading list.pdf",
}

# Keep spans whose size is within this many points of the book's body size.
SIZE_TOLERANCE = 1.0
# A page whose lines look mostly like contents/index entries is dropped whole.
JUNK_PAGE_RATIO = 0.34
MIN_PARAGRAPH_CHARS = 120
# Points past the left margin before a line counts as a paragraph indent.
INDENT_MIN = 4.0

# "....... 47", "47 .......", "Chapter 3 ... 112" -- contents/index entries
DOTTED_ENTRY = re.compile(r"\.{4,}\s*\d+\s*$|\.{4,}")
# a line that is only a page number / roman numeral
NUMBER_ONLY = re.compile(r"^[\divxlcDIVXLC\.\-–—\s]+$")
# index entries: "Capital, 44, 88, 132-140"
INDEX_LINE = re.compile(r"^[A-Z][^.!?]{0,60},(\s*\d+[\-–—]?\d*,?)+\s*$")
FOOTNOTE_MARK = re.compile(r"(?<=[a-z\.\,\)])\d{1,3}(?=\s|$)")

FRONT_BACK_TITLES = (
    "contents", "table of contents", "index", "bibliography", "references",
    "acknowledgments", "acknowledgements", "about the author", "copyright",
    "notes", "endnotes", "further reading", "appendix",
)


def norm(text):
    """Normalise ligatures, smart quotes and stray whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = (text.replace("\u2019", "'").replace("\u2018", "'")
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2013", "-").replace("\u2014", "--")
                .replace("\u00a0", " ").replace("\ufb01", "fi").replace("\ufb02", "fl"))
    return re.sub(r"[ \t]+", " ", text)


def body_font_size(doc, sample_pages=60):
    """Modal font size weighted by character count = the body text size."""
    counter = Counter()
    step = max(1, len(doc) // sample_pages)
    for i in range(0, len(doc), step):
        for block in doc[i].get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    if txt:
                        counter[round(span["size"], 1)] += len(txt)
    return counter.most_common(1)[0][0] if counter else None


def page_lines(page, body_size):
    """Body-sized lines in reading order, each with its left x-coordinate.

    The x-coordinate is what makes paragraph detection reliable: in a typeset
    book the first line of a paragraph is indented past the left margin.
    """
    out = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", [])
                     if abs(s["size"] - body_size) <= SIZE_TOLERANCE]
            if not spans:
                continue
            text = norm("".join(s.get("text", "") for s in spans)).strip()
            if text:
                out.append((text, round(line["bbox"][0], 1)))
    return out


def looks_like_junk(line):
    if NUMBER_ONLY.match(line):
        return True
    if DOTTED_ENTRY.search(line):
        return True
    if INDEX_LINE.match(line):
        return True
    digits = sum(c.isdigit() for c in line)
    if len(line) > 0 and digits / len(line) > 0.28:
        return True
    return False


def is_front_back_matter(lines):
    head = " ".join(lines[:3]).strip().lower()
    return any(head.startswith(t) or head == t for t in FRONT_BACK_TITLES)


def page_margin(lines):
    """Left margin of a single page.

    Must be per page, not per book: these books use mirrored margins, so recto
    and verso pages start at different x. A book-wide margin misreads every
    other page as one continuous indent.

    Among x positions used by at least a fifth of the page's lines, take the
    leftmost -- so a page dominated by indented quotation still resolves to the
    true margin.
    """
    counts = Counter(x for _, x in lines)
    threshold = max(2, len(lines) * 0.2)
    common = [x for x, n in counts.items() if n >= threshold]
    return min(common) if common else min(counts)


def reflow(lines, typical):
    """Rejoin PDF line wrapping into paragraphs.

    `lines` is a list of (text, rel) where rel is the line's offset from its
    own page's left margin. A paragraph boundary is a *change* in that offset,
    not merely a non-zero one: every line of an indented block quotation is
    offset, so treating each as a new paragraph would shred the quote. Falls
    back to the "short line that ends a sentence" rule for books set flush-left
    with blank-line separation.
    """
    if not lines:
        return []

    paragraphs, buf = [], []
    prev_rel = lines[0][1]

    def flush():
        if buf:
            paragraphs.append(" ".join(buf))
            buf.clear()

    for text, rel in lines:
        indented = abs(rel - prev_rel) > INDENT_MIN
        prev_rel = rel
        # Only honour an indent change if the paragraph so far actually ended a
        # sentence. Layout noise (drop caps, italic run-ins, ragged page
        # margins) produces false indents mid-sentence; a real paragraph break
        # does not appear in the middle of one.
        if indented and buf and re.search(r'[.!?]["\')\]]?$', buf[-1]):
            flush()
        if buf and buf[-1].endswith("-") and not buf[-1].endswith("--"):
            buf[-1] = buf[-1][:-1] + text      # de-hyphenate across the break
        else:
            buf.append(text)
        # flush-left fallback: a sentence ending well short of the full measure
        if not indented and re.search(r'[.!?]["\')]?$', text) and len(text) < typical * 0.72:
            flush()
    flush()
    return paragraphs


def clean_paragraph(par):
    par = re.sub(r"\s+", " ", par).strip()
    par = FOOTNOTE_MARK.sub("", par)           # drop inline footnote markers
    par = re.sub(r"\s+([,.;:!?])", r"\1", par)
    return par.strip()


def extract(path):
    doc = fitz.open(path)
    body = body_font_size(doc)
    stats = {"pages": len(doc), "body_font": body, "pages_kept": 0,
             "pages_dropped": 0, "paragraphs": 0, "chars": 0}
    if body is None:
        doc.close()
        return [], stats

    # Pass 1: collect body lines with coordinates, dropping junk pages.
    kept_pages = []
    for page in doc:
        lines = page_lines(page, body)
        if not lines:
            stats["pages_dropped"] += 1
            continue
        texts = [t for t, _ in lines]
        if is_front_back_matter(texts):
            stats["pages_dropped"] += 1
            continue
        junk = sum(looks_like_junk(t) for t in texts)
        if junk / len(texts) > JUNK_PAGE_RATIO:
            stats["pages_dropped"] += 1
            continue
        kept = [(t, x) for t, x in lines if not looks_like_junk(t)]
        if not kept:
            stats["pages_dropped"] += 1
            continue
        stats["pages_kept"] += 1
        kept_pages.append(kept)

    doc.close()

    # Pass 2: resolve each page's own margin, then flag indented lines.
    lengths = sorted(len(t) for page in kept_pages for t, _ in page)
    typical = lengths[int(len(lengths) * 0.75)] if lengths else 1

    flat = []
    for page in kept_pages:
        margin = page_margin(page)
        for text, x0 in page:
            flat.append((text, round(x0 - margin, 1)))
    boundaries = sum(1 for i in range(1, len(flat))
                     if abs(flat[i][1] - flat[i - 1][1]) > INDENT_MIN)
    stats["indent_rate"] = round(boundaries / max(len(flat), 1), 3)

    # Reflow across the whole book so paragraphs join over page breaks.
    paragraphs = reflow(flat, typical)

    final = []
    for par in paragraphs:
        par = clean_paragraph(par)
        if len(par) >= MIN_PARAGRAPH_CHARS and re.search(r"[a-z]{3}", par):
            final.append(par)
    stats["paragraphs"] = len(final)
    stats["chars"] = sum(len(p) for p in final)
    return final, stats


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    targets = sys.argv[1:] or sorted(os.listdir(SRC_DIR))
    pdfs = [t for t in targets if t.lower().endswith(".pdf") and t not in SKIP]
    skipped = [t for t in targets if t in SKIP]
    for s in skipped:
        print(f"  skipping {s} (see SKIP)", flush=True)

    all_stats, total_chars = {}, 0
    for name in pdfs:
        path = os.path.join(SRC_DIR, name)
        if not os.path.exists(path):
            print(f"  missing: {name}", flush=True)
            continue
        print(f"reading {name} ...", flush=True)
        paragraphs, stats = extract(path)
        out_path = os.path.join(OUT_DIR, os.path.splitext(name)[0] + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(paragraphs))
        all_stats[name] = stats
        total_chars += stats["chars"]
        print(f"  body {stats['body_font']}pt | pages {stats['pages_kept']} kept / "
              f"{stats['pages_dropped']} dropped | {stats['paragraphs']} paragraphs | "
              f"{stats['chars']/1e6:.2f}M chars", flush=True)

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\ntotal: {total_chars/1e6:.2f}M chars across {len(all_stats)} book(s)")
    print(f"       ~{total_chars/4/1e6:.2f}M tokens (rough)")
    print(f"output -> {OUT_DIR}")


if __name__ == "__main__":
    main()
