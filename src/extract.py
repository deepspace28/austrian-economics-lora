"""Extract raw text from the source PDFs into EXTRACTED_DIR."""

from pathlib import Path

from config import BOOKS_DIR, EXTRACTED_DIR

# A scanned or image-only page yields no text at all. Enough of them in a row
# means the PDF needs OCR rather than extraction, and silently writing an
# empty .txt file makes that look like a successful run.
EMPTY_PAGE_CHARS = 20


def extract_text(pdf_path):
    """Return (text, empty_page_count) for every page in the PDF.

    Imported lazily so this module stays importable without PyMuPDF, which
    keeps the stdlib-only CI job and the extract tests working.
    """
    import fitz

    pages = []
    empty_pages = 0
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text")
            if len(text.strip()) < EMPTY_PAGE_CHARS:
                empty_pages += 1
            pages.append(text)
    return "\n".join(pages), empty_pages


def extract_file(pdf_path, output_path) -> int:
    """Extract one PDF to a .txt file. Returns the character count."""
    text, _ = extract_text(pdf_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(text, encoding="utf-8")
    return len(text)


def main():
    if not BOOKS_DIR.exists():
        print(f"Books directory not found: {BOOKS_DIR}")
        return

    pdfs = sorted(BOOKS_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs.")

    for pdf in pdfs:
        text, empty_pages = extract_text(pdf)
        if empty_pages:
            print(f"  warning: {empty_pages} page(s) with no extractable text")

        output_path = EXTRACTED_DIR / f"{pdf.stem}.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"{pdf.name} -> {output_path.name} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
