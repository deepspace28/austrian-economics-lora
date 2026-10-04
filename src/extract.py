"""Extract raw text from the source PDFs into EXTRACTED_DIR."""

from pathlib import Path

import fitz  # PyMuPDF

from config import BOOKS_DIR, EXTRACTED_DIR


def extract_text(pdf_path) -> str:
    """Return the concatenated text of every page in the PDF."""
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return "\n".join(pages)


def extract_file(pdf_path, output_path) -> int:
    """Extract one PDF to a .txt file. Returns the character count."""
    text = extract_text(pdf_path)
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
        output_path = EXTRACTED_DIR / f"{pdf.stem}.txt"
        n = extract_file(pdf, output_path)
        print(f"{pdf.name} -> {output_path.name} ({n:,} chars)")


if __name__ == "__main__":
    main()
