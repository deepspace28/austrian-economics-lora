"""
Fetch the source texts this project trains on.

The PDFs are not redistributed with this repository. Most Mises Institute
editions are published under CC BY-NC-ND, which permits free reading and
copying but not the redistribution of derivative works — and a cleaned corpus
or a generated Q&A set is a derivative work. So you download your own copies.

    python fetch_books.py            # fetch everything it can resolve
    python fetch_books.py --list     # just show where each text lives

Anything the script cannot resolve automatically prints a search link. Drop
the PDF into books/ under the filename shown and the pipeline picks it up.
"""

import argparse
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BOOKS_DIR = Path(__file__).resolve().parent / "books"

UA = {"User-Agent": "Mozilla/5.0 (compatible; austrian-econ-lora/1.0)"}

# `pdf`  - direct link, verified
# `page` - Mises library page; the script scrapes it for a cdn.mises.org PDF
# both None - resolve by hand via the printed search link
BOOKS = [
    {
        "filename": "Mises_Human_Action.pdf",
        "title": "Human Action: A Treatise on Economics",
        "author": "Ludwig von Mises",
        "page": "https://mises.org/library/book/human-action",
        "pdf": None,
    },
    {
        "filename": "Rothbard_Man_Economy_and_State.pdf",
        "title": "Man, Economy, and State with Power and Market",
        "author": "Murray N. Rothbard",
        "page": "https://mises.org/library/book/man-economy-and-state-power-and-market",
        "pdf": "https://cdn.mises.org/Man,%20Economy,%20and%20State,"
               "%20with%20Power%20and%20Market_2.pdf",
    },
    {
        "filename": "Rothbard_For_a_New_Liberty.pdf",
        "title": "For a New Liberty: The Libertarian Manifesto",
        "author": "Murray N. Rothbard",
        "page": None,
        "pdf": None,
    },
    {
        "filename": "Mises_Interventionism.pdf",
        "title": "Interventionism: An Economic Analysis",
        "author": "Ludwig von Mises",
        "page": None,
        "pdf": None,
    },
    {
        "filename": "Hazlitt_Economics_in_One_Lesson.pdf",
        "title": "Economics in One Lesson",
        "author": "Henry Hazlitt",
        "page": None,
        "pdf": None,
    },
    {
        "filename": "Hoppe_Socialism_Capitalism.pdf",
        "title": "A Theory of Socialism and Capitalism",
        "author": "Hans-Hermann Hoppe",
        "page": None,
        "pdf": None,
    },
    {
        "filename": "Bagus_Tragedy_of_the_Euro.pdf",
        "title": "The Tragedy of the Euro",
        "author": "Philipp Bagus",
        "page": None,
        "pdf": None,
    },
]


def search_url(title):
    return "https://mises.org/search?query=" + urllib.parse.quote_plus(title)


def resolve_pdf(book):
    """Return a direct PDF URL, scraping the library page if needed."""
    if book["pdf"]:
        return book["pdf"]
    if not book["page"]:
        return None
    try:
        req = urllib.request.Request(book["page"], headers=UA)
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", "replace")
    except Exception as exc:
        print(f"    could not open library page: {exc}")
        return None
    # The download button points at the CDN.
    matches = re.findall(r'https://cdn\.mises\.org/[^"\'\s>]+?\.pdf', html)
    return urllib.parse.unquote(matches[0]) if matches else None


def download(url, dest):
    req = urllib.request.Request(urllib.parse.quote(url, safe=":/%?=&"), headers=UA)
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as fh:
            while chunk := resp.read(65536):
                fh.write(chunk)
                got += len(chunk)
                if total:
                    pct = 100 * got / total
                    print(f"\r    {got/1e6:6.1f} / {total/1e6:.1f} MB  ({pct:5.1f}%)",
                          end="", flush=True)
        print()
    tmp.replace(dest)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="show sources without downloading")
    args = ap.parse_args()

    BOOKS_DIR.mkdir(exist_ok=True)
    missing = []

    for book in BOOKS:
        dest = BOOKS_DIR / book["filename"]
        print(f"\n{book['title']}\n  {book['author']}")

        if dest.exists():
            print(f"  already present ({dest.stat().st_size/1e6:.1f} MB)")
            continue

        if args.list:
            print(f"  -> {book['pdf'] or book['page'] or search_url(book['title'])}")
            continue

        url = resolve_pdf(book)
        if not url:
            print(f"  ! not resolvable automatically")
            print(f"    search: {search_url(book['title'])}")
            print(f"    save as: books/{book['filename']}")
            missing.append(book)
            continue

        print(f"  downloading {url}")
        try:
            size = download(url, dest)
            print(f"  saved books/{book['filename']} ({size/1e6:.1f} MB)")
        except Exception as exc:
            print(f"  ! download failed: {exc}")
            print(f"    search: {search_url(book['title'])}")
            missing.append(book)

    if missing and not args.list:
        print(f"\n{len(missing)} of {len(BOOKS)} need to be fetched by hand "
              f"(see the search links above).")
    print("\nNext: python src/main.py    # extract -> clean -> chunk -> dataset")
    return 1 if missing and not args.list else 0


if __name__ == "__main__":
    sys.exit(main())
