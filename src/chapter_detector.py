import re

# Canonical roman numerals up to 3999, anchored on both ends so that only
# well-formed numerals match. The earlier "[IVXLCDM]+" idiom also matched
# ordinary words built from those letters (ITERATION, MIXED, CIVIC).
ROMAN_RE = re.compile(
    r"^(?:"
    r"M{0,3}(?:CM|CD|D?C{0,3})"
    r"(?:XC|XL|L?X{0,3})"
    r"(?:IX|IV|V?I{0,3})"
    r")$"
)

HEADING_RE = re.compile(r"^(?:CHAPTER|Chapter|PART|Part)\s+(\S+)", re.MULTILINE)


def _is_numeral(token: str) -> bool:
    """True if the heading token is a decimal or canonical roman numeral."""
    return bool(token) and (token.isdigit() or ROMAN_RE.match(token) is not None)


def detect_chapters(text):
    """
    Detects chapters in the text using common patterns.
    Currently returns the text as a single 'chapter' if no patterns are found.
    """

    matches = [m for m in HEADING_RE.finditer(text) if _is_numeral(m.group(1))]

    if not matches:
        return [{"title": "Full Text", "content": text}]

    chapters = []

    # Text before the first heading is real content (preface, introduction).
    # Keep it as its own leading chunk instead of dropping it.
    preamble = text[: matches[0].start()].strip()
    if preamble:
        chapters.append({"title": "Preamble", "content": preamble})

    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        nl = text.find('\n', start)
        title = text[start: nl if nl != -1 else len(text)].strip()
        content = text[start:end].strip()

        chapters.append({
            "title": title,
            "content": content
        })

    return chapters

def main():
    print("Chapter detection module loaded.")

if __name__ == "__main__":
    main()
