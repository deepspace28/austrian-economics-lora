import re

MIN_WORDS = 80

# A title page is short, set largely in capitals, and carries almost no
# sentence-ending punctuation. The old rule keyed on one book's title
# ("ECONOMICS IN ONE LESSON"), which does nothing for the other six texts.
TITLE_PAGE_MAX_WORDS = 150
TITLE_PAGE_MIN_PUNCT = 2
TITLE_PAGE_MIN_UPPER_RATIO = 0.6

# Publisher imprints that survive extraction as a line of their own.
PUBLISHER_NAMES = ("HARPER & BROTHERS",)

# Front and back matter is furniture, but these words also occur in ordinary
# prose - "the contents of a savings deposit", "copyright law", "an ISBN".
# Substring-matching them anywhere in the chunk silently dropped real passages
# from the corpus, so a marker only counts as furniture when it is shaped like
# a heading: the whole line, or the line plus a colon.
FURNITURE_MARKERS = (
    r"contents",
    r"table\s+of\s+contents",
    r"copyright",
    r"isbn(?:-1[03])?",
)

FURNITURE_LINE_RE = re.compile(
    r"^(?:" + "|".join(FURNITURE_MARKERS) + r")\s*[:.\-–—]*$",
    re.IGNORECASE,
)

# A contents or index page is a run of "Chapter One .......... 11" entries.
# Dot leaders are the reliable signal; the heading word alone is not.
DOTTED_ENTRY_RE = re.compile(r"\.{4,}\s*\d+\s*$|\.{4,}")
DOTTED_ENTRY_RATIO = 0.34


def _content_lines(text: str):
    return [line.strip() for line in text.splitlines() if line.strip()]


def _names_publisher(line: str) -> bool:
    upper = line.upper()
    return any(name in upper for name in PUBLISHER_NAMES)


def _is_furniture_line(line: str) -> bool:
    """True for a line that is only front/back-matter furniture."""
    return bool(FURNITURE_LINE_RE.match(line)) or _names_publisher(line)


def _is_contents_page(lines) -> bool:
    """True when most lines are dotted-leader entries, i.e. a contents/index page."""
    if not lines:
        return False
    dotted = sum(1 for line in lines if DOTTED_ENTRY_RE.search(line))
    return dotted / len(lines) > DOTTED_ENTRY_RATIO


def _is_title_page(text: str) -> bool:
    words = text.split()
    if len(words) >= TITLE_PAGE_MAX_WORDS:
        return False
    if len(re.findall(r"[.!?]", text)) >= TITLE_PAGE_MIN_PUNCT:
        return False

    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False

    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    return upper_ratio > TITLE_PAGE_MIN_UPPER_RATIO


def is_valid_chunk(text: str) -> bool:
    """
    Returns True if the chunk contains meaningful educational content.
    Returns False for title pages, copyright pages,
    table of contents, or extremely small chunks.
    """

    if not text:
        return False

    text = text.strip()

    # Too short
    if len(text.split()) < MIN_WORDS:
        return False

    lines = _content_lines(text)

    # A contents or index page: mostly dotted-leader entries.
    if _is_contents_page(lines):
        return False

    # Two or more furniture lines means a front-matter page rather than prose
    # that happens to mention one of these words.
    if sum(_is_furniture_line(line) for line in lines) > 1:
        return False

    # Title page
    if _is_title_page(text):
        return False

    # Ignore mostly numeric pages
    letters = len(re.findall(r"[A-Za-z]", text))
    numbers = len(re.findall(r"\d", text))

    if letters == 0:
        return False

    if numbers > letters:
        return False

    return True