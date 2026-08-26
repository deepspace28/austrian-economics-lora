import re

MIN_WORDS = 80


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

    upper = text.upper()

    # Ignore table of contents
    if "CONTENTS" in upper:
        return False

    # Ignore copyright pages
    if "COPYRIGHT" in upper:
        return False

    # Ignore ISBN pages
    if "ISBN" in upper:
        return False

    # Ignore publisher pages
    if "HARPER & BROTHERS" in upper:
        return False

    # Ignore title pages
    if "ECONOMICS IN ONE LESSON" in upper and len(text.split()) < 150:
        return False

    # Ignore mostly numeric pages
    letters = len(re.findall(r"[A-Za-z]", text))
    numbers = len(re.findall(r"\d", text))

    if letters == 0:
        return False

    if numbers > letters:
        return False

    return True