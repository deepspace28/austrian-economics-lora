import os
from pathlib import Path
import re

from config import (
    CLEANED_DIR,
    CHUNKS_DIR,
)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
CACHE_DIR = os.environ.get("HF_HOME")

_tokenizer = None


def get_tokenizer():
    """Load the HF tokenizer lazily so the module is importable without
    transformers installed or network access."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        print("Loading tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR
        )
        print("Tokenizer Loaded.")
    return _tokenizer


# --------------------------------------------------
# Settings
# --------------------------------------------------

TARGET_TOKENS = 800
OVERLAP_TOKENS = 100

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def split_paragraphs(text: str):
    paragraphs = re.split(r"\n\s*\n", text)

    cleaned = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if paragraph:

            cleaned.append(paragraph)

    return cleaned


# --------------------------------------------------
# Chunk Builder
# --------------------------------------------------

def build_chunks(paragraphs, count_fn=None):

    if count_fn is None:
        count_fn = count_tokens

    # A paragraph that is empty (or only whitespace) contributes no text but
    # still reaches "\n\n".join, so a run of them produced a chunk that was
    # pure separator - written to disk as an empty training example, and
    # counted by the trainer as a sample with no target. Drop them here so
    # every returned chunk carries real text.
    paragraphs = [p for p in paragraphs if p and p.strip()]

    if not paragraphs:
        return []

    chunks = []

    current_chunk = []

    current_tokens = 0

    for paragraph in paragraphs:

        paragraph_tokens = count_fn(paragraph)

        if current_tokens + paragraph_tokens > TARGET_TOKENS:

            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

            overlap = []

            overlap_tokens = 0

            for previous in reversed(current_chunk):

                t = count_fn(previous)

                if overlap_tokens + t > OVERLAP_TOKENS:
                    break

                overlap.insert(0, previous)

                overlap_tokens += t

            current_chunk = overlap

            current_tokens = overlap_tokens

        current_chunk.append(paragraph)

        current_tokens += paragraph_tokens

    if current_chunk:

        chunks.append("\n\n".join(current_chunk))

    return chunks


# --------------------------------------------------
# Save
# --------------------------------------------------

def save_chunks(book_name, chunks):

    output_folder = CHUNKS_DIR / book_name

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, chunk in enumerate(chunks):

        filename = output_folder / f"chunk_{index+1:04}.txt"

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(chunk)


# --------------------------------------------------
# Main
# --------------------------------------------------

def process_book(book_path: Path):

    print(f"\nProcessing: {book_path.name}")

    text = book_path.read_text(
        encoding="utf-8"
    )

    paragraphs = split_paragraphs(text)

    print(f"Paragraphs: {len(paragraphs)}")

    chunks = build_chunks(paragraphs)

    print(f"Chunks: {len(chunks)}")

    save_chunks(
        book_path.stem,
        chunks
    )


def process_all_books():

    files = sorted(
        CLEANED_DIR.glob("*.txt")
    )

    print(f"Found {len(files)} cleaned books.\n")

    for file in files:

        process_book(file)

    print("\nFinished.")


if __name__ == "__main__":

    process_all_books()