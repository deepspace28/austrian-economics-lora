from pathlib import Path

from config import CHUNKS_DIR


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_prompt(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def list_chunks():
    print(f"Searching in: {CHUNKS_DIR}")

    chunks = sorted(CHUNKS_DIR.rglob("*.txt"))

    print(f"Found {len(chunks)} chunk files.")

    return chunks