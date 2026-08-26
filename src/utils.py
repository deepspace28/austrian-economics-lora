from pathlib import Path
from config import CHUNKS_DIR
import json


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_jsonl(path: Path, data: list):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")


def load_prompt(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

from pathlib import Path

def list_chunks():

    print(f"Searching in: {CHUNKS_DIR}")
    print(f"Folder exists: {CHUNKS_DIR.exists()}")

    chunks = sorted(CHUNKS_DIR.rglob("*.txt"))

    print(f"Found {len(chunks)} chunk files.")

    return chunks