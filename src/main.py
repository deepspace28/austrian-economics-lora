"""Dataset generation entry point: chunks -> instruction pairs.

Run from the repository root:

    python src/main.py
"""

import argparse
import json
import sys
from pathlib import Path

# The modules in src/ import each other by bare name, which only resolves when
# src/ itself is on sys.path. Adding it here keeps `python src/main.py` and
# `python -m src.main` both working, and makes this file importable from tests.
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SRC))

from exporter import DatasetExporter
from utils import list_chunks
from config import DATASET_DIR, PROJECT_ROOT


PROGRESS_FILE = PROJECT_ROOT / "outputs" / "progress.json"


def load_progress():
    if not PROGRESS_FILE.exists():
        return 0

    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("last_chunk", 0)
    except (OSError, ValueError, AttributeError):
        # A truncated or non-object progress file should restart the run, not
        # crash it -- but it must not silently resume from 0 either, because the
        # exporter appends and would duplicate every pair already written.
        raise SystemExit(
            f"Unreadable progress file: {PROGRESS_FILE}. "
            f"Delete it to start over, or restore it to resume."
        )


def save_progress(index):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_chunk": index}, f, indent=4)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate instruction pairs from chunks.")
    parser.add_argument(
        "--reset-progress",
        action="store_true",
        help="Delete the progress file and start over from chunk 0.",
    )
    if argv is None:
        argv = sys.argv[1:] if __name__ == "__main__" else []
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.reset_progress and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print(f"Removed {PROGRESS_FILE}; starting from chunk 0.")

    print("=" * 60)
    print(" Austrian Economics Dataset Generator ")
    print("=" * 60)

    chunks = list_chunks()
    print(f"\nFound {len(chunks)} chunks.\n")

    start = load_progress()
    if start >= len(chunks):
        print(f"Already processed all {len(chunks)} chunks. Nothing to do.\n")
        return

    if start > 0:
        print(f"Resuming from chunk {start + 1}\n")

    # Imported here, not at module scope: dataset_generator pulls in torch and
    # transformers, which made `import main` cost ~20s and required a GPU stack
    # just to exercise the loop in tests.
    from dataset_generator import DatasetGenerator

    generator = DatasetGenerator()
    exporter = DatasetExporter(DATASET_DIR / "training_dataset.jsonl")

    total_examples = 0
    skipped = 0
    processed = 0
    failed = 0

    for index in range(start, len(chunks)):
        chunk = chunks[index]
        print("-" * 60)
        print(f"[{index + 1}/{len(chunks)}] {chunk.name}")

        try:
            examples = generator.generate_from_file(chunk).get("examples", [])

            if not examples:
                skipped += 1
            else:
                exporter.save(examples)
                total_examples += len(examples)

            processed += 1

        except Exception as e:
            # Progress advances past a failed chunk so one bad chunk cannot block
            # the run forever; the counter below reports how many there were.
            print(f"Error on {chunk.name}: {e}")
            failed += 1

        save_progress(index + 1)

    print("\n" + "=" * 60)
    print("Finished!")
    print("=" * 60)
    print(f"Chunks Processed : {processed}")
    print(f"Chunks Skipped  : {skipped}")
    print(f"Chunks Failed   : {failed}")
    print(f"Examples Saved  : {total_examples}")
    print(f"Dataset File    : {DATASET_DIR / 'training_dataset.jsonl'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
