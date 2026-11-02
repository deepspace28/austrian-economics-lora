from pathlib import Path

from config import DATASET_DIR, PROJECT_ROOT
from dataset_generator import DatasetGenerator
from exporter import DatasetExporter
from progress import load_progress, save_progress
from utils import list_chunks


PROGRESS_FILE = PROJECT_ROOT / "outputs" / "progress.json"


def main():
    print("=" * 60)
    print(" Austrian Economics Dataset Generator ")
    print("=" * 60)

    generator = DatasetGenerator()
    exporter = DatasetExporter(DATASET_DIR / "training_dataset.jsonl")
    chunks = list_chunks()
    print(f"\nFound {len(chunks)} chunks.\n")

    start = load_progress(PROGRESS_FILE, total=len(chunks))
    if start > 0:
        print(f"Resuming from chunk {start + 1}\n")

    total_examples = 0
    skipped = 0
    processed = 0

    for index in range(start, len(chunks)):
        chunk = chunks[index]
        print("-" * 60)
        print(f"[{index + 1}/{len(chunks)}] {chunk.name}")

        try:
            result = generator.generate_from_file(chunk)
            examples = result.get("examples", [])

            if len(examples) == 0:
                skipped += 1
            else:
                exporter.save(examples)
                total_examples += len(examples)

            processed += 1
            save_progress(index + 1, PROGRESS_FILE)
        except Exception as error:
            print(error)
            print(
                "Stopping without advancing the checkpoint. "
                "Re-run to retry this chunk.\n"
            )
            break

    print("\n")
    print("=" * 60)
    print("Finished!")
    print("=" * 60)
    print(f"Chunks Processed : {processed}")
    print(f"Chunks Skipped  : {skipped}")
    print(f"Examples Saved  : {total_examples}")
    print(f"Dataset File    : {DATASET_DIR / 'training_dataset.jsonl'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
