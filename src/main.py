import json
from pathlib import Path

from dataset_generator import DatasetGenerator
from exporter import DatasetExporter
from utils import list_chunks
from config import DATASET_DIR, PROJECT_ROOT


PROGRESS_FILE = PROJECT_ROOT / "outputs" / "progress.json"

PROGRESS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

def load_progress():

    if not PROGRESS_FILE.exists():

        return 0

    try:

        with open(PROGRESS_FILE, "r") as f:

            data = json.load(f)

            return data.get("last_chunk", 0)

    except:

        return 0


def save_progress(index):

    with open(PROGRESS_FILE, "w") as f:

        json.dump(
            {
                "last_chunk": index
            },
            f,
            indent=4
        )


def main():

    print("=" * 60)
    print(" Austrian Economics Dataset Generator ")
    print("=" * 60)

    generator = DatasetGenerator()

    exporter = DatasetExporter(

        DATASET_DIR / "training_dataset.jsonl"

    )

    chunks = list_chunks()

    print(f"\nFound {len(chunks)} chunks.\n")

    start = load_progress()

    if start > 0:

        print(
            f"Resuming from chunk {start+1}\n"
        )

    total_examples = 0

    skipped = 0

    processed = 0


    for index in range(start, len(chunks)):

        chunk = chunks[index]

        print("-" * 60)

        print(

            f"[{index+1}/{len(chunks)}] "

            f"{chunk.name}"

        )

        try:

            result = generator.generate_from_file(

                chunk

            )

            examples = result.get(

                "examples",

                []

            )

            if len(examples) == 0:

                skipped += 1

            else:

                exporter.save(

                    examples

                )

                total_examples += len(

                    examples

                )

            processed += 1

            save_progress(

                index + 1

            )

        except Exception as e:

            print(e)

            print(

                "Skipping...\n"

            )

    print("\n")

    print("=" * 60)

    print("Finished!")

    print("=" * 60)

    print(

        f"Chunks Processed : {processed}"

    )

    print(

        f"Chunks Skipped  : {skipped}"

    )

    print(

        f"Examples Saved  : {total_examples}"

    )

    print(

        f"Dataset File    : "

        f"{DATASET_DIR / 'training_dataset.jsonl'}"

    )

    print("=" * 60)


if __name__ == "__main__":

    main()