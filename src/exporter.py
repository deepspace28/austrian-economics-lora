import json
from pathlib import Path


class DatasetExporter:

    def __init__(self, output_file):

        self.output_file = Path(output_file)

    def save(self, examples):

        # The output directory does not exist on a first run (datasets/ holds
        # only tracked JSON, and the .jsonl it receives is gitignored), so
        # opening the file here raised FileNotFoundError and the pipeline
        # reported the chunk as failed.
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(
            self.output_file,
            "a",
            encoding="utf-8"
        ) as f:

            for example in examples:

                json.dump(
                    example,
                    f,
                    ensure_ascii=False
                )

                f.write("\n")