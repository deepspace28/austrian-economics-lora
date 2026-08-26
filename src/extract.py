import json


class DatasetExporter:

    def __init__(self, output_file):

        self.output_file = output_file

    def save(self, examples):

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