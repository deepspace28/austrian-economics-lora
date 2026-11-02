"""Command-line validation for generated instruction-tuning JSONL."""

import argparse
from pathlib import Path
import sys

from src.dataset_validator import validate_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="path to a generated JSONL file")
    parser.add_argument(
        "--max-issues",
        type=int,
        default=20,
        help="maximum validation errors to print (default: 20)",
    )
    args = parser.parse_args(argv)

    try:
        result = validate_dataset(args.dataset, max_issues=args.max_issues)
    except OSError as error:
        print(f"Unable to read {args.dataset}: {error}", file=sys.stderr)
        return 2

    print(f"Checked {result.records} record(s): {result.valid} valid.")
    if result.ok:
        print("Dataset is valid.")
        return 0

    print(f"Dataset is invalid: {result.records - result.valid} invalid record(s).")
    for issue in result.issues:
        location = f"line {issue.line_number}" if issue.line_number else "dataset"
        print(f"- {location}: {issue.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
