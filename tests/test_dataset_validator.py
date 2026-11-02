import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset_validator import validate_dataset, validate_sample


class TestValidateSample(unittest.TestCase):
    def test_accepts_instruction_response_schema(self):
        self.assertEqual(
            validate_sample({"instruction": "Explain money", "response": "A medium of exchange."}),
            [],
        )

    def test_accepts_messages_schema(self):
        sample = {
            "messages": [
                {"role": "system", "content": "Answer clearly."},
                {"role": "user", "content": "What is scarcity?"},
                {"role": "assistant", "content": "Scarcity means resources are limited."},
            ]
        }
        self.assertEqual(validate_sample(sample), [])

    def test_rejects_missing_response_and_bad_final_role(self):
        sample = {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "user", "content": "Still not an answer"},
            ]
        }
        errors = validate_sample(sample)
        self.assertIn("last message must have role 'assistant'", errors)

    def test_rejects_unknown_schema(self):
        self.assertEqual(
            validate_sample({"question": "What is money?"}),
            ["expected 'messages' or 'instruction'/'response' schema"],
        )


class TestValidateDataset(unittest.TestCase):
    def test_reports_bad_json_and_invalid_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"instruction": "Q", "response": "A"}),
                        "not json",
                        json.dumps({"instruction": "Q"}),
                    ]
                ),
                encoding="utf-8",
            )
            result = validate_dataset(path)
            self.assertEqual(result.records, 3)
            self.assertEqual(result.valid, 1)
            self.assertFalse(result.ok)
            self.assertEqual([issue.line_number for issue in result.issues], [2, 3])

    def test_empty_dataset_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            path.write_text("\n", encoding="utf-8")
            result = validate_dataset(path)
            self.assertFalse(result.ok)
            self.assertEqual(result.issues[0].message, "dataset contains no records")


if __name__ == "__main__":
    unittest.main()
