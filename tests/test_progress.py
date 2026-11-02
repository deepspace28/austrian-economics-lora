import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from progress import load_progress, save_progress


class TestProgress(unittest.TestCase):
    def test_missing_checkpoint_starts_at_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_progress(Path(directory) / "progress.json"), 0)

    def test_malformed_checkpoint_starts_at_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(load_progress(path), 0)

    def test_invalid_or_out_of_range_values_are_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            path.write_text(json.dumps({"last_chunk": -1}), encoding="utf-8")
            self.assertEqual(load_progress(path), 0)

            path.write_text(json.dumps({"last_chunk": 99}), encoding="utf-8")
            self.assertEqual(load_progress(path, total=12), 12)

            path.write_text(json.dumps({"last_chunk": True}), encoding="utf-8")
            self.assertEqual(load_progress(path), 0)

    def test_save_and_load_are_round_trip_and_leave_no_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "progress.json"
            save_progress(7, path)
            self.assertEqual(load_progress(path), 7)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_save_rejects_negative_or_boolean_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            with self.assertRaises(ValueError):
                save_progress(-1, path)
            with self.assertRaises(ValueError):
                save_progress(True, path)


if __name__ == "__main__":
    unittest.main()
