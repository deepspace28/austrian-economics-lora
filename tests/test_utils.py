import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import read_text, write_jsonl, load_prompt


class TestUtils(unittest.TestCase):
    def test_read_text(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.txt"
            p.write_text("hello world", encoding="utf-8")
            self.assertEqual(read_text(p), "hello world")

    def test_load_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "prompt.txt"
            p.write_text("You are a helpful assistant.", encoding="utf-8")
            self.assertEqual(load_prompt(p), "You are a helpful assistant.")

    def test_write_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out" / "data.jsonl"
            write_jsonl(p, [{"a": 1}, {"b": 2}])
            lines = p.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)
            import json
            self.assertEqual(json.loads(lines[0]), {"a": 1})
            self.assertEqual(json.loads(lines[1]), {"b": 2})

    def test_write_jsonl_appends(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.jsonl"
            write_jsonl(p, [{"a": 1}])
            write_jsonl(p, [{"b": 2}])
            lines = p.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()