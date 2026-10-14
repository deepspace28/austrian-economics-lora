import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import main as pipeline
from exporter import DatasetExporter
from utils import list_chunks, read_text


class FakeGenerator:
    """Stands in for the model-backed generator so the loop can be tested."""

    def __init__(self, behaviour):
        self.behaviour = behaviour

    def generate_from_file(self, path):
        action = self.behaviour.get(path.name)
        if action == "raise":
            raise RuntimeError("boom")
        if action == "empty":
            return {"examples": []}
        return {"examples": [{"instruction": "q", "response": "a"}]}


class TestProgressRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original = pipeline.PROGRESS_FILE
        pipeline.PROGRESS_FILE = Path(self.tmp.name) / "progress.json"

    def tearDown(self):
        pipeline.PROGRESS_FILE = self.original
        self.tmp.cleanup()

    def test_missing_file_starts_at_zero(self):
        self.assertEqual(pipeline.load_progress(), 0)

    def test_saved_index_round_trips(self):
        pipeline.save_progress(7)
        self.assertEqual(pipeline.load_progress(), 7)

    def test_corrupt_file_refuses_to_resume(self):
        # Resuming from 0 against an appending exporter duplicates every pair
        # already on disk, so a truncated progress file must stop the run.
        pipeline.PROGRESS_FILE.write_text("{not json", encoding="utf-8")
        with self.assertRaises(SystemExit):
            pipeline.load_progress()


class TestMainLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.chunks_dir = root / "chunks"
        self.chunks_dir.mkdir()
        for name in ("c1.txt", "c2.txt", "c3.txt", "c4.txt"):
            (self.chunks_dir / name).write_text("body", encoding="utf-8")

        self.originals = (
            pipeline.PROGRESS_FILE,
            pipeline.DATASET_DIR,
            pipeline.list_chunks,
        )
        pipeline.PROGRESS_FILE = root / "progress.json"
        pipeline.DATASET_DIR = root / "datasets"
        pipeline.list_chunks = lambda: sorted(self.chunks_dir.glob("*.txt"))

    def tearDown(self):
        (
            pipeline.PROGRESS_FILE,
            pipeline.DATASET_DIR,
            pipeline.list_chunks,
        ) = self.originals
        self.tmp.cleanup()

    def _run(self, behaviour):
        # main() imports DatasetGenerator inside the function, so patch the name
        # where it is looked up rather than on the pipeline module.
        fake_module = types.ModuleType("dataset_generator")
        fake_module.DatasetGenerator = lambda *a, **k: FakeGenerator(behaviour)
        with mock.patch.dict(sys.modules, {"dataset_generator": fake_module}):
            pipeline.main()

    def _dataset(self):
        path = pipeline.DATASET_DIR / "training_dataset.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_all_chunks_written_once(self):
        self._run({})
        self.assertEqual(len(self._dataset()), 4)
        self.assertEqual(pipeline.load_progress(), 4)

    def test_failed_chunk_does_not_block_later_chunks(self):
        self._run({"c2.txt": "raise"})
        rows = self._dataset()
        self.assertEqual(len(rows), 3)
        self.assertEqual(pipeline.load_progress(), 4)

    def test_resume_does_not_duplicate_written_pairs(self):
        self._run({"c2.txt": "raise"})
        pipeline.PROGRESS_FILE.unlink()
        pipeline.save_progress(3)          # chunks 1-3 already on disk
        self._run({})
        self.assertEqual(len(self._dataset()), 4)

    def test_completed_run_is_a_no_op(self):
        self._run({})
        before = (pipeline.DATASET_DIR / "training_dataset.jsonl").read_text(encoding="utf-8")
        self._run({})
        after = (pipeline.DATASET_DIR / "training_dataset.jsonl").read_text(encoding="utf-8")
        self.assertEqual(before, after)


class TestListChunks(unittest.TestCase):
    def test_missing_directory_returns_empty(self):
        # The old version printed "Folder exists: False" and then carried on;
        # an absent chunks/ dir is a clean empty result, not a crash.
        import utils

        original = utils.CHUNKS_DIR
        utils.CHUNKS_DIR = Path(tempfile.gettempdir()) / "definitely-not-here-9f3a"
        try:
            self.assertEqual(utils.list_chunks(), [])
        finally:
            utils.CHUNKS_DIR = original


class TestExporter(unittest.TestCase):
    def test_appends_one_json_object_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            exporter = DatasetExporter(path)
            exporter.save([{"instruction": "a", "response": "b"}])
            exporter.save([{"instruction": "c", "response": "d"}])
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["instruction"], "a")


if __name__ == "__main__":
    unittest.main()