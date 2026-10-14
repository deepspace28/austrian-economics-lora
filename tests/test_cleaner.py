import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cleaner import clean_file, clean_text, output_name


class TestOutputName(unittest.TestCase):
    def test_adds_suffix(self):
        self.assertEqual(output_name("Human Action.txt"), "Human Action_clean.txt")

    def test_is_idempotent(self):
        # str.replace stacked a suffix per run: x_clean.txt -> x_clean_clean.txt
        self.assertEqual(output_name("Human Action_clean.txt"), "Human Action_clean.txt")
        self.assertEqual(
            output_name(output_name("Human Action.txt")),
            "Human Action_clean.txt",
        )

    def test_keeps_dots_in_the_stem(self):
        # filename.replace(".txt", ...) hit every occurrence, not just the
        # extension, so "Vol.1.txt" became "Vol.1_clean.txt" only by luck.
        self.assertEqual(output_name("Vol.1.txt"), "Vol.1_clean.txt")


class TestCleanFile(unittest.TestCase):
    def test_creates_missing_output_directory(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.txt"
            src.write_text("a\n\nb", encoding="utf-8")
            out = Path(tmp) / "nested" / "deeper" / "in_clean.txt"

            # clean_file writes straight to output_path, so the parent has to
            # exist or the first run of the pipeline dies here.
            out.parent.mkdir(parents=True)
            clean_file(src, out)

            self.assertEqual(out.read_text(encoding="utf-8"), "a\n\nb")


class TestCleanText(unittest.TestCase):
    def test_collapses_runs_of_spaces_and_tabs(self):
        self.assertEqual(clean_text("a   b\t\tc"), "a b c")

    def test_dehyphenates_words_split_across_lines(self):
        self.assertEqual(clean_text("econom-\nics"), "economics")

    def test_joins_wrapped_single_newlines(self):
        self.assertEqual(clean_text("line one\nline two"), "line one line two")

    def test_preserves_paragraph_breaks(self):
        self.assertEqual(clean_text("para one\n\npara two"), "para one\n\npara two")

    def test_collapses_excess_blank_lines(self):
        self.assertEqual(clean_text("a\n\n\n\n\nb"), "a\n\nb")

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(clean_text("\n\nhello\n\n"), "hello")


if __name__ == "__main__":
    unittest.main()
