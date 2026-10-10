import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cleaner import clean_text


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
