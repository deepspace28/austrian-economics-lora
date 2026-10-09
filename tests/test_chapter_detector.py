import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chapter_detector import detect_chapters


class TestDetectChapters(unittest.TestCase):
    def test_no_match_falls_back_to_single_chapter(self):
        chapters = detect_chapters("Some book with no chapter headings.")
        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0]["title"], "Full Text")
        self.assertEqual(chapters[0]["content"], "Some book with no chapter headings.")

    def test_numeric_chapters_split(self):
        text = "Preface.\nCHAPTER 1 The Origins\nContent one.\nCHAPTER 2 The Decline\nContent two."
        chapters = detect_chapters(text)
        self.assertEqual(
            [c["title"] for c in chapters],
            ["Preamble", "CHAPTER 1 The Origins", "CHAPTER 2 The Decline"],
        )
        self.assertIn("Preface.", chapters[0]["content"])
        self.assertIn("Content one.", chapters[1]["content"])
        self.assertIn("Content two.", chapters[2]["content"])

    def test_roman_numerals_and_part(self):
        text = "Intro.\nPART I\nAlpha.\nCHAPTER IV\nBeta."
        chapters = detect_chapters(text)
        self.assertEqual(len(chapters), 3)
        self.assertEqual(chapters[0]["title"], "Preamble")
        self.assertEqual(chapters[1]["title"], "PART I")
        self.assertEqual(chapters[2]["title"], "CHAPTER IV")
        self.assertIn("Alpha.", chapters[1]["content"])
        self.assertIn("Beta.", chapters[2]["content"])

    def test_last_chapter_without_trailing_newline(self):
        text = "CHAPTER 1 One\nbody\nCHAPTER 2 Two"
        chapters = detect_chapters(text)
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[1]["title"], "CHAPTER 2 Two")

    def test_lowercase_chapter_not_matched(self):
        text = "chapter 1 informal heading\nbody"
        chapters = detect_chapters(text)
        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0]["title"], "Full Text")

    def test_preamble_before_first_heading_is_preserved(self):
        text = "This preface argues from first principles.\nCHAPTER 1 Origins\nBody one."
        chapters = detect_chapters(text)
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["title"], "Preamble")
        self.assertIn("first principles", chapters[0]["content"])
        self.assertEqual(chapters[1]["title"], "CHAPTER 1 Origins")

    def test_whitespace_only_preamble_not_emitted(self):
        text = "\n\n\nCHAPTER 1 Origins\nBody one."
        chapters = detect_chapters(text)
        self.assertEqual([c["title"] for c in chapters], ["CHAPTER 1 Origins"])

    def test_non_roman_word_after_part_not_matched(self):
        # "ITERATION" is not a roman numeral even though every letter is in IVXLCDM.
        for word in ("ITERATION", "MIXED", "LIVELY", "CIVIC", "DID", "MIMI"):
            text = f"PART {word} of the rules\nbody"
            chapters = detect_chapters(text)
            self.assertEqual(len(chapters), 1, f"{word} should not be a chapter heading")
            self.assertEqual(chapters[0]["title"], "Full Text")

    def test_canonical_roman_numerals_still_matched(self):
        for numeral in ("I", "IV", "IX", "XIV", "XL", "XC", "CD", "CM", "MMXXIV"):
            text = f"PART {numeral}\nbody"
            chapters = detect_chapters(text)
            self.assertEqual(len(chapters), 1, f"{numeral} should be a heading")
            self.assertEqual(chapters[0]["title"], f"PART {numeral}")


if __name__ == "__main__":
    unittest.main()
