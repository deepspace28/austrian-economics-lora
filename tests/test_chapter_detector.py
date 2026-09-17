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
        self.assertEqual([c["title"] for c in chapters], ["CHAPTER 1 The Origins", "CHAPTER 2 The Decline"])
        self.assertIn("Content one.", chapters[0]["content"])
        self.assertIn("Content two.", chapters[1]["content"])

    def test_roman_numerals_and_part(self):
        text = "Intro.\nPART I\nAlpha.\nCHAPTER IV\nBeta."
        chapters = detect_chapters(text)
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["title"], "PART I")
        self.assertEqual(chapters[1]["title"], "CHAPTER IV")
        self.assertIn("Alpha.", chapters[0]["content"])
        self.assertIn("Beta.", chapters[1]["content"])

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


if __name__ == "__main__":
    unittest.main()
