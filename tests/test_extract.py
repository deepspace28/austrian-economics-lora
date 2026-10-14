import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import fitz

    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

from extract import extract_file, extract_text


@unittest.skipUnless(HAS_FITZ, "PyMuPDF not installed")
class TestExtract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pdf_path = Path(self.tmp.name) / "sample.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Human action is purposeful behavior.")
        doc.save(self.pdf_path)
        doc.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_extract_text_roundtrip(self):
        text, empty_pages = extract_text(self.pdf_path)
        self.assertIn("Human action is purposeful behavior.", text)
        self.assertEqual(empty_pages, 0)

    def test_extract_file_writes_txt(self):
        out = Path(self.tmp.name) / "out" / "sample.txt"
        n = extract_file(self.pdf_path, out)
        self.assertTrue(out.exists())
        self.assertGreater(n, 0)
        self.assertIn("Human action", out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
