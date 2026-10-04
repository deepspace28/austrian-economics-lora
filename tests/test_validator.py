import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from validator import is_valid_chunk


VALID = (
    "Inflation is a monetary phenomenon. When the supply of money grows faster "
    "than the supply of goods and services, prices rise across the board. "
    "This paragraph is long enough to pass the minimum word count and is full "
    "of ordinary prose with no furniture that a validator should reject. "
    "Added sentence number one keeps going and going with plain economic prose. "
    "Added sentence number two keeps going and going with plain economic prose. "
    "Added sentence number three keeps going and going with plain economic prose. "
    "Added sentence number four keeps going and going with plain economic prose. "
    "Added sentence number five keeps going and going with plain economic prose."
)


class TestIsValidChunk(unittest.TestCase):
    def test_empty_and_none_rejected(self):
        self.assertFalse(is_valid_chunk(""))
        self.assertFalse(is_valid_chunk(None))

    def test_short_chunk_rejected(self):
        self.assertFalse(is_valid_chunk("too short to keep"))

    def test_contents_page_rejected(self):
        self.assertFalse(is_valid_chunk("CONTENTS " + VALID))

    def test_copyright_page_rejected(self):
        self.assertFalse(is_valid_chunk("COPYRIGHT 2020 " + VALID))

    def test_isbn_page_rejected(self):
        self.assertFalse(is_valid_chunk("ISBN 978-0-1234 " + VALID))

    def test_mostly_numeric_chunk_rejected(self):
        chunk = " ".join(["123 456 789 0"] * 40)
        self.assertFalse(is_valid_chunk(chunk))

    def test_genuine_prose_accepted(self):
        self.assertTrue(is_valid_chunk(VALID))


if __name__ == "__main__":
    unittest.main()
