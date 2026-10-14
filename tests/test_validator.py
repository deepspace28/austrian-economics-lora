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
        page = (
            "CONTENTS\n\n"
            "Preface ........................................ 1\n"
            "The Human Action Concept ....................... 3\n"
            "Ecology ........................................ 21\n"
            "Liberalism ..................................... 27\n"
            "Index .......................................... 891\n"
        )
        self.assertFalse(is_valid_chunk(page))

    def test_isbn_page_rejected(self):
        page = (
            "ISBN\n"
            "978-0-615-12345-6\n"
            "Library of Congress Cataloging in Publication Data\n"
            "Copyright notice and all rights reserved asserted and\n"
            "protected by the copyright act of this country and the\n"
            "other countries of the world and by the international\n"
            "conventions that govern the international flow of\n"
            "copyrighted works of intellect and art.\n"
        )
        self.assertFalse(is_valid_chunk(page))

    def test_mostly_numeric_chunk_rejected(self):
        chunk = " ".join(["123 456 789 0"] * 40)
        self.assertFalse(is_valid_chunk(chunk))

    def test_genuine_prose_accepted(self):
        self.assertTrue(is_valid_chunk(VALID))

    # The old validator substring-matched "CONTENTS", "COPYRIGHT" and "ISBN"
    # anywhere in the chunk, so ordinary prose using those words was silently
    # dropped from the corpus.

    def test_prose_using_contents_is_kept(self):
        chunk = (
            "The contents of a savings deposit are a promise about the future, "
            "not a commodity already in hand. " + VALID
        )
        self.assertTrue(is_valid_chunk(chunk))

    def test_prose_using_copyright_is_kept(self):
        chunk = (
            "Copyright law grants a temporary monopoly on expression and shapes "
            "the market for ideas. " + VALID
        )
        self.assertTrue(is_valid_chunk(chunk))

    def test_prose_using_isbn_is_kept(self):
        chunk = (
            "An ISBN identifies one edition of one book and tells a buyer "
            "nothing about its argument. " + VALID
        )
        self.assertTrue(is_valid_chunk(chunk))

    def test_table_of_contents_heading_alone_rejected(self):
        self.assertFalse(is_valid_chunk("Table of Contents"))

    def test_publisher_imprint_line_alone_rejected(self):
        self.assertFalse(is_valid_chunk("HARPER & BROTHERS"))

    def test_title_page_rejected(self):
        chunk = (
            "HUMAN ACTION\n"
            "A TREATISE ON HUMAN ECONOMY\n\n"
            "LUDWIG VON MISES\n"
            "NEW YORK\n\n"
            "PHILOSOPHICAL LIBRARY\n\n"
            "COPYRIGHT NOTICE AND ALL RIGHTS RESERVED ARE ASSERTED AND "
            "PROTECTED BY THE COPYRIGHT ACT OF THIS COUNTRY AND THE OTHER "
            "COUNTRIES OF THE WORLD AND BY THE INTERNATIONAL CONVENTIONS THAT "
            "GOVERN THE INTERNATIONAL FLOW OF COPYRIGHTED WORKS OF INTELLECT "
            "AND ART AND THE DECISIONS OF THE COPYRIGHT HOLDERS AND PUBLISHERS "
            "THAT GOVERN THE COPYING AND DISTRIBUTION OF SUCH WORKS AND WHICH "
            "ARE HEREIN CITED AS THE \"BERNE CONVENTION\" AND WHICH ARE ALSO "
            "GOVERNED BY THE UNIVERSAL COPYRIGHT CONVENTION"
        )
        self.assertFalse(is_valid_chunk(chunk))

    def test_short_shouty_prose_is_not_treated_as_a_title_page(self):
        chunk = "INFLATION IS A MONETARY PHENOMENON! " + VALID
        self.assertTrue(is_valid_chunk(chunk))


if __name__ == "__main__":
    unittest.main()
