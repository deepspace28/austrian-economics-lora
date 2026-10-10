import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from semantic_chunker import build_chunks, split_paragraphs


def word_count(text: str) -> int:
    return len(text.split())


class TestSplitParagraphs(unittest.TestCase):
    def test_splits_on_blank_lines(self):
        self.assertEqual(split_paragraphs("one\n\ntwo\n\n\nthree"), ["one", "two", "three"])

    def test_strips_and_drops_empty(self):
        self.assertEqual(split_paragraphs("  one  \n\n   \n\n two "), ["one", "two"])


class TestBuildChunks(unittest.TestCase):
    def test_short_text_single_chunk(self):
        paragraphs = ["alpha beta", "gamma delta"]
        chunks = build_chunks(paragraphs, count_fn=word_count)
        self.assertEqual(chunks, ["alpha beta\n\ngamma delta"])

    def test_oversized_first_paragraph_yields_no_empty_chunk(self):
        big = "word " * 900
        chunks = build_chunks([big.strip()], count_fn=word_count)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].strip())

    def test_respects_target_token_budget(self):
        paragraphs = [" ".join(["w"] * 500) for _ in range(4)]
        chunks = build_chunks(paragraphs, count_fn=word_count)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            # Each chunk should stay near the 800-token target (one paragraph over is allowed).
            self.assertLessEqual(word_count(chunk), 1500)

    def test_overlap_carries_previous_paragraphs(self):
        paragraphs = [" ".join(["a"] * 800), " ".join(["b"] * 100), " ".join(["c"] * 800)]
        chunks = build_chunks(paragraphs, count_fn=word_count)
        # The 100-word middle paragraph fits the 100-token overlap budget, so
        # it should be repeated at the start of the final chunk.
        self.assertTrue(chunks[-1].startswith(" ".join(["b"] * 100)))
        self.assertIn("c", chunks[-1])

    def test_blank_paragraphs_do_not_produce_a_whitespace_only_chunk(self):
        # "\n\n".join(["", ""]) is "\n\n" - a chunk holding nothing but the
        # separator. It reached disk as an empty training example, and the
        # trainer counted it as a sample with no target.
        self.assertEqual(build_chunks(["", ""], count_fn=word_count), [])
        self.assertEqual(build_chunks(["   ", "\t\n"], count_fn=word_count), [])

    def test_blank_paragraphs_are_dropped_without_disturbing_real_ones(self):
        paragraphs = ["alpha", "", "  ", "beta"]
        self.assertEqual(
            build_chunks(paragraphs, count_fn=word_count),
            ["alpha\n\nbeta"],
        )

    def test_every_returned_chunk_carries_real_text(self):
        # The invariant, stated once over a mix that used to break it.
        paragraphs = ["", " ".join(["w"] * 900), "", "", "tail", "  "]
        chunks = build_chunks(paragraphs, count_fn=word_count)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertTrue(chunk.strip(), f"empty chunk: {chunk!r}")

    def test_no_input_yields_no_chunks(self):
        self.assertEqual(build_chunks([], count_fn=word_count), [])
        self.assertEqual(build_chunks(["", "  "], count_fn=word_count), [])


if __name__ == "__main__":
    unittest.main()
