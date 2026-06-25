import unittest

from backend.formatting import format_bytes_ticklabel, get_bytes_tickvals_ticktext


class FormattingTests(unittest.TestCase):
    def test_format_bytes_ticklabel(self):
        self.assertEqual(format_bytes_ticklabel(512), "512 B")
        self.assertEqual(format_bytes_ticklabel(2048), "2.0 KB")
        self.assertEqual(format_bytes_ticklabel(2048 ** 2), "4.0 MB")

    def test_get_bytes_tickvals_ticktext(self):
        tickvals, ticktext = get_bytes_tickvals_ticktext(0, 2048, num_ticks=3)
        self.assertEqual(len(tickvals), len(ticktext))
        self.assertTrue(all(val >= 0 for val in tickvals))


if __name__ == "__main__":
    unittest.main()
