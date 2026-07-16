import io
import re
import unittest

from cli_branding import (
    LOGO_ANSI,
    LOGO_COMPACT,
    LOGO_PLAIN,
    LOGO_PLAIN_STACKED,
    LOGO_PLAIN_WIDE,
    _STACKED_MIN_COLUMNS,
    _WIDE_MIN_COLUMNS,
    print_logo,
    select_logo_plain,
)

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class CLIBrandingTests(unittest.TestCase):
    def test_logo_plain_is_non_empty(self):
        lines = [line for line in LOGO_PLAIN.splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 6)
        self.assertIn("█", LOGO_PLAIN)

    def test_stacked_art_is_alumet_then_insight(self):
        blocks = LOGO_PLAIN_STACKED.split("\n\n")
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len([line for line in blocks[0].splitlines() if line.strip()]), 6)
        self.assertEqual(len([line for line in blocks[1].splitlines() if line.strip()]), 6)

    def test_thresholds_derived_from_art_width(self):
        self.assertEqual(_WIDE_MIN_COLUMNS, 125)
        self.assertEqual(_STACKED_MIN_COLUMNS, 61)

    def test_logo_ansi_uses_fire_gradient(self):
        self.assertIn("\033[38;2;", LOGO_ANSI)
        self.assertNotEqual(LOGO_ANSI, LOGO_PLAIN)

    def test_print_logo_always_uses_ansi(self):
        buf = io.StringIO()
        print_logo(file=buf, columns=130)
        self.assertIn("\033[38;2;", buf.getvalue())

    def test_wide_variant_at_full_terminal_width(self):
        self.assertIs(select_logo_plain(130), LOGO_PLAIN_WIDE)

    def test_stacked_variant_at_medium_terminal_width(self):
        self.assertIs(select_logo_plain(100), LOGO_PLAIN_STACKED)
        self.assertIs(select_logo_plain(_STACKED_MIN_COLUMNS), LOGO_PLAIN_STACKED)

    def test_compact_variant_below_stacked_threshold(self):
        self.assertEqual(select_logo_plain(_STACKED_MIN_COLUMNS - 1), LOGO_COMPACT)
        buf = io.StringIO()
        print_logo(file=buf, columns=50)
        self.assertIn(LOGO_COMPACT, _strip_ansi(buf.getvalue()))
        self.assertNotIn("█", _strip_ansi(buf.getvalue()))


if __name__ == "__main__":
    unittest.main()
