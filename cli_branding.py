"""
CLI banner for Alumet Insight.

Block art from oh-my-logo (CC0)::

    npx oh-my-logo "ALUMET-INSIGHT" fire --filled --no-color      # wide
    npx oh-my-logo "ALUMET\\nINSIGHT" fire --filled --no-color    # stacked

Wide (single-line) and stacked (``ALUMET`` / ``INSIGHT``) variants are chosen
from terminal width in ``print_logo()``. ``LOGO_ANSI`` keeps the wide variant
for backwards compatibility.

Uses standard 24-bit ANSI escape codes (``\\033[38;2;…``). Supported in modern
terminals on Linux, macOS, Windows Terminal, and WSL.
"""

from __future__ import annotations

import shutil
import sys
from typing import TextIO

# oh-my-logo: npx oh-my-logo "ALUMET-INSIGHT" fire --filled --no-color (wide)
LOGO_PLAIN_WIDE = """\
  █████╗  ██╗      ██╗   ██╗ ███╗   ███╗ ███████╗ ████████╗        ██╗ ███╗   ██╗ ███████╗ ██╗  ██████╗  ██╗  ██╗ ████████╗
 ██╔══██╗ ██║      ██║   ██║ ████╗ ████║ ██╔════╝ ╚══██╔══╝        ██║ ████╗  ██║ ██╔════╝ ██║ ██╔════╝  ██║  ██║ ╚══██╔══╝
 ███████║ ██║      ██║   ██║ ██╔████╔██║ █████╗      ██║    █████╗ ██║ ██╔██╗ ██║ ███████╗ ██║ ██║  ███╗ ███████║    ██║
 ██╔══██║ ██║      ██║   ██║ ██║╚██╔╝██║ ██╔══╝      ██║    ╚════╝ ██║ ██║╚██╗██║ ╚════██║ ██║ ██║   ██║ ██╔══██║    ██║
 ██║  ██║ ███████╗ ╚██████╔╝ ██║ ╚═╝ ██║ ███████╗    ██║           ██║ ██║ ╚████║ ███████║ ██║ ╚██████╔╝ ██║  ██║    ██║
 ╚═╝  ╚═╝ ╚══════╝  ╚═════╝  ╚═╝     ╚═╝ ╚══════╝    ╚═╝           ╚═╝ ╚═╝  ╚═══╝ ╚══════╝ ╚═╝  ╚═════╝  ╚═╝  ╚═╝    ╚═╝"""

# oh-my-logo: npx oh-my-logo "ALUMET\nINSIGHT" fire --filled --no-color
LOGO_PLAIN_STACKED = """\
  █████╗  ██╗      ██╗   ██╗ ███╗   ███╗ ███████╗ ████████╗
 ██╔══██╗ ██║      ██║   ██║ ████╗ ████║ ██╔════╝ ╚══██╔══╝
 ███████║ ██║      ██║   ██║ ██╔████╔██║ █████╗      ██║
 ██╔══██║ ██║      ██║   ██║ ██║╚██╔╝██║ ██╔══╝      ██║
 ██║  ██║ ███████╗ ╚██████╔╝ ██║ ╚═╝ ██║ ███████╗    ██║
 ╚═╝  ╚═╝ ╚══════╝  ╚═════╝  ╚═╝     ╚═╝ ╚══════╝    ╚═╝

 ██╗ ███╗   ██╗ ███████╗ ██╗  ██████╗  ██╗  ██╗ ████████╗
 ██║ ████╗  ██║ ██╔════╝ ██║ ██╔════╝  ██║  ██║ ╚══██╔══╝
 ██║ ██╔██╗ ██║ ███████╗ ██║ ██║  ███╗ ███████║    ██║
 ██║ ██║╚██╗██║ ╚════██║ ██║ ██║   ██║ ██╔══██║    ██║
 ██║ ██║ ╚████║ ███████║ ██║ ╚██████╔╝ ██║  ██║    ██║
 ╚═╝ ╚═╝  ╚═══╝ ╚══════╝ ╚═╝  ╚═════╝  ╚═╝  ╚═╝    ╚═╝"""

LOGO_COMPACT = "Alumet-Insight: analyze experiments faster"

# Backwards-compatible alias (wide variant).
LOGO_PLAIN = LOGO_PLAIN_WIDE

# oh-my-logo "fire" palette, horizontal gradient (default filled mode direction)
_FIRE_GRADIENT = ("#ff0844", "#ffb199")

# Margin added to measured art width so the logo fits without terminal wrapping.
_WIDTH_MARGIN = 2


def _art_width(plain: str) -> int:
    return max((len(line) for line in plain.splitlines()), default=0)


_WIDE_MIN_COLUMNS = _art_width(LOGO_PLAIN_WIDE) + _WIDTH_MARGIN
_STACKED_MIN_COLUMNS = _art_width(LOGO_PLAIN_STACKED) + _WIDTH_MARGIN


def _enable_windows_vt_processing() -> None:
    """Enable ANSI colors in legacy Windows consoles when possible."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        for handle_id in (-11, -12):  # stdout, stderr
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                enable_vt = 0x0004
                kernel32.SetConsoleMode(handle, mode.value | enable_vt)
    except (AttributeError, OSError, ImportError):
        pass


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _lerp_rgb(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    return tuple(int(a + (b - a) * t) for a, b in zip(start, end))


def _ansi_fg(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


def _apply_fire_gradient(plain: str) -> str:
    """Apply oh-my-logo fire palette with horizontal gradient."""
    lines = plain.splitlines()
    width = _art_width(plain)
    start = _hex_to_rgb(_FIRE_GRADIENT[0])
    end = _hex_to_rgb(_FIRE_GRADIENT[1])
    span = max(width - 1, 1)

    colored_lines: list[str] = []
    for line in lines:
        parts: list[str] = []
        for x, char in enumerate(line):
            if char == " ":
                parts.append(char)
                continue
            rgb = _lerp_rgb(start, end, x / span)
            parts.append(f"{_ansi_fg(*rgb)}{char}")
        colored_lines.append("".join(parts) + "\033[0m")
    return "\n".join(colored_lines)


def _terminal_columns() -> int:
    try:
        return shutil.get_terminal_size(fallback=(_WIDE_MIN_COLUMNS, 24)).columns
    except OSError:
        return _WIDE_MIN_COLUMNS


def select_logo_plain(columns: int) -> str:
    if columns >= _WIDE_MIN_COLUMNS:
        return LOGO_PLAIN_WIDE
    if columns >= _STACKED_MIN_COLUMNS:
        return LOGO_PLAIN_STACKED
    return LOGO_COMPACT


def logo_ansi_for_columns(columns: int) -> str:
    """Return the fire-gradient banner sized for the given terminal width."""
    return _apply_fire_gradient(select_logo_plain(columns))


_enable_windows_vt_processing()

LOGO_ANSI = logo_ansi_for_columns(_WIDE_MIN_COLUMNS)


def print_logo(file: TextIO | None = None, *, columns: int | None = None) -> None:
    """Print the fire-gradient CLI banner, sized to the terminal width."""
    stream = file if file is not None else sys.stderr
    width = columns if columns is not None else _terminal_columns()
    print(logo_ansi_for_columns(width), file=stream)
    print(file=stream)
