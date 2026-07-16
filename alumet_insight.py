"""Alumet Insight — unified entry point.

Usage:
    python alumet_insight.py dashboard
    python alumet_insight.py cli /path/to/measurements --summary
    python alumet_insight.py cli /path/to/measurements --export-csv /path/to/output
"""

import argparse
import sys


def main() -> None:
    # Dispatch CLI before argparse so -h/--help reaches cli.py.
    # argparse cannot forward unrecognized options like --help via REMAINDER.
    if len(sys.argv) >= 2 and sys.argv[1] == "cli":
        from cli import main as cli_main

        forwarded = sys.argv[2:]
        # Empty argv: avoids parse_args(None) treating "cli" as a directory path.
        cli_main(forwarded if forwarded else ["--help"])
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    subparsers.add_parser("dashboard", help="Interactive dashboard for measurement analysis")
    subparsers.add_parser("cli", help="Command-line interface for measurement analysis")

    parsed = parser.parse_args()

    if parsed.command == "dashboard":
        from frontend.app import app
        from frontend.layout import create_layout
        import frontend.panes  # registers all @app.callback decorators

        app.layout = create_layout(app)
        app.run(debug=True, host="0.0.0.0", port=8051)


if __name__ == "__main__":
    main()
