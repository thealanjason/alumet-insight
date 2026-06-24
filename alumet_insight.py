"""Alumet Insight — unified entry point.

Usage:
    python alumet_insight.py dashboard
    python alumet_insight.py cli /path/to/measurements --summary
    python alumet_insight.py cli /path/to/measurements --export-csv /path/to/output
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alumet Insight: interactive dashboard or command-line analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    subparsers.add_parser("dashboard", help="Launch the interactive Dash web dashboard")

    cli_parser = subparsers.add_parser("cli", help="Command-line measurement analysis")
    cli_parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the CLI tool")

    parsed = parser.parse_args()

    if parsed.command == "dashboard":
        from frontend.app import app
        from frontend.layout import create_layout
        import frontend.panes  # registers all @app.callback decorators

        app.layout = create_layout(app)
        app.run(debug=True, host="0.0.0.0", port=8051)

    elif parsed.command == "cli":
        from cli import main as cli_main

        cli_main(parsed.args if parsed.args else None)


if __name__ == "__main__":
    main()
