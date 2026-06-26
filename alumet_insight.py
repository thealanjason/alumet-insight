"""Alumet Insight — unified entry point.

Usage:
  - Dashboard:
    python alumet_insight.py dashboard

  - CLI:
    python alumet_insight.py cli -h
    python alumet_insight.py cli /path/to/measurements --summary
    python alumet_insight.py cli /path/to/measurements --list-metric-ids --category energy
    python alumet_insight.py cli /path/to/measurements --list-metric-ids --metric-name rapl_consumed_energy_J --limit 50
    python alumet_insight.py cli /path/to/measurements --export-csv /path/to/output --category energy
    python alumet_insight.py cli /path/to/measurements --export-figures /path/to/output --category energy
    python alumet_insight.py cli /path/to/measurements --export-csv /path/to/output --metric-id <metric_id>
    python alumet_insight.py cli /path/to/measurements --export-figures /path/to/output --start-time 2024-01-01T00:00:00 --end-time 2024-01-01T00:01:00
    python alumet_insight.py cli /path/to/measurements --export-csv /path/to/output --process-specific

Exports are written under /path/to/output/<measurement-folder-name>/.
Run ``python alumet_insight.py cli -h`` for full flag reference and workflows.
"""

import argparse
import sys


def _cli_help() -> bool:
    if len(sys.argv) >= 3 and sys.argv[1] == "cli" and sys.argv[2] in ("-h", "--help"):
        from cli import main as cli_main
        cli_main(["--help"])
        return True
    return False


def main() -> None:
    if _cli_help():
        return

    parser = argparse.ArgumentParser(
        description="Alumet Insight: interactive dashboard or command-line analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    subparsers.add_parser("dashboard", help="Launch the interactive Dash web dashboard")

    cli_parser = subparsers.add_parser(
        "cli",
        help="Command-line measurement analysis",
        add_help=False,
    )
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

        args = list(parsed.args)
        if args and args[0] in ("--", "-"):
            parser.error(
                "Unexpected '-' or '--' after 'cli'. "
                "Pass the measurement directory directly, e.g. "
                "'python alumet_insight.py cli /path/to/measurements --summary'. "
                "Use 'python alumet_insight.py cli -h' for help."
            )
        cli_main(args if args else None)


if __name__ == "__main__":
    main()
