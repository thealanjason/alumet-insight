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
    python alumet_insight.py cli /path/to/measurements --metric-id ID_X --compare-metric-id ID_Y --export-csv /path/to/output
    python alumet_insight.py cli /path/to/measurements --metric-id ID_X --compare-metric-id ID_Y --export-figures /path/to/output
    python alumet_insight.py cli /path/to/measurements --metric-id ID_X --compare-metric-id ID_Y --export-figures /path/to/output --scatter
    python alumet_insight.py cli /path/to/measurements --export-figures /path/to/output --start-time 2024-01-01T00:00:00 --end-time 2024-01-01T00:01:00
    python alumet_insight.py cli /path/to/measurements --export-csv /path/to/output --process-specific

Exports are written under /path/to/output/<measurement-folder-name>/.
Comparative analysis files go under that folder's comparative/csv/ and comparative/plots/.
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
