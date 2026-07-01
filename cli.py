"""Command-line analysis for Alumet measurements.

Examples:
    python alumet_insight.py cli /path/to/measurements --summary
    python alumet_insight.py cli /path/to/measurements --export-csv /path/to/output
    python alumet_insight.py cli /path/to/measurements --export-figures /path/to/output --category energy
    python alumet_insight.py cli /path/to/measurements --export-csv /path/to/output --process-specific

Exports are written under /path/to/output/<measurement-folder-name>/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.categories import CATEGORY_VALUES
from backend.cli_export import export_csvs, export_figures, summary
from backend.data import AlumetData
from backend.figures import SUPPORTED_FIGURE_FORMATS


def _measurement_output_root(output_dir: str | Path, measurement_dir: str | Path) -> Path:
    """Return output_dir/<last measurement subfolder name>."""
    measurement_name = Path(measurement_dir).expanduser().resolve().name
    return Path(output_dir).expanduser() / measurement_name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Alumet measurement analysis: summary, CSV export, and static time-series figure export.",
    )
    parser.add_argument("directory", help="Path to measurement directory containing .csv and optional .log/.txt files")
    parser.add_argument("--summary", action="store_true", help="Print a human-readable text summary")
    parser.add_argument(
        "--export-csv",
        type=str,
        metavar="DIR",
        help="Export category CSV files under DIR/<measurement-folder-name>/<category>/csv/",
    )
    parser.add_argument(
        "--export-figures",
        type=str,
        metavar="DIR",
        help="Export one static figure per metric_id under DIR/<measurement-folder-name>/<category>/plots/",
    )
    parser.add_argument(
        "--category",
        choices=CATEGORY_VALUES,
        default=None,
        help="Dashboard time-series category to export (default: all available categories)",
    )
    parser.add_argument("--cpu-core", type=str, default=None, help="CPU core filter for the kernel_cpu_time category")
    parser.add_argument(
        "--process-specific",
        action="store_true",
        help="Filter exported CSVs and figures to the measured process active time range",
    )
    parser.add_argument(
        "--figure-format",
        choices=SUPPORTED_FIGURE_FORMATS,
        default="png",
        help="Static figure format for --export-figures",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster figure exports")

    args = parser.parse_args(argv)
    data = AlumetData(args.directory)

    ran_action = False

    if args.summary:
        print(summary(data))
        ran_action = True

    if args.export_csv:
        out = _measurement_output_root(args.export_csv, args.directory)
        out.mkdir(parents=True, exist_ok=True)
        created = export_csvs(
            data,
            out,
            category=args.category,
            cpu_core=args.cpu_core,
            process_specific=args.process_specific,
        )
        print(f"Exported {len(created)} CSV file(s) under {out}")
        ran_action = True

    if args.export_figures:
        out = _measurement_output_root(args.export_figures, args.directory)
        out.mkdir(parents=True, exist_ok=True)
        created = export_figures(
            data,
            out,
            category=args.category,
            cpu_core=args.cpu_core,
            figure_format=args.figure_format,
            dpi=args.dpi,
            process_specific=args.process_specific,
        )
        print(f"Exported {len(created)} figure file(s) under {out}")
        ran_action = True

    if not ran_action:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
