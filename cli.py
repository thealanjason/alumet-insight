"""Command-line entry point for Alumet measurement exports.

Examples:
    python cli.py /path/to/measurements --summary
    python cli.py /path/to/measurements --export-csv /path/to/output
    python cli.py /path/to/measurements --export-figures /path/to/output --category energy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.categories import CATEGORY_VALUES, available_category_values
from backend.data import AlumetData
from visualization.static_plotting import SUPPORTED_FIGURE_FORMATS, export_category_figures


def _selected_categories(data: AlumetData, category: str | None) -> list[str]:
    """Return the requested category, or all available categories when omitted."""
    if category:
        return [category]
    return available_category_values(data.processed_df)


def export_figures(
    data: AlumetData,
    output_root: Path,
    category: str | None = None,
    cpu_core: str | None = None,
    figure_format: str = "png",
    dpi: int = 150,
) -> list[Path]:
    """Export one static time-series figure per metric_id under output_root/<category>/plots/."""
    proc_start, proc_end = data.process_time_range
    created: list[Path] = []

    for category_value in _selected_categories(data, category):
        df_category = data.filter_by_category(category_value, cpu_core=cpu_core)
        if df_category.empty:
            continue

        plots_dir = output_root / category_value / "plots"
        created.extend(
            export_category_figures(
                df_category,
                plots_dir,
                category_value,
                figure_format=figure_format,
                proc_start=proc_start,
                proc_end=proc_end,
                dpi=dpi,
            )
        )

    return created


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Alumet measurement analysis: summary, CSV export, and static time-series figure export.",
    )
    parser.add_argument("directory", help="Path to measurement directory containing .csv and optional .log/.txt files")
    parser.add_argument("--summary", action="store_true", help="Print a human-readable text summary")
    parser.add_argument("--export-csv", type=str, metavar="DIR", help="Export category CSV files under DIR/<category>/csv/")
    parser.add_argument(
        "--export-figures",
        type=str,
        metavar="DIR",
        help="Export one static figure per metric_id under DIR/<category>/plots/",
    )
    parser.add_argument(
        "--category",
        choices=CATEGORY_VALUES,
        default=None,
        help="Dashboard time-series category to export (default: all available categories)",
    )
    parser.add_argument("--cpu-core", type=str, default=None, help="CPU core filter for the kernel_cpu_time category")
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
        print(data.summary())
        ran_action = True

    if args.export_csv:
        out = Path(args.export_csv)
        out.mkdir(parents=True, exist_ok=True)
        created = data.export_csvs(out, category=args.category, cpu_core=args.cpu_core)
        print(f"Exported {len(created)} CSV file(s) under {out}")
        ran_action = True

    if args.export_figures:
        out = Path(args.export_figures)
        out.mkdir(parents=True, exist_ok=True)
        created = export_figures(
            data,
            out,
            category=args.category,
            cpu_core=args.cpu_core,
            figure_format=args.figure_format,
            dpi=args.dpi,
        )
        print(f"Exported {len(created)} figure file(s) under {out}")
        ran_action = True

    if not ran_action:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
