"""Command-line entry point for Alumet measurement analysis.

Examples:
    python alumet_eda.py /path/to/measurements --summary
    python alumet_eda.py /path/to/measurements --export-csv /path/to/output
    python alumet_eda.py /path/to/measurements --export-figures /path/to/output --category energy
    python alumet_eda.py /path/to/measurements --export-csv /path/to/output --process-specific

Exports are written under /path/to/output/<measurement-folder-name>/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.categories import CATEGORY_VALUES
from backend.data import AlumetData
from backend.visualization.static import SUPPORTED_FIGURE_FORMATS, export_category_figures


def _measurement_output_root(output_dir: str | Path, measurement_dir: str | Path) -> Path:
    """Return output_dir/<last measurement subfolder name>."""
    measurement_name = Path(measurement_dir).expanduser().resolve().name
    return Path(output_dir).expanduser() / measurement_name


def export_figures(
    data: AlumetData,
    output_root: Path,
    category: str | None = None,
    cpu_core: str | None = None,
    figure_format: str = "png",
    dpi: int = 150,
    process_specific: bool = False,
) -> list[Path]:
    """Export one static time-series figure per metric_id under output_root/<category>/plots/."""
    proc_start, proc_end = data.process_time_range
    created: list[Path] = []

    for category_value in data.selected_categories(category):
        df_category = data.filter_by_category(category_value, cpu_core=cpu_core)
        if process_specific:
            df_category = data.filter_to_process_time_range(df_category)
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
        print(data.summary())
        ran_action = True

    if args.export_csv:
        out = _measurement_output_root(args.export_csv, args.directory)
        out.mkdir(parents=True, exist_ok=True)
        created = data.export_csvs(
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
