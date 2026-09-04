"""Command-line analysis for Alumet measurements.

Examples:
    python alumet_insight.py cli /path/to/measurements --summary
    python alumet_insight.py cli /path/to/measurements --list-metric-ids --category energy
    python alumet_insight.py cli /path/to/measurements --export-csv /path/to/output
    python alumet_insight.py cli /path/to/measurements --export-figures /path/to/output --category energy
    python alumet_insight.py cli /path/to/measurements --export-csv /path/to/output --process-specific
    python alumet_insight.py cli DIR --metric-id ID_X --compare-metric-id ID_Y --export-csv OUT
    python alumet_insight.py cli DIR --metric-id ID_X --compare-metric-id ID_Y --export-figures OUT
    python alumet_insight.py cli DIR --metric-id ID_X --compare-metric-id ID_Y --export-figures OUT --scatter

Exports are written under /path/to/output/<measurement-folder-name>/.
Compare files go under that folder's comparative/csv/ and comparative/plots/.
Run ``python alumet_insight.py cli -h`` for full flag reference and workflows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.categories import CATEGORY_VALUES, validate_metric_id_in_category
from backend.cli_export import (
    build_metric_id_listing,
    comparative_mode,
    export_comparative_csv,
    export_comparative_figure,
    export_csvs,
    export_figures,
    summary,
)
from backend.data import AlumetData
from backend.figures import SUPPORTED_FIGURE_FORMATS
from backend.transforms import parse_timestamp, validate_time_range
from cli_branding import print_logo

CLI_EPILOG = """\
workflows:
  overview            --summary
  find metric IDs     --list-metric-ids [--category CAT] [--metric-name NAME] [--limit N]
  one series          --export-csv|figures OUT_DIR --metric-id ID
  compare pair        --export-csv|figures OUT_DIR --metric-id ID_X --compare-metric-id ID_Y [--scatter]
                      --scatter with --export-figures for aligned interval X vs Y
                      files: <out>/<measurement>/comparative/csv/ and comparative/plots/
  whole category      --export-csv|figures OUT_DIR [--category CAT]
  time window         --start-time / --end-time on exports
  process window      --process-specific

By default, without additional specified arguments, export all series (For figures, one file per series; for CSV, one file per category).
Comparative analysis requires an explicit pair. CSV and figures match the Comparative
tab (Download CSV / current plot). Interval-delta pairs are running totals;
power/gauges stay a dual-axis time series.
"""

TIMESTAMP_ARG_HELP = (
    "Inclusive {bound} timestamp for exports. Use date+time with a T, e.g. 2026-03-24T23:51:41+00:00 "
    "(check min/max timestamps from --summary)."
)


def _measurement_output_root(output_dir: str | Path, measurement_dir: str | Path) -> Path:
    """Return output_dir/<last measurement subfolder name>."""
    measurement_name = Path(measurement_dir).expanduser().resolve().name
    return Path(output_dir).expanduser() / measurement_name

class _CLIArgumentParser(argparse.ArgumentParser):
    """Print the CLI banner before argparse help text."""

    def print_help(self, file=None):
        if file is None:
            file = sys.stderr
        print_logo(file=file)
        super().print_help(file=file)

def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace, data: AlumetData) -> None:
    """Validate argument combinations before executing any action."""
    if args.metric_id and args.metric_id not in data.metric_ids:
        parser.error(f"Unknown --metric-id '{args.metric_id}'. Run --list-metric-ids to list available series.")

    if getattr(args, "compare_metric_id", None):
        if not args.metric_id:
            parser.error("--compare-metric-id requires --metric-id (X series).")
        if args.compare_metric_id not in data.metric_ids:
            parser.error(
                f"Unknown --compare-metric-id '{args.compare_metric_id}'. "
                "Run --list-metric-ids to list available series."
            )
        if args.compare_metric_id == args.metric_id:
            parser.error("--metric-id and --compare-metric-id must name two different series.")
        if args.metric_name:
            parser.error("--metric-name cannot be used with --compare-metric-id.")
        if not (args.export_csv or args.export_figures):
            parser.error("--compare-metric-id requires --export-csv and/or --export-figures.")
        if args.category:
            try:
                validate_metric_id_in_category(
                    data.processed_df, args.metric_id, args.category, selected_cpu_core=args.cpu_core
                )
                validate_metric_id_in_category(
                    data.processed_df, args.compare_metric_id, args.category, selected_cpu_core=args.cpu_core
                )
            except ValueError as exc:
                parser.error(str(exc))
        if getattr(args, "scatter", False) and not args.export_figures:
            parser.error("--scatter is only valid with --export-figures.")

    if args.metric_name and args.metric_name not in data.metrics:
        parser.error(
            f"Unknown --metric-name '{args.metric_name}'. "
            f"Available: {', '.join(data.metrics)}"
        )

    if args.metric_id and args.metric_name:
        parser.error("--metric-id and --metric-name cannot be used together; they select at different granularities.")

    if args.metric_id and args.category:
        try:
            validate_metric_id_in_category(data.processed_df, args.metric_id, args.category, selected_cpu_core=args.cpu_core)
        except ValueError as exc:
            parser.error(str(exc))

    if args.cpu_core and not (
        args.category == "kernel_cpu_time"
        or (args.metric_id and "kernel_cpu_time" in args.metric_id)
    ):
        parser.error("--cpu-core is only valid with --category kernel_cpu_time or a kernel_cpu_time metric ID.")

    if args.start_time:
        try:
            parse_timestamp(args.start_time, "--start-time")
        except ValueError as exc:
            parser.error(str(exc))

    if args.end_time:
        try:
            parse_timestamp(args.end_time, "--end-time")
        except ValueError as exc:
            parser.error(str(exc))

    if args.start_time or args.end_time:
        try:
            validate_time_range(args.start_time, args.end_time, *data.data_time_range)
        except (AssertionError, ValueError) as exc:
            parser.error(str(exc))

    has_export = args.export_csv or args.export_figures
    if args.metric_name and has_export:
        parser.error("--metric-name is for discovery only (use with --list-metric-ids). For export use --metric-id or --category.")

    if args.limit is not None and not args.list_metric_ids:
        parser.error("--limit is only valid with --list-metric-ids.")

    if getattr(args, "scatter", False) and not getattr(args, "compare_metric_id", None):
        parser.error("--scatter requires --compare-metric-id.")

def main(argv: list[str] | None = None) -> None:
    entry = Path(sys.argv[0]).name
    prog = f"{entry} cli" if entry.startswith("alumet_insight") else None
    parser = _CLIArgumentParser(
        prog=prog,
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("directory", help="Path to measurement directory containing .csv and optional .log/.txt files")

    discovery = parser.add_argument_group("discovery", "Explore the dataset before exporting")
    discovery.add_argument("--summary", action="store_true", help="Print a compact overview of the dataset")
    discovery.add_argument(
        "--list-metric-ids",
        action="store_true",
        help="List exact metric IDs for use with --metric-id. Combine with --category or --metric-name to filter.",
    )
    discovery.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of metric IDs printed by --list-metric-ids",
    )

    selectors = parser.add_argument_group("selectors", "Choose which metrics to operate on")
    selectors.add_argument(
        "--category",
        choices=CATEGORY_VALUES,
        default=None,
        help="Dashboard category filter (default: all). With --metric-id, validates the metric belongs to it.",
    )
    selectors.add_argument(
        "--metric-name",
        type=str,
        default=None,
        help="Base metric name filter for --list-metric-ids (e.g. rapl_consumed_energy_J).",
    )
    selectors.add_argument(
        "--metric-id",
        type=str,
        default=None,
        help="Export exactly one metric series (X series when comparing). Run --list-metric-ids to find IDs.",
    )
    selectors.add_argument(
        "--compare-metric-id",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "Y series for a two-metric export (X is --metric-id). Requires --export-csv "
            "and/or --export-figures. Discover IDs with --list-metric-ids. Writes one "
            "file under <out>/<measurement>/comparative/csv/ or comparative/plots/. "
        ),
    )
    selectors.add_argument("--cpu-core", type=str, default=None, help="CPU core filter (only for kernel_cpu_time)")

    time_window = parser.add_argument_group("time window", "Restrict exports to a time range")
    time_window.add_argument(
        "--process-specific",
        action="store_true",
        help="Clip to the measured process active time range",
    )
    time_window.add_argument(
        "--start-time",
        type=str,
        default=None,
        help=TIMESTAMP_ARG_HELP.format(bound="start"),
    )
    time_window.add_argument(
        "--end-time",
        type=str,
        default=None,
        help=TIMESTAMP_ARG_HELP.format(bound="end"),
    )

    export = parser.add_argument_group("export", "Write CSV or figure files")
    export.add_argument(
        "--export-csv",
        type=str,
        metavar="DIR",
        help="Export CSV files under DIR/<measurement-folder-name>/",
    )
    export.add_argument(
        "--export-figures",
        type=str,
        metavar="DIR",
        help="Export figure files under DIR/<measurement-folder-name>/",
    )
    export.add_argument(
        "--figure-format",
        choices=SUPPORTED_FIGURE_FORMATS,
        default="png",
        help="Static figure format for --export-figures",
    )
    export.add_argument("--dpi", type=int, default=300, help="DPI for raster figure exports")
    export.add_argument(
        "--scatter",
        action="store_true",
        help=(
            "With --compare-metric-id and --export-figures, plot nearest-aligned interval "
            "X-Y scatter instead of the default cumulative X-Y or dual-axis figure"
        ),
    )

    args = parser.parse_args(argv)
    data = AlumetData(args.directory)
    _validate_args(parser, args, data)

    ran_action = False

    if args.summary:
        print(summary(data))
        ran_action = True

    if args.list_metric_ids:
        print(build_metric_id_listing(
            data,
            category=args.category,
            metric_name=args.metric_name,
            cpu_core=args.cpu_core,
            limit=args.limit,
        ))
        ran_action = True

    if args.export_csv:
        out = _measurement_output_root(args.export_csv, args.directory)
        out.mkdir(parents=True, exist_ok=True)
        if args.compare_metric_id:
            created = export_comparative_csv(
                data,
                out,
                args.metric_id,
                args.compare_metric_id,
                process_specific=args.process_specific,
                start_time=args.start_time,
                end_time=args.end_time,
            )
            print(
                f"Comparative mode: {comparative_mode(args.metric_id, args.compare_metric_id)}. "
                f"Exported {len(created)} CSV file(s) under {out}"
            )
        else:
            created = export_csvs(
                data,
                out,
                category=args.category,
                cpu_core=args.cpu_core,
                process_specific=args.process_specific,
                metric_id=args.metric_id,
                start_time=args.start_time,
                end_time=args.end_time,
            )
            print(f"Exported {len(created)} CSV file(s) under {out}")
        ran_action = True

    if args.export_figures:
        if not args.category and not args.metric_id:
            series_count = len(data.metric_ids)
            print(
                f"Warning: exporting figures for all {series_count} metric series "
                f"(one file per series). This can take a while. "
                f"Use --category or --metric-id to narrow the export.",
                file=sys.stderr,
            )
        out = _measurement_output_root(args.export_figures, args.directory)
        out.mkdir(parents=True, exist_ok=True)
        if args.compare_metric_id:
            created = export_comparative_figure(
                data,
                out,
                args.metric_id,
                args.compare_metric_id,
                process_specific=args.process_specific,
                start_time=args.start_time,
                end_time=args.end_time,
                figure_format=args.figure_format,
                dpi=args.dpi,
                scatter=args.scatter,
            )
            mode = "scatter" if args.scatter else comparative_mode(args.metric_id, args.compare_metric_id)
            print(f"Comparative mode: {mode}. Exported {len(created)} figure file(s) under {out}")
        else:
            created = export_figures(
                data,
                out,
                category=args.category,
                cpu_core=args.cpu_core,
                figure_format=args.figure_format,
                dpi=args.dpi,
                process_specific=args.process_specific,
                metric_id=args.metric_id,
                start_time=args.start_time,
                end_time=args.end_time,
            )
            print(f"Exported {len(created)} figure file(s) under {out}")
        ran_action = True

    if not ran_action:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
