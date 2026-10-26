"""Build/resume multi-Probe predictions, run Cases, or regenerate the report."""
import argparse
import multiprocessing
from pathlib import Path
from navlab.multi_probe import load_settings, run_comparison
from navlab.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="comparison JSON; default probe_comparison_config.json")
    parser.add_argument("--output", type=Path, help="new directory or existing comparison to resume; paths relative to project")
    parser.add_argument("--case", nargs="+", choices=["A", "B", "C", "E"], help="Case subset; use a new output when changing it")
    parser.add_argument("--library-only", action="store_true", help="build/resume all candidate response caches only")
    parser.add_argument("--report-only", action="store_true", help="validate saved data and regenerate HTML; requires --output")
    args = parser.parse_args()
    output = args.output
    if output is not None and not output.is_absolute():
        output = ROOT/output
    if args.report_only:
        if output is None or args.library_only:
            parser.error("--report-only requires --output and excludes --library-only")
        from navlab.multi_probe_report import generate_report, validate_saved
        validate_saved(output)
        generate_report(output)
    else:
        settings, cfg = load_settings(args.config)
        if args.case:
            settings["cases"] = args.case
        output = run_comparison(settings, cfg, output, args.library_only)
    print("Saved:", output if args.library_only else output/"report.html", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
