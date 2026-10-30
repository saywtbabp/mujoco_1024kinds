"""Build force-profile streams, run A/B/C/E, or regenerate the offline report."""
import argparse
import multiprocessing
from pathlib import Path
from navlab.config import ROOT
from navlab.profile_comparison import load_profile_settings, run_profile_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", nargs="+", choices=["A", "B", "C", "E"])
    parser.add_argument("--library-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    output = args.output
    if output is not None and not output.is_absolute():
        output = ROOT/output
    if args.report_only:
        if output is None or args.library_only:
            parser.error("--report-only requires --output and excludes --library-only")
        from navlab.profile_report import validate_profile_saved, generate_profile_report
        validate_profile_saved(output)
        for answer in generate_profile_report(output):
            print(answer, flush=True)
    else:
        settings, cfg, profiles = load_profile_settings(args.config)
        if args.case:
            settings["cases"] = args.case
        output = run_profile_comparison(settings, cfg, profiles, output, args.library_only)
    print("Saved:", output if args.library_only else output/"report.html", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
