"""Build/resume the 1024 time-matched Probe response streams (no windows)."""
import multiprocessing
import argparse
from pathlib import Path
from navlab.config import load_config
from navlab.map import MapSpec
from navlab.library import build_library
from navlab.probe_stream import build_probe_stream


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", action="store_true", help="build the five configured force-profile streams")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.profiles:
        from navlab.profile_comparison import load_profile_settings, run_profile_comparison
        from navlab.config import ROOT
        settings, cfg, profiles = load_profile_settings(args.config)
        output = args.output
        if output is not None and not output.is_absolute():
            output = ROOT/output
        print("Saved:", run_profile_comparison(settings, cfg, profiles, output, library_only=True))
    else:
        if args.output is not None:
            parser.error("--output requires --profiles")
        cfg = load_config(args.config)
        library = build_library(MapSpec(),cfg)
        stream = build_probe_stream(MapSpec(),cfg,library)
        print("Saved:",stream["path"])
