"""PyCharm: right-click -> Run to build/resume the complete 1024-world library."""
import argparse
import multiprocessing
import sys
from dataclasses import replace
from navlab.config import load_config
from navlab.map import MapSpec
from navlab.library import build_library


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force",action="store_true")
    parser.add_argument("--workers",type=int)
    args = parser.parse_args()
    cfg = load_config()
    if args.workers is not None:
        cfg = replace(cfg,workers=args.workers).validate()
    library = build_library(MapSpec(),cfg,force=args.force)
    print("Saved:",library["folder"])


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
