"""Build/resume the 1024 time-matched Probe response streams (no windows)."""
import multiprocessing
from navlab.config import load_config
from navlab.map import MapSpec
from navlab.library import build_library
from navlab.probe_stream import build_probe_stream


if __name__ == "__main__":
    multiprocessing.freeze_support()
    cfg = load_config()
    library = build_library(MapSpec(),cfg)
    stream = build_probe_stream(MapSpec(),cfg,library)
    print("Saved:",stream["path"])
