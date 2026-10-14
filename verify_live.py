"""Actual Tk + MuJoCo render verification. Hidden by default; no desktop capture.

--visible displays the window. --native also opens the native MuJoCo viewer.
Tests automatically switch scenes and close. PNGs use our own render/drawing data.
"""
import argparse
import multiprocessing
import sys
import time
from navlab.config import load_config
from navlab.map import MapSpec
from navlab.library import build_library
from navlab.probe_stream import build_probe_stream
from navlab.experiments import run_case,new_output,save_json
from navlab.report import write_report
from navlab.live_views import LiveViews


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible",action="store_true")
    parser.add_argument("--native",action="store_true")
    parser.add_argument("--case",choices=list("ABCDE"))
    args = parser.parse_args()
    cfg = load_config()
    library = build_library(MapSpec(),cfg)
    stream = build_probe_stream(MapSpec(),cfg,library)
    output = new_output()
    live = LiveViews(speed=20.,native_viewer=args.native,hidden=not args.visible)
    results = []
    names = [args.case] if args.case else list("ABCDE")
    try:
        for name in names:
            frames_before = live.frames_rendered
            result = run_case(name,cfg,library,output,live=live,probe_library=stream)
            assert result["mission_success"] and result["shared_tail_success"]
            assert live.frames_rendered > frames_before+20,"actual video must keep refreshing"
            assert (output/("case_"+name)/"live_views/final_dashboard.png").exists()
            live.toggle_pause(); assert live.paused
            live.toggle_pause(); assert not live.paused
            live.native_key(32); live.pump(); assert live.paused
            live.native_key(32); live.pump(); assert not live.paused
            before_time = float(live.world.data.time)
            live.toggle_pause()
            live.root.after(40,live.toggle_pause)
            pause_started = time.perf_counter()
            live.on_world(live.world)
            assert not live.paused and time.perf_counter()-pause_started>=.03
            assert live.world.data.time==before_time,"pausing must not advance physical time"
            live.root.after(40,live.request_next)
            live.wait_for_next()
            assert live.next_requested and live.world.data.time==before_time
            results.append(result)
            save_json(output/"results.json",results)
            write_report(output,results,library,cfg)
        print("Actual Tk / MuJoCo live views passed:",output)
    finally:
        live.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
