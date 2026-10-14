"""PyCharm Run: 1024 worlds -> live Probe weights + MuJoCo scenes A-E -> replay report.

Existing results are versioned/cached. No command-line arguments are necessary.
CLI --headless runs without OpenGL windows; --case E runs just one case.
"""
import argparse
import multiprocessing
import sys
from dataclasses import replace
from navlab.config import ROOT,load_config
from navlab.map import MapSpec
from navlab.library import build_library
from navlab.experiments import CASES,run_case,new_output,save_json
from navlab.report import write_report
from navlab.probe_stream import build_probe_stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--case",choices=list(CASES))
    parser.add_argument("--workers",type=int)
    parser.add_argument("--config")
    parser.add_argument("--force-library",action="store_true")
    parser.add_argument("--speed",type=float,default=1.,help="实时仿真播放速度，默认 1 倍")
    parser.add_argument("--auto-advance",action="store_true",help="场景完成后自动进入下一场景")
    parser.add_argument("--close-on-done",action="store_true",help="最后一个场景结束后关闭窗口")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    cfg = load_config(args.config)
    if args.workers is not None:
        cfg = replace(cfg,workers=args.workers).validate()
    print("Interpreter:",sys.executable,flush=True)
    print("Model:",cfg.model_source," | N =",2**cfg.sobol_power,flush=True)
    library = build_library(MapSpec(),cfg,force=args.force_library)
    stream = build_probe_stream(MapSpec(),cfg,library)
    output = new_output()
    print("实时实验与回放报告输出：",output,flush=True)
    names = [args.case] if args.case else list(CASES)
    results = []
    live = None
    if not args.headless:
        from navlab.live_views import LiveViews
        live = LiveViews(speed=args.speed)
    try:
        for index,name in enumerate(names):
            results.append(run_case(name,cfg,library,output,live=live,probe_library=stream))
            save_json(output/"results.json",results)
            write_report(output,results,library,cfg)
            if live:
                live.wait_for_next(is_last=index==len(names)-1,auto_advance=args.auto_advance,
                                   close_on_done=args.close_on_done)
    finally:
        if live:
            live.close()
    (ROOT/"results"/"LATEST.txt").write_text(str(output),encoding="utf-8")
    print("\nFinished. Report:",output/"report.html",flush=True)
    for result in results:
        print(f"  {result['case']}: {result['prior']['action']} -> {result['final_action']}; M={result['mission_success']}",flush=True)
    failed = [r["case"] for r in results if not r["mission_success"]]
    if failed:
        raise RuntimeError("mission failures (reports preserved): " + ",".join(failed))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
