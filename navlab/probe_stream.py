"""Time-matched Probe predictions without changing the physical Probe controller.

Each prefix contains peak contact force SO FAR and angle change SO FAR. A
read-only callback records the same 20 Hz observations as the GUI. The stream
is bound to the current full library signature; a controller version change
requires a compatible rebuilt library even when the standard Probe is unchanged.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import time
from . import runtime
import numpy as np
from .physics import World
from .library import atomic_npz

_stream_worker = None


def init_stream_worker(spec,cfg,approach):
    global _stream_worker
    _stream_worker = World(spec,cfg),approach


def capture_probe(world):
    """Record the actual standard Probe; no interpolation of its terminal output."""
    start = float(world.data.time)
    angle0 = float(world.data.qpos[world.qa])
    rows = [[0.,0.,0.]]
    old_callback = world.callback
    # These are sampling/diagnostic counters. Preserve the 5-step control cadence.
    if world.step_count % 5:
        raise ValueError("Probe must start at a controller tick")
    world.step_count = 0
    world.route_contact_peak = world.route_wall_peak = 0.
    def record(current):
        rows.append([float(current.data.time-start),float(current.route_contact_peak),
                     float(current.data.qpos[current.qa]-angle0)])
        if old_callback:
            old_callback(current)
    world.callback = record
    try:
        z,wall = world.probe()
    finally:
        world.callback = old_callback
    final = [float(world.data.time-start),float(z[0]),float(z[1])]
    if abs(rows[-1][0]-final[0]) < 1e-9:
        rows[-1] = final
    else:
        rows.append(final)
    return np.asarray(rows,dtype=float),z,wall


def stream_candidate(item):
    index,theta = item
    world,approach = _stream_worker
    world.restore(approach,theta)
    rows,z,_ = capture_probe(world)
    return index,rows,z


def build_probe_stream(spec,cfg,library):
    key = hashlib.sha256((library["signature"]+"probe-prefix-v1").encode()+
                         __file_bytes()).hexdigest()
    path = library["folder"]/"probe_prefix.npz"
    n = len(library["theta"])
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            if str(saved["signature"]) == key and np.all(saved["done"]):
                print(f"Probe prefix cache: {n}/{n} worlds",flush=True)
                return dict(times=saved["times"].copy(),predictions=saved["predictions"].copy(),
                            signature=key,path=path)
    times = np.r_[np.arange(0.,cfg.probe_duration-1e-9,25*cfg.timestep),cfg.probe_duration]
    predictions = np.full((n,len(times),2),np.nan)
    done = np.zeros(n,dtype=bool)
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            if str(saved["signature"]) == key and np.array_equal(saved["times"],times):
                predictions,done = saved["predictions"].copy(),saved["done"].copy()
    pending = [(i,library["theta"][i]) for i in range(n) if not done[i]]
    def save():
        atomic_npz(path,signature=np.array(key),times=times,predictions=predictions,done=done)
    started = time.perf_counter()
    print(f"Recording time-matched Probe predictions: {len(pending)} worlds",flush=True)
    pool = None
    if cfg.workers == 1:
        init_stream_worker(spec,cfg,library["approach"])
        results = map(stream_candidate,pending)
    else:
        pool = ProcessPoolExecutor(max_workers=cfg.workers,initializer=init_stream_worker,
                                   initargs=(spec,cfg,library["approach"]))
        results = pool.map(stream_candidate,pending,chunksize=8)
    try:
        for i,rows,z in results:
            np.testing.assert_allclose(z,library["z_hat"][i],atol=1e-9,rtol=0,
                                       err_msg="prefix instrumentation changed the standard Probe")
            for axis in (0,1):
                predictions[i,:,axis] = np.interp(times,rows[:,0],rows[:,axis+1])
            predictions[i,-1] = z
            done[i] = True
            count = int(done.sum())
            if count % 128 == 0 or count == n:
                save()
                print(f"  Probe prefixes {count}/{n} elapsed={time.perf_counter()-started:.1f}s",flush=True)
    finally:
        save()
        if pool:
            pool.shutdown(wait=True,cancel_futures=True)
    if not done.all() or not np.isfinite(predictions).all():
        raise RuntimeError("incomplete Probe-prefix library; rerun to resume")
    return dict(times=times,predictions=predictions,signature=key,path=path)


def __file_bytes():
    from pathlib import Path
    return Path(__file__).read_bytes()


def predictions_at(stream,elapsed):
    times = stream["times"]
    t = float(np.clip(elapsed,times[0],times[-1]))
    hi = min(int(np.searchsorted(times,t,side="right")),len(times)-1)
    lo = max(0,hi-1)
    fraction = (t-times[lo])/max(times[hi]-times[lo],1e-12)
    prediction = stream["predictions"]
    return prediction[:,lo]*(1-fraction)+prediction[:,hi]*fraction
