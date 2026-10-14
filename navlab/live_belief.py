"""Scientific state behind the live window and the report replay; usable headless."""
import csv
from . import runtime
import numpy as np
from .belief import consistency_update,normalized,support
from .probe_stream import predictions_at
from .library import atomic_npz


class LiveBelief:
    def __init__(self,cfg,library,stream,labels,geometry_value):
        self.cfg,self.library,self.stream = cfg,library,stream
        self.prior = np.full(len(labels),1./len(labels))
        self.weights = self.prior.copy()
        self.labels = np.asarray(labels,dtype=int).copy()
        self.G = float(geometry_value)
        self.error = np.zeros(len(labels))
        self.p,self.D = support(self.weights,self.labels)
        self.action = "INITIAL"
        self.phase = "INITIAL"
        self.label_source = "关闭门初始路线标签"
        self.probe_start = None
        self.probe_angle0 = 0.
        self.probe_elapsed = 0.
        self.observation = np.zeros(2)
        self.records,self.timeline = [],[]
        self.record(0.,"INITIAL")

    def record(self,time_s,phase):
        self.phase = phase
        self.records.append(dict(time=float(time_s),phase=phase,elapsed=float(self.probe_elapsed),
                                 weights=self.weights.copy(),labels=self.labels.copy(),
                                 observation=self.observation.copy(),p=self.p,D=self.D))

    def begin_probe(self,world):
        if world.step_count % 5:
            raise ValueError("Probe must begin on a controller tick")
        world.step_count = 0
        world.route_contact_peak = world.route_wall_peak = 0.
        self.probe_start = float(world.data.time)
        self.probe_angle0 = float(world.data.qpos[world.qa])
        self.action = "Probe"
        self.phase = "PROBE"

    def on_world(self,world):
        if self.probe_start is not None and world.stage == "PROBE":
            elapsed = float(world.data.time-self.probe_start)
            if elapsed > self.probe_elapsed+1e-9:
                self.probe_elapsed = elapsed
                self.observation = np.array([world.route_contact_peak,
                    float(world.data.qpos[world.qa])-self.probe_angle0])
                # The accumulated prefix is correlated with the previous prefix.
                # Always use the original prior, never multiply repeated frame scores.
                self.weights,self.error = consistency_update(
                    self.prior,predictions_at(self.stream,elapsed),self.observation,self.cfg)
                self.p,self.D = support(self.weights,self.labels)
                self.record(world.data.time,"PROBE")
        self.timeline.append([float(world.data.time),world.stage,self.phase,self.action,
                              float(world.force),float(world.data.qpos[world.qa]),
                              float(self.observation[0]),float(self.observation[1]),
                              self.G,self.p,self.D,float(self.weights.max())])

    def finish_probe(self,observation,time_s):
        self.probe_elapsed = self.cfg.probe_duration
        self.observation = np.asarray(observation).copy()
        self.weights,self.error = consistency_update(self.prior,self.library["z_hat"],self.observation,self.cfg)
        self.p,self.D = support(self.weights,self.labels)
        self.record(time_s,"PROBE_COMPLETE")
        return self.weights.copy(),self.error.copy()

    def set_continuation(self,labels,action,time_s):
        self.labels = np.asarray(labels,dtype=int).copy()
        self.p,self.D = support(self.weights,self.labels)
        self.action = action
        self.label_source = "同一 Probe 后状态的剩余路线标签"
        self.record(time_s,"STATE_CONDITIONED")

    def save(self,folder):
        atomic_npz(folder/"belief_history.npz",time=np.array([r["time"] for r in self.records]),
                   phase=np.array([r["phase"] for r in self.records]),
                   probe_elapsed=np.array([r["elapsed"] for r in self.records]),
                   weights=np.array([r["weights"] for r in self.records]),
                   labels=np.array([r["labels"] for r in self.records]),
                   observation=np.array([r["observation"] for r in self.records]),
                   p_D=np.array([r["p"] for r in self.records]),D=np.array([r["D"] for r in self.records]))
        with (folder/"live_timeline.csv").open("w",newline="",encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time_s","physical_stage","belief_phase","action","contact_force_N",
                             "door_angle_rad","probe_peak_N","probe_delta_rad","G_m","p_D","D","max_weight"])
            writer.writerows(self.timeline)
