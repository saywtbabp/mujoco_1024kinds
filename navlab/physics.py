"""Full original tensegrity drone; all motion is integrated by MuJoCo.

Only reset/clone initializes qpos. No kinematic dragging, hinge motor, applied
door torque or teleportation is used during Probe or either navigation route.
Stage 1 uses ideal proprioception and ideal contact/door-angle instruments.
"""
from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET
from . import runtime
import mujoco
import numpy as np
from .config import ROOT
from .map import HINGE, DOOR_WIDTH, DOOR_HEIGHT, DOOR_THICKNESS, Planner


def unit(x):
    return x / max(float(np.linalg.norm(x)), 1e-12)


def clip_norm(x, limit):
    return x * min(1., limit / max(float(np.linalg.norm(x)), 1e-12))


class FlightFailure(Exception):
    """A valid physical rollout left the declared fixed-height navigation envelope."""


def scene_xml(spec, cfg):
    root = ET.Element("mujoco", model="decision_relevant_door")
    ET.SubElement(root,"include",file=(ROOT/"models"/cfg.model_source/"tensegrity_real_model.xml").as_posix())
    ET.SubElement(root,"compiler",angle="degree",autolimits="true")
    ET.SubElement(root,"option",timestep=str(cfg.timestep),integrator="implicitfast",
                  gravity="0 0 -9.81",iterations="80")
    visual = ET.SubElement(root,"visual")
    ET.SubElement(visual,"global",offwidth="1280",offheight="720")
    ET.SubElement(visual,"headlight",diffuse="0.7 0.7 0.7",ambient="0.4 0.4 0.4")
    wb = ET.SubElement(root,"worldbody")
    ET.SubElement(wb,"light",pos="0 1 8",dir="0 0 -1",directional="true")
    ET.SubElement(wb,"geom",name="floor",type="plane",size="12 8 0.1",rgba="0.88 0.89 0.91 1")
    for name, center, half in spec.walls():
        ET.SubElement(wb,"geom",name="wall_"+name,type="box",
                      pos=f"{center[0]} {center[1]} 1.1",size=f"{half[0]} {half[1]} 1.1",
                      rgba="0.4 0.47 0.56 1",friction="0.3 0.005 0.0001",solref="0.012 1")
    body = ET.SubElement(wb,"body",name="door",pos=f"{HINGE[0]} {HINGE[1]} 0")
    ET.SubElement(body,"joint",name="door_hinge",type="hinge",axis="0 0 1",range="0 88",
                  damping=str(cfg.hinge_damping),frictionloss="0.01")
    ET.SubElement(body,"geom",name="door_panel",type="box",
                  pos=f"{DOOR_WIDTH/2} 0 {DOOR_HEIGHT/2}",
                  size=f"{DOOR_WIDTH/2} {DOOR_THICKNESS/2} {DOOR_HEIGHT/2}",mass="1",
                  rgba="0.72 0.39 0.13 1",friction="0.3 0.005 0.0001",
                  solref="0.012 1",solimp="0.95 0.99 0.001",condim="3")
    for name, point, color in (("S",spec.start,"0.1 0.8 0.3 0.8"),
                               ("M",spec.merge,"1 0.65 0.1 0.8"),
                               ("G",spec.goal,"0.8 0.1 0.5 0.8")):
        ET.SubElement(wb,"site",name=name,pos=" ".join(map(str,point)),size="0.07",rgba=color)
    ET.SubElement(wb,"camera",name="overview",pos="0 1.7 13",xyaxes="1 0 0 0 1 0",fovy="47")
    return ET.tostring(root,encoding="unicode")


@dataclass
class RouteResult:
    time: float
    success: bool
    reason: str
    contact_peak: float
    wall_peak: float
    final_position: list
    final_speed: float
    final_angle: float


class World:
    def __init__(self, spec, cfg):
        self.cfg, self.spec = cfg, spec
        self.model = m = mujoco.MjModel.from_xml_string(scene_xml(spec,cfg))
        self.data = d = mujoco.MjData(m)
        self.frame_id, self.door_body = m.body("quad_frame").id, m.body("door").id
        self.hinge = m.joint("door_hinge").id
        self.qa, self.da = int(m.jnt_qposadr[self.hinge]), int(m.jnt_dofadr[self.hinge])
        roots = {m.body(name).id for name in ("link1","link2","link3","link4","quad_frame")}
        self.robot_ids = set(roots)
        for bid in range(1,m.nbody):
            if int(m.body_parentid[bid]) in self.robot_ids:
                self.robot_ids.add(bid)
        self.robot_geoms = {gid for gid in range(m.ngeom) if int(m.geom_bodyid[gid]) in self.robot_ids}
        self.door_geom = m.geom("door_panel").id
        self.wall_geoms = {gid for gid in range(m.ngeom) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,gid) or "").startswith("wall_")}
        self.motor_ids = np.array([m.actuator(f"thruster{i}").id for i in range(1,5)])
        self.mass = float(sum(m.body_mass[list(self.robot_ids)]))
        columns = []
        for aid in self.motor_ids:
            sid = int(m.actuator_trnid[aid,0])
            force = m.actuator_gear[aid,:3]
            torque = np.cross(m.site_pos[sid],force)+m.actuator_gear[aid,3:]
            columns.append(np.r_[force[2],torque])
        self.mixer = np.linalg.inv(np.column_stack(columns))
        self.limits = m.actuator_ctrlrange[self.motor_ids].copy()
        self.base_mass = float(m.body_mass[self.door_body])
        self.base_inertia = m.body_inertia[self.door_body].copy()
        self.wrench = np.zeros(6)
        self.force = self.wall_force = 0.
        self.route_contact_peak = self.route_wall_peak = 0.
        self.trace = None
        self.callback = None
        self.stage = "RESET"
        self.target = spec.start.copy()
        self.yaw_reference = self.yaw_target = 0.
        self.step_count = 0
        self.ctrl = np.zeros(4)
        self.reset((cfg.inertia_min,cfg.friction_min))

    def set_parameters(self, theta):
        inertia, friction = map(float, theta)
        # Uniform box scaling gives I_hinge=m*(width^2/3+thickness^2/12).
        # The body inertia is at its COM; setting its z entry to I_hinge would be WRONG.
        mass = inertia / (DOOR_WIDTH**2/3 + DOOR_THICKNESS**2/12)
        self.model.body_mass[self.door_body] = mass
        self.model.body_inertia[self.door_body] = self.base_inertia * (mass/self.base_mass)
        self.model.dof_frictionloss[self.da] = friction
        mujoco.mj_setConst(self.model,self.data)
        self.theta = np.array([inertia,friction])

    def hinge_inertia(self):
        mass_matrix = np.empty((self.model.nv,self.model.nv))
        mujoco.mj_fullM(self.model,mass_matrix,self.data.qM)
        return float(mass_matrix[self.da,self.da])

    def reset(self, theta):
        m,d = self.model,self.data
        mujoco.mj_resetData(m,d)
        self.set_parameters(theta)
        mujoco.mj_forward(m,d)
        delta = self.spec.start-d.sensor("pos").data
        for jid in range(m.njnt):
            if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE and int(m.jnt_bodyid[jid]) in self.robot_ids:
                a = int(m.jnt_qposadr[jid]); d.qpos[a:a+3] += delta
        mujoco.mj_forward(m,d)
        self.target = self.spec.start.copy()
        self.yaw_target = self.yaw_reference = 0.
        self.step_count = 0
        self.force = self.wall_force = 0.
        self.route_contact_peak = self.route_wall_peak = 0.

    def snapshot(self):
        mujoco.mj_forward(self.model,self.data)
        return dict(qpos=self.data.qpos.copy(),qvel=self.data.qvel.copy(),
                    act=self.data.act.copy(),time=float(self.data.time),
                    yaw_reference=self.yaw_reference)

    def restore(self, state, theta):
        mujoco.mj_resetData(self.model,self.data)
        self.set_parameters(theta)
        for key in ("qpos","qvel","act"):
            getattr(self.data,key)[:] = state[key]
        self.data.time = state["time"]
        self.yaw_reference = self.yaw_target = state["yaw_reference"]
        self.target = self.spec.approach.copy()
        self.step_count = 0
        self.force = self.wall_force = 0.
        self.route_contact_peak = self.route_wall_peak = 0.
        mujoco.mj_forward(self.model,self.data)

    def observed(self):
        d = self.data
        pos, vel = d.sensor("pos").data, d.sensor("linvel").data
        R = d.xmat[self.frame_id].reshape(3,3)
        gyro = d.sensor("imu_gyro").data
        return pos, vel, R, gyro

    def control(self, horizontal=None, yaw=None):
        cfg = self.cfg
        pos, vel, R, gyro = self.observed()
        error = self.target-pos
        v = clip_norm((4./3.2)*error[:2],cfg.flight_speed)
        acc = np.r_[3.2*(v-vel[:2]), np.clip(10.*error[2]-5.*vel[2],-4.,4.)]
        acc[:2] = clip_norm(acc[:2],4. if self.stage == "BACKOFF" else 2.)
        force = self.mass*(acc+np.array([0.,0.,9.81]))
        if horizontal is not None:
            force[:2] = horizontal
        if yaw is not None:
            self.yaw_target = float(yaw)
        dyaw = (self.yaw_target-self.yaw_reference+math.pi)%(2*math.pi)-math.pi
        self.yaw_reference += float(np.clip(dyaw,-.7*5*cfg.timestep,.7*5*cfg.timestep))
        b3 = unit(force)
        b2 = unit(np.cross(b3,[math.cos(self.yaw_reference),math.sin(self.yaw_reference),0.]))
        desired = np.column_stack((np.cross(b2,b3),b2,b3))
        e = .5*(desired.T@R-R.T@desired)
        er = np.array([e[2,1],e[0,2],e[1,0]])
        authority = 1.6 if self.stage == "BACKOFF" else 1.
        torque = -np.array([.65*authority,.65*authority,.35])*er-np.array([.10,.10,.08])*gyro
        thrust = max(0.,float(force@R[:,2]))
        return np.clip(self.mixer@np.r_[thrust,torque],self.limits[:,0],self.limits[:,1])

    def step(self, horizontal=None, yaw=None):
        m,d = self.model,self.data
        mujoco.mj_step1(m,d)
        if self.step_count % 5 == 0:
            self.ctrl = self.control(horizontal,yaw)
        d.ctrl[:] = 0.
        d.ctrl[self.motor_ids] = self.ctrl
        mujoco.mj_step2(m,d)
        force, wall = 0.,0.
        for ci in range(d.ncon):
            contact = d.contact[ci]
            a,b = int(contact.geom1),int(contact.geom2)
            if (a in self.robot_geoms) == (b in self.robot_geoms):
                continue
            other = b if a in self.robot_geoms else a
            if other == self.door_geom or other in self.wall_geoms:
                mujoco.mj_contactForce(m,d,ci,self.wrench)
                normal = max(0.,float(self.wrench[0]))
                if other == self.door_geom:
                    force += normal
                else:
                    wall += normal
        self.force,self.wall_force = force,wall
        self.route_contact_peak = max(self.route_contact_peak,force)
        self.route_wall_peak = max(self.route_wall_peak,wall)
        self.step_count += 1
        if self.step_count % 25 == 0:
            pos,vel,R,_ = self.observed()
            if np.any(d.warning.number) or not np.all(np.isfinite(d.qpos)):
                raise RuntimeError("MuJoCo numerical warning; world is invalid, not a physical failure")
            if pos[2] < .5 or pos[2] > 2.3:
                raise FlightFailure("drone left flight-height envelope: " + str(pos))
            if self.trace is not None:
                self.trace.append([float(d.time),self.stage,*pos.tolist(),float(self.force),
                                   float(d.qpos[self.qa]),float(np.linalg.norm(vel))])
            if self.callback:
                self.callback(self)

    def hold(self, target, duration, stage="HOLD"):
        self.stage,self.target = stage,np.asarray(target).copy()
        end = self.data.time+duration
        while self.data.time < end-1e-10:
            self.step()

    def fly(self, path, settle=True):
        self.stage = "FLIGHT"
        start_time = self.data.time
        for i,point in enumerate(path):
            self.target = np.r_[np.asarray(point)[:2],1.25]
            last = i == len(path)-1
            dwell = 0.
            while self.data.time-start_time < self.cfg.route_timeout:
                self.step()
                if self.step_count % 5:
                    continue
                pos,vel,R,gyro = self.observed()
                reached = np.linalg.norm(pos-self.target) < (self.cfg.merge_position_tolerance if last else .10)
                if last and settle:
                    yaw = math.atan2(R[1,0],R[0,0])
                    reached &= (np.linalg.norm(vel) < self.cfg.merge_speed_tolerance
                                and abs(yaw) < self.cfg.merge_yaw_tolerance
                                and np.linalg.norm(gyro) < self.cfg.merge_gyro_tolerance)
                    dwell = dwell+5*self.cfg.timestep if reached else 0.
                    reached = dwell >= self.cfg.merge_dwell
                if reached:
                    break
            else:
                return False
        return True

    def probe(self):
        """Fixed world +Y push, nominal contact x=.225; feedback law identical for all worlds."""
        self.stage = "PROBE"
        self.target = self.spec.approach.copy()
        self.yaw_target = 0.
        angle0 = float(self.data.qpos[self.qa])
        start = float(self.data.time)
        peak = wall_peak = 0.
        while self.data.time-start < self.cfg.probe_duration-1e-10:
            pos,vel,_,_ = self.observed()
            fx = 4.*(self.spec.approach[0]-pos[0])-3.*vel[0]
            fy = self.cfg.probe_force_max-self.cfg.probe_velocity_damping*vel[1]
            horizontal = clip_norm(np.array([fx,max(0.,fy)]),self.cfg.probe_force_max)
            self.step(horizontal,yaw=0.)
            peak = max(peak,self.force); wall_peak = max(wall_peak,self.wall_force)
        mujoco.mj_forward(self.model,self.data)
        return np.array([peak,float(self.data.qpos[self.qa])-angle0]),wall_peak

    def push_open(self):
        self.stage = "PUSH"
        start = self.data.time
        while self.data.time-start < self.cfg.push_timeout:
            angle = float(self.data.qpos[self.qa])
            if angle >= math.radians(self.cfg.open_angle_deg):
                return True
            tangent = np.array([math.cos(angle),math.sin(angle)])
            normal = np.array([-math.sin(angle),math.cos(angle)])
            # Follow the same nominal lever arm as the door rotates; no theta access.
            contact = HINGE + (.225-HINGE[0])*tangent
            pos,vel,_,_ = self.observed()
            tangential_error = float((contact-pos[:2])@tangent)
            tangential_force = 4.*tangential_error-3.*float(vel[:2]@tangent)
            normal_force = self.cfg.push_force_max-1.2*float(vel[:2]@normal)
            horizontal = clip_norm(tangential_force*tangent+max(0.,normal_force)*normal,self.cfg.push_force_max)
            self.target = np.r_[contact-.15*normal,1.25]
            self.step(horizontal,yaw=angle)
        return False

    def disengage(self, bypass=False):
        self.stage = "BACKOFF"
        pos,_,_,_ = self.observed()
        angle = float(self.data.qpos[self.qa])
        normal = np.array([-math.sin(angle),math.cos(angle)])
        point = pos.copy(); point[:2] -= .38*normal
        if bypass:
            point[1] = min(point[1],.60)
        else:
            point[0] = max(point[0],self.spec.approach[0])
        self.target = point
        self.yaw_target = 0.
        end = self.data.time+2.5
        while self.data.time < end-1e-10:
            self.step()

    def route(self, kind, state, theta, prefix_time=0.):
        self.restore(state,theta)
        start = self.data.time
        peak,wall = 0.,0.
        old_callback = self.callback
        def observe(world):
            nonlocal peak,wall
            peak,wall = max(peak,world.force),max(wall,world.wall_force)
            if old_callback:
                old_callback(world)
        self.callback = observe
        success,reason = False,""
        try:
            if kind == "door":
                if not self.push_open():
                    reason = "push timeout before required clearance angle"
                    return self.route_result(start,prefix_time,False,reason,peak,wall)
                self.disengage()
                angle = float(self.data.qpos[self.qa])
                path,_ = Planner(self.spec,self.cfg,door_angle=angle,force_door=True).plan(
                    self.observed()[0],self.spec.merge)
            elif kind == "bypass":
                # A post-Probe state may still be in contact. Recover to the lower corridor first.
                if float(self.data.time) > .01 and np.linalg.norm(self.observed()[0][:2]-self.spec.start[:2]) > .2:
                    self.disengage(bypass=True)
                    if not self.fly([self.spec.approach],settle=True):
                        return self.route_result(start,prefix_time,False,"backoff timeout",peak,wall)
                path,_ = Planner(self.spec,self.cfg,door_angle=0.).plan(self.observed()[0],self.spec.merge)
            else:
                raise ValueError(kind)
            self.yaw_target = 0.
            success = self.fly(path[1:],settle=True)
            reason = "reached common merge state" if success else "flight timeout"
        except (ValueError,FlightFailure) as exc:
            reason = "physical navigation failure: " + str(exc)
        finally:
            self.callback = old_callback
        return self.route_result(start,prefix_time,success,reason,peak,wall)

    def route_result(self, start, prefix, success, reason, peak, wall):
        mujoco.mj_forward(self.model,self.data)
        pos,vel,_,_ = self.observed()
        return RouteResult(float(self.data.time-start+prefix) if success else float("inf"),
                           success,reason,float(max(peak,self.route_contact_peak)),
                           float(max(wall,self.route_wall_peak)),pos.tolist(),
                           float(np.linalg.norm(vel)),float(self.data.qpos[self.qa]))


def prepare(spec,cfg):
    world = World(spec,cfg)
    world.hold(spec.start,.8,"INITIAL_SETTLE")
    start_state = world.snapshot()
    initial_time = world.data.time
    if not world.fly([spec.approach]):
        raise RuntimeError("cannot reach standardized Probe approach pose")
    approach_time = float(world.data.time-initial_time)
    approach_state = world.snapshot()
    if abs(world.data.qpos[world.qa]) > 1e-7:
        raise RuntimeError("approach touched the door; standard initial state is invalid")
    return world,start_state,approach_state,approach_time
