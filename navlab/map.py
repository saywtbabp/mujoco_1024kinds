"""Door-stage crop of newmujoco's architectural shell; no glass stage.

The same wall definitions generate the physical scene and A* occupancy.
right_edge changes only the length of the existing right-hand detour block.
"""
from dataclasses import dataclass
import heapq
import math
from . import runtime
import numpy as np


@dataclass(frozen=True)
class MapSpec:
    name: str = "large_shortcut"
    right_edge: float = 3.2625

    @property
    def start(self):
        return np.array([-3.8, 0.5125, 1.25])

    @property
    def approach(self):
        return np.array([0.225, 0.66, 1.25])

    @property
    def merge(self):
        return np.array([0.225, 2.8875, 1.25])

    @property
    def goal(self):
        return np.array([-2.0, 2.8875, 1.25])

    @property
    def bounds(self):
        return (-4.9, self.right_edge + 1.75, 0.0, 3.3875)

    def walls(self):
        left, right, bottom, top = self.bounds
        walls = []
        def box(name, x0, x1, y0, y1):
            walls.append((name, np.array([(x0+x1)/2, (y0+y1)/2]),
                          np.array([(x1-x0)/2, (y1-y0)/2])))
        t = 0.045
        box("boundary_bottom", left-t, right+t, bottom-t, bottom+t)
        box("boundary_top", left-t, right+t, top-t, top+t)
        box("boundary_left", left-t, left+t, bottom, top)
        box("boundary_right", right-t, right+t, bottom, top)
        box("left_block_bottom", left, -0.70, 1.0625-t, 1.0625+t)
        box("left_block_top", left, -0.70, 2.30-t, 2.30+t)
        box("left_block_end", -0.70-t, -0.70+t, 1.0625, 2.30)
        box("right_block_bottom", 0.6625, self.right_edge, 1.0625-t, 1.0625+t)
        box("right_block_top", 0.6625, self.right_edge, 2.30-t, 2.30+t)
        box("right_block_end", self.right_edge-t, self.right_edge+t, 1.0625, 2.30)
        box("right_block_start", 0.6625-t, 0.6625+t, 1.0625, 2.30)
        return walls


HINGE = np.array([-0.61, 1.0625])
DOOR_WIDTH = 1.1825
DOOR_THICKNESS = 0.05
DOOR_HEIGHT = 2.0


class Planner:
    def __init__(self, spec, cfg, door_angle=None, force_door=False):
        self.cfg = cfg
        self.res = cfg.grid_resolution
        self.origin = np.array(spec.bounds[::2])  # xmin, ymin
        self.upper = np.array([spec.bounds[1], spec.bounds[3]])
        self.shape = tuple(np.floor((self.upper-self.origin)/self.res).astype(int)+1)
        self.boxes = [(c, np.eye(2), h + cfg.clearance) for _, c, h in spec.walls()]
        if door_angle is not None:
            c, s = math.cos(door_angle), math.sin(door_angle)
            rot = np.array([[c, -s], [s, c]])
            center = HINGE + rot @ np.array([DOOR_WIDTH/2, 0.])
            self.boxes.append((center, rot, np.array([DOOR_WIDTH/2, DOOR_THICKNESS/2])+cfg.clearance))
        if force_door:
            # Seal the alternative right-hand opening so the Door hypothesis really uses the door.
            self.boxes.append((np.array([(spec.right_edge+spec.bounds[1])/2, 1.68]),
                               np.eye(2), np.array([(spec.bounds[1]-spec.right_edge)/2, 0.8])))
        cells = np.indices(self.shape).reshape(2, -1).T
        self.blocked = self.collides(self.origin+cells*self.res).reshape(self.shape)

    def collides(self, points):
        points = np.atleast_2d(points)
        hit = np.any((points < self.origin) | (points > self.upper), axis=1)
        for center, rotation, extent in self.boxes:
            hit |= np.all(np.abs((points-center) @ rotation) <= extent, axis=1)
        return hit

    def clear(self, a, b):
        n = max(2, int(np.linalg.norm(b-a)/(self.res/3))+2)
        return not np.any(self.collides(np.linspace(a, b, n)))

    def connect(self, point):
        if self.collides(point)[0]:
            raise ValueError("A* endpoint has insufficient clearance: " + str(point))
        mid = np.rint((point-self.origin)/self.res).astype(int)
        candidates = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                cell = tuple(mid + [dx, dy])
                if all(0 <= cell[k] < self.shape[k] for k in (0,1)) and not self.blocked[cell]:
                    q = self.origin + np.array(cell)*self.res
                    if self.clear(point, q):
                        candidates.append((float(np.linalg.norm(q-point)), cell))
        if not candidates:
            raise ValueError("cannot connect endpoint to A*")
        return min(candidates)[1]

    def plan(self, start, goal):
        start, goal = np.asarray(start)[:2], np.asarray(goal)[:2]
        a, b = self.connect(start), self.connect(goal)
        queue, parents, costs, closed = [(0., a)], {}, {a: 0.}, set()
        while queue:
            _, p = heapq.heappop(queue)
            if p in closed:
                continue
            closed.add(p)
            if p == b:
                break
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                q = (p[0]+dx, p[1]+dy)
                if not all(0 <= q[k] < self.shape[k] for k in (0,1)):
                    continue
                if self.blocked[q] or self.blocked[p[0]+dx,p[1]] or self.blocked[p[0],p[1]+dy]:
                    continue
                cost = costs[p]+math.hypot(dx,dy)*self.res
                if cost < costs.get(q, float("inf")):
                    costs[q], parents[q] = cost, p
                    heapq.heappush(queue, (cost+math.dist(q,b)*self.res, q))
        if b not in closed:
            raise ValueError("A* route unavailable")
        chain = [b]
        while chain[-1] != a:
            chain.append(parents[chain[-1]])
        raw = np.array([start]+[self.origin+np.array(c)*self.res for c in reversed(chain)]+[goal])
        raw_length = float(np.linalg.norm(np.diff(raw,axis=0),axis=1).sum())
        route, i = [start], 0
        while i < len(raw)-1:
            j = len(raw)-1
            while j > i+1 and not self.clear(raw[i],raw[j]):
                j -= 1
            route.append(raw[j]); i = j
        return np.array(route), raw_length


def geometry(spec, cfg):
    door, jd = Planner(spec,cfg,force_door=True).plan(spec.start,spec.merge)
    bypass, jb = Planner(spec,cfg,door_angle=0.).plan(spec.start,spec.merge)
    return dict(J_door_geo=jd, J_bypass_geo=jb, G=jb-jd,
                door_path=door.tolist(), bypass_path=bypass.tolist(),
                units="m", method="8-neighbour A* grid cost including endpoint connectors")
