"""Privileged simulator state as a flat, named vector.

Every environment emits one float32 vector per step under the observation key ``priv``.
The vector layout is described by a :class:`PrivSpec` (ordered named fields), which is
what abstraction programs index into and what the LLM prompt describes. Storing the raw
vector in replay (instead of the abstraction's output) is what lets abstractions be
re-scored and swapped at any point in training without collecting new data.
"""

import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Field:
    name: str
    dim: int
    doc: str = ""


class PrivSpec:
    def __init__(self, fields):
        self.fields = [f if isinstance(f, Field) else Field(**f) for f in fields]
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate privileged field names: {names}")
        self.slices = {}
        start = 0
        for f in self.fields:
            self.slices[f.name] = slice(start, start + f.dim)
            start += f.dim
        self.dim = start

    @property
    def names(self):
        return [f.name for f in self.fields]

    def split(self, vec):
        """Named views into a (..., dim) array or tensor."""
        if vec.shape[-1] != self.dim:
            raise ValueError(f"privileged vector has dim {vec.shape[-1]}, spec expects {self.dim}")
        return {name: vec[..., sl] for name, sl in self.slices.items()}

    def to_json(self):
        return [{"name": f.name, "dim": f.dim, "doc": f.doc} for f in self.fields]

    @classmethod
    def from_json(cls, obj):
        return cls(obj if isinstance(obj, list) else json.loads(obj))

    def __eq__(self, other):
        return isinstance(other, PrivSpec) and self.to_json() == other.to_json()


class MujocoReader:
    """Reads per-body pose, velocity and contact force from raw MuJoCo structs.

    ``model``/``data`` are ``mujoco.MjModel``/``mujoco.MjData`` (for dm_control pass
    ``physics.model.ptr`` and ``physics.data.ptr``).
    """

    def __init__(self, model, bodies=None, with_q=False):
        import mujoco

        self._mj = mujoco
        if bodies is None:
            bodies = []
            for i in range(1, model.nbody):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                if name:
                    bodies.append(name)
        self.bodies = list(bodies)
        self._ids = []
        for name in self.bodies:
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if idx < 0:
                raise KeyError(f"body {name!r} not found in the MuJoCo model")
            self._ids.append(idx)
        self._row = {idx: k for k, idx in enumerate(self._ids)}
        self._with_q = bool(with_q)
        fields = []
        for name in self.bodies:
            fields += [
                Field(f"pos.{name}", 3, f"world position (m) of body '{name}'"),
                Field(f"quat.{name}", 4, f"world orientation quaternion (w, x, y, z) of body '{name}'"),
                Field(f"vel.{name}", 3, f"world linear velocity (m/s) of body '{name}'"),
                Field(f"force.{name}", 1, f"summed contact normal force (N) on body '{name}'"),
            ]
        if self._with_q:
            fields += [
                Field("qpos", int(model.nq), "generalized positions (joint angles, root pose)"),
                Field("qvel", int(model.nv), "generalized velocities"),
            ]
        self.fields = fields
        self._vel = np.zeros(6)
        self._wrench = np.zeros(6)

    def read(self, model, data):
        mj = self._mj
        force = np.zeros(len(self._ids))
        for c in range(data.ncon):
            con = data.contact[c]
            rows = {self._row.get(int(model.geom_bodyid[g])) for g in (con.geom1, con.geom2)}
            rows.discard(None)
            if not rows:
                continue
            mj.mj_contactForce(model, data, c, self._wrench)
            for row in rows:
                force[row] += abs(self._wrench[0])
        out = []
        for row, idx in enumerate(self._ids):
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, idx, self._vel, 0)
            out += [data.xpos[idx], data.xquat[idx], self._vel[3:], force[row : row + 1]]
        if self._with_q:
            out += [data.qpos, data.qvel]
        return np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in out])


# Meta-World: one layout shared by all 50 tasks so abstractions transfer across tasks.
# The object slots follow Meta-World's own observation layout (unused slots are zero).
METAWORLD_BODIES = ("hand", "leftpad", "rightpad")
METAWORLD_OBS_FIELDS = (
    Field("hand_pos", 3, "world position (m) of the gripper (end-effector)"),
    Field("gripper_open", 1, "gripper opening, about 0 (closed) to 1 (open)"),
    Field("obj_pos", 3, "world position (m) of the primary manipulated object"),
    Field("obj_quat", 4, "orientation quaternion of the primary object"),
    Field("obj2_pos", 3, "world position (m) of the secondary object; all zeros if the task has none"),
    Field("obj2_quat", 4, "orientation quaternion of the secondary object; all zeros if none"),
    Field("goal_pos", 3, "world position (m) of the task goal"),
    Field("hand_step", 3, "gripper displacement (m) since the previous control step"),
    Field("obj_step", 3, "primary-object displacement (m) since the previous control step"),
    Field("obj2_step", 3, "secondary-object displacement (m) since the previous control step"),
)


class MetaWorldPriv:
    def __init__(self, env):
        self._reader = MujocoReader(env.model, METAWORLD_BODIES)
        self.spec = PrivSpec(list(METAWORLD_OBS_FIELDS) + self._reader.fields)

    def __call__(self, env, state):
        cur, prev = np.asarray(state[:18], np.float64), np.asarray(state[18:36], np.float64)
        goal = np.asarray(env._target_pos, np.float64).ravel()[:3]
        step = cur - prev
        vec = np.concatenate([
            cur[0:3],
            cur[3:4],
            cur[4:7],
            cur[7:11],
            cur[11:14],
            cur[14:18],
            goal,
            step[0:3],
            step[4:7],
            step[11:14],
            self._reader.read(env.model, env.data),
        ])
        return vec.astype(np.float32)


class DMCPriv:
    def __init__(self, physics):
        self._reader = MujocoReader(physics.model.ptr, bodies=None, with_q=True)
        self.spec = PrivSpec(self._reader.fields)

    def __call__(self, physics):
        return self._reader.read(physics.model.ptr, physics.data.ptr).astype(np.float32)
