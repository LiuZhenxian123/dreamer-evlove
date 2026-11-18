"""Environment tests. They need MuJoCo with a working offscreen renderer (MUJOCO_GL=egl)."""

import numpy as np
import pytest
import torch

from sea.privileged import PrivSpec
from sea.program import Program

metaworld = pytest.importorskip("metaworld")


def _mw(name="pick-place", **kw):
    from envs.metaworld import MetaWorld

    return MetaWorld(name, action_repeat=2, size=(64, 64), camera="corner2", seed=0, **kw)


def test_metaworld_privileged_state_matches_simulator():
    env = _mw()
    spec = PrivSpec.from_json(env.priv_spec())
    obs = env.reset()
    assert obs["priv"].shape == (spec.dim,) and obs["priv"].dtype == np.float32 and obs["image"].shape == (64, 64, 3)
    for _ in range(5):
        obs, _, _, _ = env.step(np.array([0.5, 0.2, -0.3, 1.0], dtype=np.float32))
    s = spec.split(obs["priv"])
    assert np.allclose(s["hand_pos"], obs["state"][:3], atol=1e-6)
    assert np.allclose(s["obj_pos"], obs["state"][4:7], atol=1e-6)
    assert np.allclose(s["goal_pos"], env._env._target_pos, atol=1e-6)
    assert np.allclose(s["hand_step"], obs["state"][0:3] - obs["state"][18:21], atol=1e-6)
    assert np.abs(s["hand_step"]).max() > 1e-4  # the arm moved
    assert np.allclose(s["obj2_pos"], 0.0)  # pick-place has no second object
    assert np.linalg.norm(s["pos.leftpad"] - s["pos.rightpad"]) > 0.01
    assert abs(np.linalg.norm(s["quat.hand"]) - 1) < 1e-4
    assert np.isfinite(obs["priv"]).all()


def test_metaworld_layout_is_shared_across_tasks():
    assert _mw("pick-place").priv_spec() == _mw("stick-push").priv_spec() == _mw("door-open").priv_spec()


def test_metaworld_pad_force_registers_contact():
    env = _mw("pick-place")
    spec = PrivSpec.from_json(env.priv_spec())
    obs = env.reset()
    peak = 0.0
    for _ in range(60):  # drive the open gripper down onto the table and close it
        target = np.r_[np.clip((obs["state"][4:7] - obs["state"][:3]) * 10, -1, 1), 1.0].astype(np.float32)
        obs, _, _, _ = env.step(target)
        s = spec.split(obs["priv"])
        peak = max(peak, float(s["force.leftpad"][0] + s["force.rightpad"][0]))
    assert peak > 0.0


def test_metaworld_train_and_test_layouts_differ_and_are_fixed():
    def pool(env):
        """Goal of every layout the env can start from, enumerated directly."""
        out = set()
        for layout in env._layouts:
            env._env.set_task(layout)
            env._env.reset()
            out.add(tuple(np.round(np.asarray(env._env._target_pos, np.float64), 4)))
        return out

    train_env, test_env = _mw("push", configs="train"), _mw("push", configs="test")
    train, test = pool(train_env), pool(test_env)
    assert len(train) == 50 and len(test) == 50 and not (train & test)
    # The split depends on config_seed only, not on the run seed.
    from envs.metaworld import MetaWorld

    other = MetaWorld("push", 2, (64, 64), "corner2", seed=123, configs="train")
    assert pool(other) == train
    # Episodes sample from that pool.
    spec = PrivSpec.from_json(train_env.priv_spec())
    for _ in range(10):
        goal = spec.split(train_env.reset()["priv"])["goal_pos"].astype(np.float64)
        assert min(np.abs(np.array(g) - goal).max() for g in train) < 1e-4


def test_metaworld_multitask_observation():
    env = _mw("push", task_index=2, num_tasks=5)
    obs = env.reset()
    assert obs["task"].tolist() == [0, 0, 1, 0, 0] and obs["task_id"].tolist() == [2]
    assert "task" not in _mw("push").reset()


def test_dmc_privileged_state_and_dynamics_shift():
    pytest.importorskip("dm_control")
    from envs.dmc import DeepMindControl

    env = DeepMindControl("walker_walk", action_repeat=2, size=(64, 64), seed=0)
    spec = PrivSpec.from_json(env.priv_spec())
    obs = env.reset()
    assert "pos.torso" in spec.slices and "qvel" in spec.slices and obs["priv"].shape == (spec.dim,)
    physics = env._env.physics
    s = spec.split(obs["priv"])
    assert np.allclose(s["pos.torso"], physics.named.data.xpos["torso"], atol=1e-5)
    assert np.allclose(s["qpos"], physics.data.qpos, atol=1e-5)
    act = np.zeros(env.action_space.shape, dtype=np.float32)
    speed = contact = 0.0
    for _ in range(20):  # with no torque the walker collapses onto the ground
        obs, _, _, _ = env.step(act)
        s = spec.split(obs["priv"])
        speed = max(speed, float(np.abs(s["vel.torso"]).max()))
        contact = max(contact, float(s["force.left_foot"][0] + s["force.right_foot"][0]))
    assert speed > 1e-3 and contact > 1.0

    weak = DeepMindControl("walker_walk", action_repeat=2, size=(64, 64), seed=0, shift={"motor": 0.5, "friction": 2.0})
    base = env._env.physics.model
    assert np.allclose(weak._env.physics.model.actuator_gear[:, 0], 0.5 * base.actuator_gear[:, 0])
    assert np.allclose(weak._env.physics.model.geom_friction[:, 0], 2.0 * base.geom_friction[:, 0])
    with pytest.raises(ValueError):
        DeepMindControl("walker_walk", shift={"gravity": 2.0})


def test_program_runs_on_real_metaworld_state():
    env = _mw("pick-place")
    spec = PrivSpec.from_json(env.priv_spec())
    src = '''def phi(s):
    """Grasp geometry."""
    mid = 0.5 * (s["pos.leftpad"] + s["pos.rightpad"])
    return {"pad_to_obj": dist(mid, s["obj_pos"]), "obj_to_goal": dist(s["obj_pos"], s["goal_pos"]),
            "lift": height(s["obj_pos"]), "grip": s["force.leftpad"] + s["force.rightpad"]}
'''
    out = Program(src, spec)(torch.from_numpy(np.stack([env.reset()["priv"] for _ in range(3)])))
    assert out.shape == (3, 4) and torch.isfinite(out).all() and float(out[:, 1].min()) > 0.01
