import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sea.fitness import Dataset  # noqa: E402
from sea.privileged import Field, PrivSpec  # noqa: E402


@pytest.fixture(scope="session")
def spec():
    return PrivSpec([
        Field("pos", 2, "agent position"),
        Field("goal", 2, "goal position"),
        Field("clock", 1, "time since episode start"),
        Field("noise", 3, "unrelated noise"),
        Field("dead", 2, "always zero"),
    ])


def make_data(spec, n=96, t=48, seed=0, with_feat=True):
    """Point-mass reaching: actions move `pos`, reward is negative distance to `goal`.

    The fake latent `feat` encodes pos and goal nonlinearly but carries nothing about
    `noise`, which is how 'not inferable from pixels' is modelled here.
    """
    g = torch.Generator().manual_seed(seed)
    pos = torch.zeros(n, t, 2)
    goal = torch.rand(n, 1, 2, generator=g).expand(n, t, 2) * 2 - 1
    action = torch.zeros(n, t, 2)
    pos[:, 0] = torch.rand(n, 2, generator=g) * 2 - 1
    for k in range(1, t):
        toward = goal[:, k - 1] - pos[:, k - 1]
        a = (0.6 * toward + 0.5 * torch.randn(n, 2, generator=g)).clamp(-1, 1)
        action[:, k] = a  # replay convention: action[t] led to step t
        pos[:, k] = pos[:, k - 1] + 0.1 * a
    reward = -(pos - goal).norm(dim=-1)
    clock = torch.arange(t).float().expand(n, t)[..., None] / t
    noise = torch.randn(n, t, 3, generator=g)
    priv = torch.cat([pos, goal, clock, noise, torch.zeros(n, t, 2)], -1)
    is_first = torch.zeros(n, t, dtype=torch.bool)
    is_first[:, 0] = True
    feat = None
    if with_feat:
        w = torch.randn(4, 64, generator=g)
        feat = torch.tanh(torch.cat([pos, goal], -1) @ w) + 0.01 * torch.randn(n, t, 64, generator=g)
    return Dataset(priv, action, reward, is_first, feat, None, spec, {"tasks": ["toy_reach"]})


@pytest.fixture(scope="session")
def data(spec):
    return make_data(spec)


GOOD = '''def phi(s):
    """Offset and distance from agent to goal."""
    rel = s["goal"] - s["pos"]
    return {"to_goal": rel, "goal_dist": norm(rel)}
'''
CLOCK = 'def phi(s):\n    return {"t": s["clock"]}\n'
NOISE = 'def phi(s):\n    return {"n": s["noise"]}\n'
CONSTANT = 'def phi(s):\n    return {"c": s["dead"]}\n'
POS_ONLY = 'def phi(s):\n    return {"p": s["pos"]}\n'
