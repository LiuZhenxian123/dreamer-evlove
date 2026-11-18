import pytest
import torch
from conftest import CLOCK, CONSTANT, GOOD, NOISE, POS_ONLY, make_data

from sea import fitness
from sea.archive import Archive, Entry
from sea.program import Program, ProgramError, identity_source, random_projection_source
from sea.targets import TargetBank


# ---- program sandbox ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef phi(s):\n    return {'a': s['pos']}\n",
        "def phi(s):\n    return {'a': s['pos'].__class__}\n",
        "def phi(s):\n    while True:\n        pass\n",
        "def phi(s):\n    return {'a': open('/etc/passwd')}\n",
        "def phi(s):\n    return {'a': s['pos'].numpy()}\n",
        "def phi(s):\n    f = lambda x: x\n    return {'a': s['pos']}\n",
        "x = 1\ndef phi(s):\n    return {'a': s['pos']}\n",
        "def phi(s):\n    return {'a': s['nope']}\n",  # unknown field
        "def phi(s):\n    return {'a': s['pos'].mean()}\n",  # reduces over batch/time
        "def phi(s):\n    return s['pos']\n",  # not a dict
        "def other(s):\n    return {'a': s['pos']}\n",  # no phi
    ],
)
def test_bad_programs_are_rejected(spec, source):
    with pytest.raises(ProgramError):
        Program(source, spec)


def test_program_runs_on_any_leading_shape(spec):
    p = Program(GOOD, spec)
    assert p.dim == 3 and p.dim_names == ["to_goal[0]", "to_goal[1]", "goal_dist"]
    assert p.fields == ["goal", "pos"]
    for lead in [(), (5,), (4, 7)]:
        assert p(torch.randn(*lead, spec.dim)).shape == (*lead, 3)


def test_generated_baselines_compile(spec):
    raw = Program(identity_source(spec), spec, max_dim=64)
    assert raw.dim == spec.dim
    x = torch.randn(3, 4, spec.dim)
    assert torch.allclose(raw(x), x)
    rnd = Program(random_projection_source(spec, 6, seed=1), spec, max_dim=64)
    assert rnd.dim == 6 and torch.isfinite(rnd(x)).all()


# ---- fitness --------------------------------------------------------------------------------
def test_fitness_orders_abstractions_as_expected(spec, data):
    score = {name: fitness.score_program(Program(src, spec), data)
             for name, src in dict(good=GOOD, clock=CLOCK, noise=NOISE, pos=POS_ONLY).items()}  # fmt: skip
    good = score["good"]
    assert good["closure"] > 0.5 and good["reward"] > 0.8 and good["visible"] > 0.8 and good["control"] > 0.3
    # A clock is perfectly predictable but the action tells you nothing about it, nor does reward.
    assert score["clock"]["control"] < 0.05 and score["clock"]["reward"] < 0.3
    # Noise is neither closed nor visible nor rewarding.
    assert score["noise"]["closure"] < 0.1 and score["noise"]["visible"] < 0.1 and score["noise"]["reward"] < 0.1
    # Position without the goal cannot explain reward.
    assert score["pos"]["reward"] < good["reward"] - 0.2
    assert good["score"] > max(score[k]["score"] for k in ("clock", "noise", "pos"))


def test_constant_output_is_rejected(spec, data):
    m = fitness.score_program(Program(CONSTANT, spec), data)
    assert m["score"] == 0.0 and "constant" in m["reject"]


def test_nonfinite_output_is_rejected(spec, data):
    src = 'def phi(s):\n    return {"bad": log(s["pos"] - 100.0)}\n'
    assert "non-finite" in fitness.score_program(Program(src, spec), data)["reject"]


def test_score_is_invariant_to_output_scale(spec, data):
    # Small scales on purpose: symlog is close to linear there, so this isolates rescaling.
    small = 'def phi(s):\n    return {"r": 0.01 * (s["goal"] - s["pos"])}\n'
    tiny = 'def phi(s):\n    return {"r": 0.00001 * (s["goal"] - s["pos"])}\n'
    a, b = (fitness.score_program(Program(src, spec), data) for src in (small, tiny))
    for key in ("closure", "control", "reward", "visible"):
        assert abs(a[key] - b[key]) < 0.02, key


def test_identity_pays_for_its_size(spec, data):
    good = fitness.score_program(Program(GOOD, spec), data)
    raw = fitness.score_program(Program(identity_source(spec), spec, max_dim=64), data)
    assert raw["score"] < good["score"]


def test_fitness_without_latent_features(spec):
    m = fitness.score_program(Program(GOOD, spec), make_data(spec, with_feat=False))
    assert m["visible"] is None and m["score"] > 0


def test_explained_detects_redundancy(spec, data):
    a = Program(GOOD, spec)(data.priv)
    b = Program('def phi(s):\n    return {"d": 2.0 * (s["goal"] - s["pos"]) + 1.0}\n', spec)(data.priv)
    c = Program(NOISE, spec)(data.priv)
    assert fitness.explained(a, b) > 0.95 and fitness.explained(a, c) < 0.1


# ---- archive --------------------------------------------------------------------------------
def test_archive_selects_without_redundancy(spec, data):
    archive = Archive(capacity=8)
    dup = 'def phi(s):\n    d = s["goal"] - s["pos"]\n    return {"offset": 3.0 * d, "gap": norm(d)}\n'
    for src in (GOOD, dup, CLOCK, POS_ONLY):
        p = Program(src, spec)
        assert archive.add(Entry(p, fitness.score_program(p, data)))
    assert not archive.add(Entry(Program(GOOD, spec), {"score": 1.0}))  # same program twice
    picked = archive.select_active(data, budget_dims=16, max_redundancy=0.9)
    sources = [e.program.source for e in picked]
    assert (GOOD in sources) != (dup in sources)  # exactly one of the two equivalent programs
    assert sum(e.program.dim for e in picked) <= 16


def test_archive_capacity_keeps_the_best(spec, data):
    archive = Archive(capacity=2)
    for src in (CLOCK, NOISE, POS_ONLY, GOOD):
        p = Program(src, spec)
        archive.add(Entry(p, fitness.score_program(p, data)))
    assert len(archive) <= 2 and archive.ranked()[0].program.source == GOOD


# ---- target bank ----------------------------------------------------------------------------
def test_target_bank_slots_masks_and_normalization(spec, data):
    bank = TargetBank(spec, max_dim=8, n_tasks=2)
    good, clock, pos = (Program(s, spec) for s in (GOOD, CLOCK, POS_ONLY))
    assert sorted(bank.set_programs([(good, [0]), (clock, [0, 1])], data)) == [0, 1, 2, 3]
    priv = data.priv[:4]
    task = torch.tensor([0, 0, 1, 1])[:, None].expand(4, data.t)
    target, mask = bank(priv, task)
    assert target.shape == (4, data.t, 8)
    assert mask[0, 0].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]  # task 0: good (3) + clock (1)
    assert mask[2, 0].tolist() == [0, 0, 0, 1, 0, 0, 0, 0]  # task 1: clock only
    full, _ = bank(data.priv, torch.zeros(data.n, data.t, dtype=torch.long))
    assert abs(float(full[..., :4].mean())) < 0.1 and abs(float(full[..., :4].std()) - 1) < 0.1
    # Swapping keeps the surviving program's slots and reuses the freed ones.
    slots_before = bank._programs[clock.pid][1].tolist()
    new = bank.set_programs([(clock, [0, 1]), (pos, [1])], data)
    assert bank._programs[clock.pid][1].tolist() == slots_before
    assert sorted(new) == [0, 1] and bank.used_dims == 3
    with pytest.raises(ValueError):
        bank.set_programs([(Program(identity_source(spec), spec, max_dim=64), [0])])


def test_target_bank_tracks_head_error(spec, data):
    bank = TargetBank(spec, max_dim=4)
    bank.set_programs([(Program(GOOD, spec), [0])], data)
    bank.observe_error(torch.tensor([10.0, 20.0, 30.0, 0.0]), torch.tensor([100.0, 100.0, 100.0, 0.0]))
    diag = bank.diagnostics()
    assert diag == pytest.approx({"to_goal[0]": 0.1, "to_goal[1]": 0.2, "goal_dist": 0.3})
