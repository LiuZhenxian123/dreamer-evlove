"""Training-free fitness for abstraction programs, computed on replay data.

An abstraction z = phi(x) of the privileged state x is scored against the conditions of
an approximate MDP homomorphism, plus what a pixel-based world model can make of it:

closure   z a few steps ahead is predictable from (z, actions)   transition-respecting
control   the actions improve that prediction                    not an exogenous signal
reward    return over a short horizon is predictable from z     reward-respecting
visible   z is decodable from the world model's latent state    inferable from pixels
known     z is *linearly* decodable from that latent            already represented
compact   few dimensions, short program

All predictors are closed-form ridge regressions on random Fourier features, fit on a
train split of sequences and scored by R^2 on held-out sequences, so one candidate costs
well under a second and no gradient steps. Every term is scale-free (R^2 on standardized
targets), so a program cannot improve its score by shrinking its outputs.

These are proxies. Buffer-averaged L2 errors of a small regressor are much weaker than the
sup-norm/Wasserstein conditions the value-loss bounds assume; whether the score tracks
downstream return is an empirical question that `python -m sea.cli rankcorr` answers.
"""

import math

import torch

DEFAULT_WEIGHTS = {"closure": 1.0, "reward": 1.0, "visible": 1.0, "control": 0.25}
EPS = 1e-8


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


class Ridge:
    """Ridge regression on [x, random Fourier features of x], inputs standardized."""

    def __init__(self, n_features=384, reg=1e-2, seed=0, linear_only=False, max_inputs=128):
        self.n_features, self.reg, self.seed = int(n_features), float(reg), int(seed)
        self.linear_only, self.max_inputs = bool(linear_only), int(max_inputs)

    def _phi(self, x):
        x = (x - self.mu) / self.sd
        if self.proj is not None:
            x = x @ self.proj
        if self.linear_only:
            feats = [x]
        else:
            feats = [x, math.sqrt(2.0 / self.n_features) * torch.cos(x @ self.w + self.b)]
        return torch.cat(feats + [torch.ones_like(x[:, :1])], dim=1)

    def fit(self, x, y):
        x, y = x.double(), y.double()
        g = torch.Generator().manual_seed(self.seed)
        self.mu, self.sd = x.mean(0), x.std(0).clamp_min(1e-6)
        self.proj = None
        d = x.shape[1]
        if not self.linear_only and d > self.max_inputs:
            # PCA to keep the random-feature kernel meaningful for wide inputs (latent states).
            _, _, v = torch.pca_lowrank((x - self.mu) / self.sd, q=self.max_inputs, center=False, niter=2)
            self.proj, d = v, self.max_inputs
        if not self.linear_only:
            self.w = torch.randn(d, self.n_features, generator=g, dtype=torch.float64) / math.sqrt(d)
            self.b = 2 * math.pi * torch.rand(self.n_features, generator=g, dtype=torch.float64)
        f = self._phi(x)
        a = f.T @ f + self.reg * len(x) * torch.eye(f.shape[1], dtype=torch.float64)
        self.coef = torch.linalg.solve(a, f.T @ y)
        return self

    def predict(self, x):
        return self._phi(x.double()) @ self.coef


def heldout_mse(x, y, train, test, **kw):
    """Per-output-dimension test MSE of a ridge fit on the train rows."""
    model = Ridge(**kw).fit(x[train], y[train])
    return ((model.predict(x[test]) - y[test].double()) ** 2).mean(0)


def _r2(mse, y, train, test):
    var = ((y[test].double() - y[train].double().mean(0)) ** 2).mean(0)
    return 1.0 - mse / var.clamp_min(EPS)


def split_rows(n_seq, seq_len, test_frac=0.3, seed=0):
    """Train/test split by whole sequences, returned as (n_seq, seq_len) boolean masks."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_seq, generator=g)
    n_test = max(1, int(round(test_frac * n_seq)))
    test = torch.zeros(n_seq, dtype=torch.bool)
    test[perm[:n_test]] = True
    test = test[:, None].expand(n_seq, seq_len)
    return ~test, test


def nstep_return(reward, is_first, horizon=15, gamma=0.99):
    """Discounted return over the next `horizon` steps and a mask of where it is defined."""
    n, t = reward.shape
    h = min(horizon, t - 1)
    ret = torch.zeros(n, t - h, dtype=torch.float64)
    ok = torch.ones(n, t - h, dtype=torch.bool)
    for k in range(1, h + 1):
        ret += gamma ** (k - 1) * reward[:, k : t - h + k].double()
        ok &= ~is_first[:, k : t - h + k].bool()
    return ret, ok, h


class Dataset:
    """Replay sequences used for scoring. Tensors are (N, T, ...), on CPU.

    ``action[:, t]`` is the action that led to step t (replay convention), so the
    transition out of step t uses ``action[:, t + 1]``.
    """

    def __init__(self, priv, action, reward, is_first, feat=None, task=None, spec=None, meta=None):
        self.priv = priv.float().cpu()
        self.action = action.float().cpu()
        self.reward = reward.float().cpu().reshape(priv.shape[:2])
        self.is_first = is_first.cpu().reshape(priv.shape[:2]).bool()
        self.feat = None if feat is None else feat.float().cpu()
        self.task = None if task is None else task.cpu().reshape(priv.shape[:2]).long()
        self.spec = spec
        self.meta = dict(meta or {})
        self.n, self.t = self.priv.shape[:2]
        self._cache = {}

    def subset(self, rows):
        pick = lambda v: None if v is None else v[rows]  # noqa: E731
        return Dataset(
            self.priv[rows], self.action[rows], self.reward[rows], self.is_first[rows],
            pick(self.feat), pick(self.task), self.spec, self.meta,
        )  # fmt: skip

    def for_task(self, task_id):
        if self.task is None:
            return self
        return self.subset(self.task[:, 0] == int(task_id))

    def save(self, path):
        torch.save(
            {
                "priv": self.priv, "action": self.action, "reward": self.reward, "is_first": self.is_first,
                "feat": None if self.feat is None else self.feat.half(), "task": self.task,
                "spec": None if self.spec is None else self.spec.to_json(), "meta": self.meta,
            },
            path,
        )  # fmt: skip

    @classmethod
    def load(cls, path):
        from sea.privileged import PrivSpec

        d = torch.load(path, map_location="cpu", weights_only=False)
        spec = None if d["spec"] is None else PrivSpec.from_json(d["spec"])
        return cls(d["priv"], d["action"], d["reward"], d["is_first"], d["feat"], d["task"], spec, d["meta"])

    def field_stats(self):
        """Per-field min/max/std over the data, shown to the LLM so it knows what is live."""
        flat = self.priv.reshape(-1, self.priv.shape[-1])
        out = {}
        for name, sl in self.spec.slices.items():
            v = flat[:, sl]
            out[name] = {"min": float(v.min()), "max": float(v.max()), "std": float(v.std(0).mean())}
        return out


def standardize(z):
    """symlog + per-dimension standardization; returns (z_std, live mask, finite fraction)."""
    flat = z.reshape(-1, z.shape[-1])
    finite = torch.isfinite(flat)
    valid_frac = float(finite.float().mean())
    v = symlog(torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)).double()
    mu, sd = v.mean(0), v.std(0)
    live = sd > 1e-6
    v = (v - mu) / sd.clamp_min(1e-6)
    return v.reshape(z.shape).float(), live, valid_frac


def score_program(program, data, weights=None, seed=0, horizon=15, closure_steps=4, dim_scale=48.0,
                  size_scale=1500.0):  # fmt: skip
    """All fitness terms for one program on one dataset. Returns a JSON-serializable dict."""
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    z_raw = program(data.priv)
    z, live, valid_frac = standardize(z_raw)
    names = program.dim_names
    out = {
        "dim": program.dim,
        "size": program.size,
        "valid_frac": valid_frac,
        "live_frac": float(live.float().mean()),
        "dead_dims": [n for n, ok in zip(names, live) if not ok],
    }
    if valid_frac < 0.99 or not bool(live.any()):
        out.update({"score": 0.0, "reject": "non-finite outputs" if valid_frac < 0.99 else "all outputs constant"})
        return out
    z = z[..., live]
    live_names = [n for n, ok in zip(names, live) if ok]
    n, t, d = z.shape
    train, test = split_rows(n, t, seed=seed)

    # --- closure and control: z after k steps from (z, the k actions) vs from z alone ----------
    k = max(1, min(int(closure_steps), t - 1))
    pair_ok = torch.ones(n, t - k, dtype=torch.bool)
    for j in range(1, k + 1):
        pair_ok &= ~data.is_first[:, j : t - k + j]
    z0, z1 = z[:, : t - k], z[:, k:]
    acts = torch.cat([data.action[:, j : t - k + j] for j in range(1, k + 1)], dim=-1)
    tr, te = (train[:, : t - k] & pair_ok).reshape(-1), (test[:, : t - k] & pair_ok).reshape(-1)
    z0f, z1f, af = z0.reshape(-1, d), z1.reshape(-1, d), acts.reshape(-1, acts.shape[-1])
    dzf = z1f - z0f
    dz_var = dzf[tr].double().var(0)
    moving = dz_var > 1e-6  # standardized z has unit variance, so this is relative
    dzn = dzf / dz_var.sqrt().clamp_min(1e-6).float()
    mse_za = heldout_mse(torch.cat([z0f, af], 1), dzn, tr, te, seed=seed)
    mse_z = heldout_mse(z0f, dzn, tr, te, seed=seed)
    # Predictable beyond persistence (R^2 of the change) AND in absolute terms (R^2 of the
    # value). The first alone gives 0.5 to white noise, which mean-reverts predictably; the
    # second alone gives ~1 to anything that merely changes slowly.
    r2_change = _r2(mse_za, dzn, tr, te)
    r2_value = 1.0 - mse_za * dz_var / z1f[te].double().var(0).clamp_min(EPS)
    closure_d = torch.minimum(r2_change, r2_value).clamp(0, 1)
    control_d = (1.0 - mse_za / mse_z.clamp_min(EPS)).clamp(0, 1)
    closure_d = torch.where(moving, closure_d, torch.ones_like(closure_d))
    control_d = torch.where(moving, control_d, torch.zeros_like(control_d))

    # --- reward sufficiency: short-horizon return from z -----------------------------------
    ret, ret_ok, h = nstep_return(data.reward, data.is_first, horizon)
    zr = z[:, : t - h]
    rtr, rte = (train[:, : t - h] & ret_ok).reshape(-1), (test[:, : t - h] & ret_ok).reshape(-1)
    zrf, retf = zr.reshape(-1, d), ret.reshape(-1, 1).float()
    reward = float(_r2(heldout_mse(zrf, retf, rtr, rte, seed=seed), retf, rtr, rte).clamp(0, 1))
    reward_d = torch.stack([
        _r2(heldout_mse(zrf[:, i : i + 1], retf, rtr, rte, seed=seed), retf, rtr, rte).clamp(0, 1)[0] for i in range(d)
    ])

    # --- visibility: decode z from the world model's latent --------------------------------
    visible = known = None
    visible_d = known_d = None
    if data.feat is not None:
        ff, zf = data.feat.reshape(n * t, -1), z.reshape(-1, d)
        ftr, fte = train.reshape(-1), test.reshape(-1)
        known_d = _r2(heldout_mse(ff, zf, ftr, fte, linear_only=True, seed=seed), zf, ftr, fte).clamp(0, 1)
        visible_d = _r2(heldout_mse(ff, zf, ftr, fte, n_features=768, seed=seed), zf, ftr, fte).clamp(0, 1)
        visible_d = torch.maximum(visible_d, known_d)
        visible, known = float(visible_d.mean()), float(known_d.mean())

    corr = torch.corrcoef(z.reshape(-1, d).T.double()) if d > 1 else torch.ones(1, 1, dtype=torch.float64)
    redundancy = float((corr.abs().sum() - d) / max(d * (d - 1), 1))
    out.update({
        "closure": float(closure_d.mean()),
        "control": float(control_d.mean()),
        "reward": reward,
        "visible": visible,
        "known": known,
        "gap": None if visible is None else visible - known,
        "static_frac": float((~moving).float().mean()),
        "redundancy": redundancy,
        "per_dim": {
            name: {
                "closure": float(closure_d[i]),
                "control": float(control_d[i]),
                "reward": float(reward_d[i]),
                **({} if visible_d is None else {"visible": float(visible_d[i]), "known": float(known_d[i])}),
            }
            for i, name in enumerate(live_names)
        },
    })
    out["score"] = combine(out, weights, dim_scale, size_scale)
    return out


def combine(m, weights=None, dim_scale=48.0, size_scale=1500.0, floor=0.02):
    """Scalar score: weighted geometric mean of the terms times mild compactness penalties."""
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    logsum = wsum = 0.0
    for key, w in weights.items():
        value = m.get(key)
        if value is None or w <= 0:
            continue
        logsum += w * math.log(max(float(value), floor))
        wsum += w
    if wsum == 0:
        return 0.0
    compact = math.exp(-m["dim"] / dim_scale) * math.exp(-m["size"] / size_scale)
    return math.exp(logsum / wsum) * compact * m.get("live_frac", 1.0)


def explained(z_from, z_to, seed=0):
    """Mean held-out R^2 of predicting abstraction `z_to` from abstraction `z_from` (N, T, d)."""
    a, live_a, _ = standardize(z_from)
    b, live_b, _ = standardize(z_to)
    if not bool(live_a.any()) or not bool(live_b.any()):
        return 0.0
    a, b = a[..., live_a], b[..., live_b]
    train, test = split_rows(a.shape[0], a.shape[1], seed=seed)
    af, bf = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    tr, te = train.reshape(-1), test.reshape(-1)
    return float(_r2(heldout_mse(af, bf, tr, te, seed=seed), bf, tr, te).clamp(0, 1).mean())
