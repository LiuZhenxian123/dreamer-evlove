"""Turns the active abstraction programs into normalized training targets.

The bank owns a fixed number of target slots so the world model's semantic head (and the
compiled update function) keep a fixed shape while the set of programs changes during
training. A program that survives a swap keeps its slots, so what the head has learned
about it is preserved.
"""

import torch
from torch import nn

from sea.fitness import symlog


class TargetBank(nn.Module):
    def __init__(self, spec, max_dim, n_tasks=1, decay=0.99, eps=1e-4, err_decay=0.999):
        super().__init__()
        self.spec = spec
        self.max_dim, self.n_tasks = int(max_dim), int(n_tasks)
        self.decay, self.eps, self.err_decay = float(decay), float(eps), float(err_decay)
        # Running first and second moments of the symlog-ed targets. Tracking moments (rather
        # than averaging per-batch variances) keeps the between-batch variance, which dominates
        # when a batch holds only a few correlated sequences.
        self.register_buffer("mean", torch.zeros(self.max_dim))
        self.register_buffer("sq", torch.ones(self.max_dim))
        self.register_buffer("count", torch.zeros(self.max_dim))  # batches seen per slot
        self.register_buffer("active", torch.zeros(self.n_tasks, self.max_dim, dtype=torch.bool))
        self.register_buffer("err", torch.ones(self.max_dim))  # EMA of normalized squared error
        self.register_buffer("err_seen", torch.zeros(self.max_dim, dtype=torch.bool))
        self._programs = {}  # pid -> (Program, slot index tensor, task ids)

    @property
    def programs(self):
        return [p for p, _, _ in self._programs.values()]

    @property
    def used_dims(self):
        return sum(p.dim for p in self.programs)

    def set_programs(self, assignments, data=None):
        """assignments: iterable of (Program, task ids). Returns the slots that were (re)assigned.

        With `data` (a fitness.Dataset), normalization statistics of new slots are computed
        from it; otherwise they are initialized from the first training batch.
        """
        merged = {}
        for program, tasks in assignments:
            prev = merged.get(program.pid)
            merged[program.pid] = (program, sorted(set(tasks) | set(prev[1] if prev else ())))
        total = sum(p.dim for p, _ in merged.values())
        if total > self.max_dim:
            raise ValueError(f"programs need {total} target dims but the bank has {self.max_dim}")
        kept = {pid: v for pid, v in self._programs.items() if pid in merged}
        taken = torch.zeros(self.max_dim, dtype=torch.bool)
        for _, slots, _ in kept.values():
            taken[slots] = True
        free = (~taken).nonzero().flatten().tolist()
        device = self.mean.device
        new_slots, programs = [], {}
        for pid, (program, tasks) in merged.items():
            if pid in kept:
                slots = kept[pid][1]
            else:
                slots, free = torch.tensor(free[: program.dim]), free[program.dim :]
                new_slots += slots.tolist()
            programs[pid] = (program, slots, tasks)
        self._programs = programs
        self.active.zero_()
        for _, slots, tasks in programs.values():
            for task in tasks:
                self.active[task, slots.to(device)] = True
        if new_slots:
            idx = torch.tensor(new_slots, device=device)
            self.count[idx] = 0.0
            self.err[idx] = 1.0
            self.err_seen[idx] = False
            self.mean[idx] = 0.0
            self.sq[idx] = 1.0
        if data is not None and new_slots:
            raw = self._raw(data.priv.to(device))
            v = symlog(torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)).reshape(-1, self.max_dim)
            idx = torch.tensor(new_slots, device=device)
            self.mean[idx] = v.mean(0)[idx]
            self.sq[idx] = (v * v).mean(0)[idx]
            self.count[idx] = 1.0 / (1.0 - self.decay)  # as trusted as a converged running average
        return new_slots

    def _raw(self, priv):
        out = torch.zeros(priv.shape[:-1] + (self.max_dim,), dtype=torch.float32, device=priv.device)
        for program, slots, _ in self._programs.values():
            out[..., slots.to(priv.device)] = program(priv)
        return out

    @torch.no_grad()
    def forward(self, priv, task_id=None):
        """priv (B, T, P), task_id (B, T) or None -> normalized target and mask, both (B, T, D)."""
        raw = self._raw(priv.float())
        finite = torch.isfinite(raw)
        v = symlog(torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0))
        if task_id is None:
            active = self.active[0].expand_as(finite)
        else:
            active = self.active[task_id.long().reshape(priv.shape[:-1])]
        mask = finite & active
        if self.training:
            n = mask.sum((0, 1)).float()
            seen = n > 0
            denom = n.clamp_min(1.0)
            bmean = (v * mask).sum((0, 1)) / denom
            bsq = (v * v * mask).sum((0, 1)) / denom
            # Plain average over the first batches, exponential average afterwards.
            rate = (1.0 / (self.count + 1.0)).clamp_min(1.0 - self.decay) * seen.float()
            self.mean.lerp_(bmean, rate)
            self.sq.lerp_(bsq, rate)
            self.count += seen.float()
        var = (self.sq - self.mean**2).clamp_min(0.0)
        target = (v - self.mean) / torch.sqrt(var + self.eps)
        return target * mask, mask.float()

    @torch.no_grad()
    def observe_error(self, err_sum, count):
        """Track per-slot squared error of the head on normalized targets (about 1 - R^2)."""
        seen = count > 0
        batch = err_sum / count.clamp_min(1.0)
        first = seen & ~self.err_seen
        later = seen & self.err_seen
        self.err.copy_(torch.where(first, batch, torch.where(later, torch.lerp(batch, self.err, self.err_decay), self.err)))  # fmt: skip  # noqa: E501
        self.err_seen |= seen

    def diagnostics(self):
        """{feature name: world-model prediction error} for the active targets."""
        err = self.err.detach().float().cpu()
        out = {}
        for program, slots, _ in self._programs.values():
            for name, slot in zip(program.dim_names, slots.tolist()):
                out[name] = float(err[slot].clamp(0, 2))
        return out

    def describe(self):
        return [
            {**program.to_json(), "slots": slots.tolist(), "tasks": tasks, "features": program.dim_names}
            for program, slots, tasks in self._programs.values()
        ]
