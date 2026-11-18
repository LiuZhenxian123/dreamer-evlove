"""Population of scored abstraction programs with Pareto-based survival."""

import json
import random

from sea import fitness
from sea.program import Program, ProgramError

OBJECTIVES = ("closure", "reward", "visible", "control")


class Entry:
    def __init__(self, program, metrics=None, generation=0, origin="init", parents=()):
        self.program = program
        self.metrics = metrics or {}
        self.generation = int(generation)
        self.origin = origin
        self.parents = list(parents)

    @property
    def score(self):
        return float(self.metrics.get("score", 0.0))

    def objectives(self):
        vals = [self.metrics.get(k) for k in OBJECTIVES]
        vals = [0.0 if v is None else float(v) for v in vals]
        return vals + [-float(self.metrics.get("dim", 0))]

    def to_json(self):
        return {
            **self.program.to_json(),
            "metrics": self.metrics,
            "generation": self.generation,
            "origin": self.origin,
            "parents": self.parents,
        }


def dominates(a, b):
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


class Archive:
    def __init__(self, capacity=48):
        self.capacity = int(capacity)
        self.entries = {}

    def __len__(self):
        return len(self.entries)

    def __contains__(self, pid):
        return pid in self.entries

    def add(self, entry):
        if entry.program.pid in self.entries or entry.metrics.get("reject"):
            return False
        self.entries[entry.program.pid] = entry
        self._prune()
        return entry.program.pid in self.entries

    def ranked(self):
        return sorted(self.entries.values(), key=lambda e: -e.score)

    def front(self):
        items = list(self.entries.values())
        objs = [e.objectives() for e in items]
        return [e for e, o in zip(items, objs) if not any(dominates(p, o) for p in objs)]

    def _prune(self):
        if len(self.entries) <= self.capacity:
            return
        keep = {e.program.pid for e in self.front()}
        for e in self.ranked():
            if len(keep) >= self.capacity:
                break
            keep.add(e.program.pid)
        # If the front alone exceeds capacity, keep its best-scoring members.
        if len(keep) > self.capacity:
            keep = {e.program.pid for e in self.ranked() if e.program.pid in keep}
            keep = set(list(sorted(keep, key=lambda p: -self.entries[p].score))[: self.capacity])
        self.entries = {pid: e for pid, e in self.entries.items() if pid in keep}

    def rescore(self, data, **score_kw):
        """Re-evaluate everything on new data (the data distribution shifts as the agent improves)."""
        for pid in list(self.entries):
            entry = self.entries[pid]
            try:
                entry.metrics = fitness.score_program(entry.program, data, **score_kw)
            except ProgramError as e:
                entry.metrics = {"score": 0.0, "reject": str(e), "dim": entry.program.dim}
            if entry.metrics.get("reject"):
                del self.entries[pid]

    def sample_parents(self, rng: random.Random, k=1, explore=0.3):
        """Mostly exploit the best scores, sometimes draw from the Pareto front for diversity."""
        ranked = self.ranked()
        if not ranked:
            return []
        top = ranked[: max(3, len(ranked) // 4)]
        front = self.front()
        picks = []
        for _ in range(20 * k):
            if len(picks) >= min(k, len(ranked)):
                break
            cand = rng.choice(front if rng.random() < explore else top)
            if cand not in picks:
                picks.append(cand)
        return picks

    def select_active(self, data, budget_dims, max_redundancy=0.9, min_score=0.0):
        """Greedy, redundancy-aware choice of programs to supervise the world model with."""
        chosen, used, cache = [], 0, {}
        for entry in self.ranked():
            if entry.score <= min_score or used + entry.program.dim > budget_dims:
                continue
            z = cache.setdefault(entry.program.pid, entry.program(data.priv))
            if chosen:
                import torch

                joint = torch.cat([cache[c.program.pid] for c in chosen], dim=-1)
                if fitness.explained(joint, z) > max_redundancy:
                    continue
            chosen.append(entry)
            used += entry.program.dim
        return chosen

    def to_json(self):
        return [e.to_json() for e in self.ranked()]

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1)

    @classmethod
    def load(cls, path, spec, capacity=48, max_dim=None):
        archive = cls(capacity)
        with open(path) as f:
            items = json.load(f)
        for item in items:
            kw = {} if max_dim is None else {"max_dim": max_dim}
            program = Program(item["source"], spec, item.get("helpers"), item.get("name"), **kw)
            archive.entries[program.pid] = Entry(
                program, item.get("metrics"), item.get("generation", 0), item.get("origin", "loaded"),
                item.get("parents", ()),
            )  # fmt: skip
        return archive


def load_programs(path, spec, max_dim=None):
    """Programs from an archive/library JSON file or a single .py file."""
    kw = {} if max_dim is None else {"max_dim": max_dim}
    if str(path).endswith(".py"):
        with open(path) as f:
            return [Program(f.read(), spec, **kw)]
    with open(path) as f:
        items = json.load(f)
    return [Program(i["source"], spec, i.get("helpers"), i.get("name"), **kw) for i in items]
