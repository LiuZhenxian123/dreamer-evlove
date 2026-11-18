"""Glue between the training loop and the abstraction search.

At configured points in training this (1) snapshots replay into a scoring dataset together
with the world model's current latent features, (2) lets each task's Evolver re-score its
archive on that data and evolve further, (3) swaps the selected abstractions into the
world model's target bank, and (4) folds the results into the cross-task library.
"""

import json
import pathlib

import torch

from sea import tasks as task_text
from sea.archive import load_programs
from sea.evolve import Evolver
from sea.fitness import Dataset
from sea.library import Library
from sea.llm import make_llm


@torch.no_grad()
def build_dataset(agent, replay_buffer, n_batches, spec, meta=None):
    """Sample replay sequences and attach the current world model's latent features."""
    keys = ("priv", "action", "reward", "is_first")
    parts = {k: [] for k in keys + ("feat", "task")}
    was_training = agent.training
    agent.eval()
    for _ in range(int(n_batches)):
        data, _, initial = replay_buffer.sample(prefer_online=False)
        for k in keys:
            parts[k].append(data[k].detach().float().cpu())
        if "task_id" in data.keys():
            parts["task"].append(data["task_id"].detach().cpu())
        parts["feat"].append(agent.latent_features(data, initial).cpu())
    agent.train(was_training)
    cat = {k: torch.cat(v) if v else None for k, v in parts.items()}
    return Dataset(cat["priv"], cat["action"], cat["reward"], cat["is_first"], cat["feat"], cat["task"], spec, meta)


class SeaRuntime:
    def __init__(self, config, priv_spec, task_names, total_steps, logdir, logger, seed=0):
        self.cfg = config
        self.spec = priv_spec
        self.task_names = list(task_names)
        self.logdir = pathlib.Path(logdir) / "sea"
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.enabled = bool(config.enabled)
        self.mode = str(config.mode)
        if self.mode not in ("fixed", "evolve"):
            raise ValueError(f"sea.mode must be 'fixed' or 'evolve', got {self.mode!r}")
        self.budget = int(config.budget_dims)
        self.max_dim = int(config.max_dim)
        self._evolve_at = sorted(int(f * total_steps) for f in config.evolve_at)
        self._dump_at = sorted(int(f * total_steps) for f in config.dump_data_at)
        self._min_sequences = int(config.min_sequences)
        self.library = None
        self.evolvers = {}
        self.round = 0
        if self.enabled and self.mode == "evolve":
            llm = make_llm(config.llm, self.logdir)
            if config.library:
                self.library = Library(config.library, dup_r2=float(config.library_dup_r2))
            for i, name in enumerate(self.task_names):
                self.evolvers[i] = Evolver(
                    self.spec, task_text.describe(name), llm, config.evolver,
                    logdir=self.logdir / name, library=self.library, seed=seed + i, name=name,
                )  # fmt: skip
            self.llm = llm

    # -- startup ---------------------------------------------------------------------------
    def _fixed_programs(self):
        """{task index: [Program]} from `sea.programs` (one path for all tasks, or a per-task mapping)."""
        src = self.cfg.programs
        if not src:
            return {}
        cap = int(self.cfg.evolver.max_program_dim) if self.mode == "evolve" else self.max_dim
        if isinstance(src, str):
            programs = load_programs(src, self.spec, max_dim=cap)
            return {i: programs for i in range(len(self.task_names))}
        return {
            i: load_programs(src[name], self.spec, max_dim=cap) for i, name in enumerate(self.task_names) if name in src
        }

    def start(self, agent):
        if not self.enabled:
            return
        fixed = self._fixed_programs()
        if self.mode == "fixed" and not fixed:
            raise ValueError("sea.mode=fixed needs sea.programs (a .py file or an archive/library JSON)")
        if fixed:
            pairs = [(p, [i]) for i, programs in fixed.items() for p in programs]
            agent.set_abstractions(pairs)
            self._write_active(agent, 0)
        self._seed_sources = {i: [p.source for p in programs] for i, programs in fixed.items()}

    # -- per-step hook ---------------------------------------------------------------------
    def step(self, step, agent, replay_buffer):
        if self._dump_at and step >= self._dump_at[0]:
            self._dump_at.pop(0)
            out = self.logdir.parent / "datasets"
            out.mkdir(exist_ok=True)
            data = build_dataset(agent, replay_buffer, self.cfg.data_batches, self.spec, {"tasks": self.task_names})
            data.save(out / f"step_{step}.pt")
            print(f"[sea] saved scoring dataset ({data.n} sequences) to {out / f'step_{step}.pt'}")
        if self.enabled and self.mode == "evolve" and self._evolve_at and step >= self._evolve_at[0]:
            self._evolve_at.pop(0)
            self.evolve(step, agent, replay_buffer)

    def evolve(self, step, agent, replay_buffer):
        data = build_dataset(agent, replay_buffer, self.cfg.data_batches, self.spec, {"tasks": self.task_names})
        diagnostics = agent.targets.diagnostics() or None
        chosen = {}
        for i, evolver in self.evolvers.items():
            task_data = data.for_task(i)
            if task_data.n < self._min_sequences:
                print(f"[sea] {evolver.name}: only {task_data.n} sequences, skipping this round")
                continue
            if self.round == 0 and self._seed_sources.get(i):
                evolver.add_sources(self._seed_sources[i], task_data)
            archive = evolver.run(task_data, diagnostics=diagnostics)
            picked = archive.select_active(task_data, self.budget, float(self.cfg.max_redundancy))
            chosen[i] = picked
            if self.library is not None and bool(self.cfg.library_write):
                info = self.library.absorb(evolver.name, archive, task_data, picked)
                print(f"[sea] library after {evolver.name}: {info}")
        if not chosen:
            return
        pairs = self._fit_budget(chosen)
        new_slots = agent.set_abstractions(pairs, data)
        self.round += 1
        if self.library is not None and bool(self.cfg.library_write):
            if int(self.cfg.insights_every) and self.round % int(self.cfg.insights_every) == 0:
                self.library.update_insights(self.llm)
            if int(self.cfg.refactor_every) and self.round % int(self.cfg.refactor_every) == 0:
                print(f"[sea] library refactor: {self.library.refactor(self.llm, data)}")
            with (self.logdir / "library_stats.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, **self.library.reuse_stats()}) + "\n")
        best = [e.archive.ranked()[0].score for e in self.evolvers.values() if len(e.archive)]
        self.logger.scalar("sea/best_score", sum(best) / max(1, len(best)))
        self.logger.scalar("sea/active_dims", agent.targets.used_dims)
        self.logger.scalar("sea/new_slots", len(new_slots))
        self.logger.scalar("sea/llm_calls", self.llm.calls)
        self.logger.scalar("sea/round", self.round)
        self._write_active(agent, step)

    def _fit_budget(self, chosen):
        """(program, tasks) pairs within the bank's slot count, dropping the weakest first."""
        owners, entries = {}, {}
        for task, picked in chosen.items():
            for e in picked:
                owners.setdefault(e.program.pid, []).append(task)
                entries.setdefault(e.program.pid, e)
        order = sorted(entries.values(), key=lambda e: -e.score)
        pairs, used = [], 0
        for e in order:
            if used + e.program.dim > self.max_dim:
                continue
            pairs.append((e.program, owners[e.program.pid]))
            used += e.program.dim
        return pairs

    def _write_active(self, agent, step):
        with (self.logdir / "active.json").open("w") as f:
            json.dump({"step": step, "programs": agent.targets.describe()}, f, indent=1)
        with (self.logdir / "active_history.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, "features": [p["features"] for p in agent.targets.describe()]}) + "\n")

    def log_metrics(self, agent):
        if not self.enabled:
            return
        diag = agent.targets.diagnostics()
        if diag:
            self.logger.scalar("sea/head_error_mean", sum(diag.values()) / len(diag))
