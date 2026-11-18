"""Evolution of abstraction programs: LLM proposes, the training-free fitness selects."""

import json
import pathlib
import random
import time

from sea import fitness, prompts
from sea.archive import Archive, Entry
from sea.llm import code_blocks
from sea.program import Program, ProgramError


class Evolver:
    """One task's search state. Call :meth:`run` whenever new replay data is available."""

    def __init__(self, spec, task_text, llm, cfg, logdir=None, library=None, seed=0, name="task"):
        self.spec, self.task_text, self.llm, self.library = spec, task_text, llm, library
        self.name = name
        self.per_call = int(cfg.per_call)
        self.calls_per_gen = int(cfg.calls_per_gen)
        self.generations = int(cfg.generations)
        self.max_dim = int(cfg.max_program_dim)
        self.weights = dict(cfg.weights)
        self.mode_probs = dict(cfg.modes)
        self.archive = Archive(int(cfg.capacity))
        self.rng = random.Random(seed)
        self.generation = 0
        self.failures = []
        self._library_seen = set()
        self.stats = {"proposed": 0, "valid": 0, "added": 0, "library_tried": 0, "library_kept": 0}
        self.logdir = None if logdir is None else pathlib.Path(logdir)
        if self.logdir is not None:
            self.logdir.mkdir(parents=True, exist_ok=True)

    # -- helpers ---------------------------------------------------------------------------
    def _helpers(self):
        return {} if self.library is None else dict(self.library.helpers)

    def _compile(self, source):
        return Program(source, self.spec, helpers=self._helpers(), max_dim=self.max_dim)

    def _score(self, program, data):
        return fitness.score_program(program, data, weights=self.weights)

    def _consider(self, source, data, origin, parents=()):
        """Compile, score and try to add one candidate. Returns the Entry, or None if rejected."""
        self.stats["proposed"] += 1
        try:
            program = self._compile(source)
            if program.pid in self.archive:
                return None
            metrics = self._score(program, data)
        except ProgramError as e:
            self._fail(str(e))
            return None
        if metrics.get("reject"):
            self._fail(metrics["reject"])
            return None
        self.stats["valid"] += 1
        entry = Entry(program, metrics, self.generation, origin, parents)
        if self.archive.add(entry):
            self.stats["added"] += 1
        return entry

    def _fail(self, message):
        self.failures = (self.failures + [message[:300]])[-5:]

    def _pick_mode(self):
        if len(self.archive) < 2:
            return "init"
        modes = [m for m in self.mode_probs if m != "crossover" or len(self.archive) >= 2]
        return self.rng.choices(modes, weights=[self.mode_probs[m] for m in modes])[0]

    def _propose(self, data, stats, diagnostics, index):
        mode = self._pick_mode()
        parents = self.archive.sample_parents(self.rng, 2 if mode == "crossover" else 1)
        if mode == "crossover" and len(parents) < 2:
            mode = "mutate"
        messages = prompts.build(
            self.task_text, self.spec, stats, mode, self.per_call,
            parents=parents,
            elites=[e for e in self.archive.ranked()[:3] if e not in parents],
            failures=self.failures,
            diagnostics=diagnostics,
            insights="" if self.library is None else self.library.insights,
            helper_docs="" if self.library is None else self.library.helper_docs(),
        )  # fmt: skip
        text = self.llm.chat(messages, tag=f"{self.name}/gen{self.generation}/{mode}", index=index)
        blocks = code_blocks(text)
        if not blocks:
            self._fail("the response contained no ```python code block")
        parent_ids = [p.program.pid for p in parents]
        return [self._consider(src, data, mode, parent_ids) for src in blocks]

    # -- public API ------------------------------------------------------------------------
    def seed_from_library(self, data):
        """Re-score transferable abstractions from earlier tasks on this task's own data."""
        if self.library is None:
            return
        for source in self.library.retrieve(self.spec, exclude_task=self.name):
            if source in self._library_seen:
                continue
            self._library_seen.add(source)
            self.stats["library_tried"] += 1
            entry = self._consider(source, data, "library")
            if entry is not None and entry.program.pid in self.archive:
                self.stats["library_kept"] += 1

    def add_sources(self, sources, data, origin="seed"):
        return [self._consider(src, data, origin) for src in sources]

    def run(self, data, generations=None, diagnostics=None):
        """Re-score the archive on `data`, then evolve for `generations`. Returns the archive."""
        start = time.time()
        stats = data.field_stats()
        self.archive.rescore(data, weights=self.weights)
        self.seed_from_library(data)
        for _ in range(self.generations if generations is None else int(generations)):
            for call in range(self.calls_per_gen):
                self._propose(data, stats, diagnostics, index=self.generation * 1000 + call)
            self.generation += 1
            self._log(data, time.time() - start)
        return self.archive

    def one_shot(self, n, stats=None, index=0):
        """`n` programs from the LLM with no scoring or feedback (the one-shot baseline)."""
        programs, attempt = [], 0
        while len(programs) < n and attempt < 4 * max(1, n // self.per_call + 1):
            messages = prompts.build(self.task_text, self.spec, stats, "init", self.per_call)
            for src in code_blocks(self.llm.chat(messages, tag=f"{self.name}/oneshot", index=index + attempt)):
                try:
                    program = self._compile(src)
                except ProgramError:
                    continue
                if all(program.pid != p.pid for p in programs):
                    programs.append(program)
            attempt += 1
        return programs[:n]

    def _log(self, data, seconds):
        ranked = self.archive.ranked()
        record = {
            "task": self.name,
            "generation": self.generation,
            "size": len(ranked),
            "best": ranked[0].score if ranked else 0.0,
            "mean_top5": sum(e.score for e in ranked[:5]) / max(1, len(ranked[:5])),
            "llm_calls": self.llm.calls,
            "llm_usage": dict(self.llm.usage),
            "seconds": round(seconds, 1),
            "data_sequences": data.n,
            **self.stats,
        }
        print(f"[evolve:{self.name}] " + " ".join(f"{k}={v}" for k, v in record.items() if k != "llm_usage"))
        if self.logdir is not None:
            with (self.logdir / "history.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            self.archive.save(self.logdir / "archive.json")
