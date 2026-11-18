"""Cross-task library of abstractions, shared helper code and distilled lessons.

What makes this more than a cache is that reuse is *verified*: an abstraction retrieved for
a new task is re-scored on that task's own replay data before it is used (see
``Evolver.seed_from_library``), and a refactoring is accepted only if the rewritten
programs reproduce the original outputs numerically.
"""

import ast
import json
import pathlib
import re
import time

import torch

from sea import fitness
from sea.llm import code_blocks
from sea.program import HELPERS, TORCH_NAMESPACE, Program, ProgramError, compile_functions

_METRIC_KEYS = ("score", "closure", "control", "reward", "visible", "known", "dim")


class Library:
    def __init__(self, path=None, max_retrieve=12, dup_r2=0.95):
        self.path = None if path is None else pathlib.Path(path)
        self.max_retrieve, self.dup_r2 = int(max_retrieve), float(dup_r2)
        self.entries, self.helpers, self.insights = [], {}, ""
        if self.path is not None and (self.path / "library.json").exists():
            with (self.path / "library.json").open() as f:
                state = json.load(f)
            self.entries, self.helpers = state["entries"], state.get("helpers", {})
            self.insights = state.get("insights", "")

    def save(self):
        if self.path is None:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        tmp = self.path / "library.json.tmp"
        with tmp.open("w") as f:
            json.dump({"entries": self.entries, "helpers": self.helpers, "insights": self.insights}, f, indent=1)
        tmp.replace(self.path / "library.json")

    # -- retrieval -------------------------------------------------------------------------
    @staticmethod
    def _compatible(entry, spec):
        return all(name in spec.slices and spec.slices[name].stop - spec.slices[name].start == dim
                   for name, dim in entry["fields"].items())  # fmt: skip

    @staticmethod
    def _mean_score(entry):
        scores = [t["score"] for t in entry["tasks"].values()]
        return sum(scores) / max(1, len(scores))

    def retrieve(self, spec, exclude_task=None):
        """Sources of abstractions whose inputs exist in `spec`, best average score first."""
        pool = [e for e in self.entries if self._compatible(e, spec)]
        if exclude_task is not None:
            pool = [e for e in pool if set(e["tasks"]) != {exclude_task}]
        pool.sort(key=lambda e: -self._mean_score(e))
        return [e["source"] for e in pool[: self.max_retrieve]]

    def helper_docs(self):
        """One line per shared helper function: signature and docstring."""
        lines = []
        for source in self.helpers.values():
            for node in ast.parse(source).body:
                if isinstance(node, ast.FunctionDef):
                    args = ", ".join(a.arg for a in node.args.args)
                    lines.append(f"- {node.name}({args}): {(ast.get_docstring(node) or '').strip()}")
        return "\n".join(lines)

    # -- growth ----------------------------------------------------------------------------
    def absorb(self, task, archive, data, selected=(), top_k=8):
        """Record a finished search: add new abstractions, credit existing ones that were reused."""
        selected = {e.program.pid for e in selected}
        ranked = archive.ranked()
        candidates = ranked[:top_k] + [e for e in ranked[top_k:] if e.program.pid in selected]
        outputs = {}
        for item in self.entries:
            if not self._compatible(item, data.spec):
                continue
            try:
                outputs[item["id"]] = Program(item["source"], data.spec, self.helpers, max_dim=10**6)(data.priv)
            except ProgramError:
                continue
        added = reused = 0
        for entry in candidates:
            record = {k: entry.metrics.get(k) for k in _METRIC_KEYS}
            record.update({"selected": entry.program.pid in selected, "origin": entry.origin, "time": time.time()})
            z = entry.program(data.priv)
            match = None
            for item in self.entries:
                if item["id"] in outputs and (
                    item["source"] == entry.program.source
                    or min(fitness.explained(outputs[item["id"]], z), fitness.explained(z, outputs[item["id"]]))
                    > self.dup_r2
                ):
                    match = item
                    break
            if match is None:
                match = {
                    "id": entry.program.pid,
                    "source": entry.program.source,
                    "fields": {name: int(data.spec.slices[name].stop - data.spec.slices[name].start)
                               for name in entry.program.fields},
                    "features": entry.program.dim_names,
                    "tasks": {},
                }  # fmt: skip
                self.entries.append(match)
                outputs[match["id"]] = z
                added += 1
            elif task not in match["tasks"]:
                reused += 1
            prev = match["tasks"].get(task)
            if prev is None or (record["score"] or 0) >= (prev["score"] or 0):
                record["per_dim"] = entry.metrics.get("per_dim", {})
                match["tasks"][task] = record
        self.save()
        return {"added": added, "reused": reused, "size": len(self.entries)}

    def reuse_stats(self):
        """How often entries are actually reused across tasks (report this, with a no-library ablation)."""
        n = len(self.entries)
        multi = [e for e in self.entries if len(e["tasks"]) > 1]
        chosen = [e for e in self.entries if sum(bool(t.get("selected")) for t in e["tasks"].values()) > 1]
        return {
            "entries": n,
            "used_on_multiple_tasks": len(multi),
            "selected_on_multiple_tasks": len(chosen),
            "mean_tasks_per_entry": sum(len(e["tasks"]) for e in self.entries) / max(1, n),
            "helpers": len(self.helpers),
        }

    # -- consolidation ---------------------------------------------------------------------
    def update_insights(self, llm, max_items=12):
        """Distil per-feature scores across tasks into short lessons used in later prompts."""
        rows = []
        for e in self.entries:
            for task, rec in e["tasks"].items():
                for feat, m in (rec.get("per_dim") or {}).items():
                    rows.append((m.get("reward", 0) * m.get("closure", 0) * (m.get("visible") or 1), task, feat, m, e))
        if len(rows) < 6:
            return self.insights
        rows.sort(key=lambda r: -r[0])

        def fmt(r):
            m = r[3]
            vis = "n/a" if m.get("visible") is None else f"{m['visible']:.2f}"
            return f"- [{r[1]}] {r[2]}: closure {m['closure']:.2f}, reward {m['reward']:.2f}, visible {vis}"

        user = (
            "Features that scored best:\n" + "\n".join(fmt(r) for r in rows[:max_items])
            + "\n\nFeatures that scored worst:\n" + "\n".join(fmt(r) for r in rows[-max_items:])
            + "\n\nSources of the three best programs:\n\n"
            + "\n\n".join(f"```python\n{e['source']}```" for e in sorted(self.entries, key=lambda e: -self._mean_score(e))[:3])  # noqa: E501
            + "\n\nWrite at most 8 one-line lessons about what kinds of state abstractions score well or "
            "badly on closure, reward relevance and visibility, phrased so they transfer to new tasks. "
            "Plain lines starting with '- ', nothing else."
        )  # fmt: skip
        text = llm.chat([{"role": "user", "content": user}], tag="library/insights")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("- ")][:8]
        if lines:
            self.insights = "\n".join(lines)
            self.save()
        return self.insights

    def refactor(self, llm, data, top_k=10, atol=1e-4):
        """Ask for shared helpers; accept only rewrites that reproduce the original outputs."""
        pool = [e for e in sorted(self.entries, key=lambda e: -self._mean_score(e)) if self._compatible(e, data.spec)]
        pool = pool[:top_k]
        if len(pool) < 3:
            return {"helpers_added": 0, "rewritten": 0}
        listing = "\n\n".join(f"# id: {e['id']}\n```python\n{e['source']}```" for e in pool)
        user = (
            "These state-abstraction programs share repeated computations.\n\n"
            + listing
            + "\n\n"
            "1. Write up to 3 small helper functions capturing computations that recur in at least two "
            "programs. Put all helpers in ONE ```python block; each needs a one-line docstring and a new "
            "snake_case name. Helpers take and return tensors and follow the same restrictions.\n"
            "2. Rewrite each program that benefits, each in its own ```python block whose first line is "
            "`# id: <the id above>`. A rewrite must return exactly the same features and values.\n"
            "Existing helpers you may use: "
            + ", ".join(sorted(set(HELPERS) | set(self.helpers)))
            + "."
        )
        blocks = code_blocks(llm.chat([{"role": "user", "content": user}], tag="library/refactor"))
        if not blocks:
            return {"helpers_added": 0, "rewritten": 0}
        helper_block = next((b for b in blocks if "# id:" not in b.split("\n", 1)[0]), None)
        if helper_block is None:
            return {"helpers_added": 0, "rewritten": 0}
        try:
            new_funcs = compile_functions(helper_block, dict(self._helper_namespace()))
        except ProgramError:
            return {"helpers_added": 0, "rewritten": 0}
        taken = set(HELPERS) | set(TORCH_NAMESPACE) | set(self.helpers) | {"phi"}
        if any(name in taken for name in new_funcs):
            return {"helpers_added": 0, "rewritten": 0}
        trial = {**self.helpers, "refactor_" + str(len(self.helpers)): helper_block}
        by_id = {e["id"]: e for e in pool}
        accepted, uses = {}, {name: 0 for name in new_funcs}
        for block in blocks:
            head, _, body = block.partition("\n")
            m = re.match(r"#\s*id:\s*(\w+)", head)
            if not m or m.group(1) not in by_id:
                continue
            entry = by_id[m.group(1)]
            try:
                old = Program(entry["source"], data.spec, self.helpers, max_dim=10**6)
                new = Program(body, data.spec, trial, max_dim=10**6)
                same = new.dim_names == old.dim_names and torch.allclose(
                    new(data.priv), old(data.priv), atol=atol, rtol=1e-3, equal_nan=True
                )
            except ProgramError:
                continue
            if not same:
                continue
            called = [name for name in new_funcs if re.search(rf"\b{name}\(", body)]
            if called:
                accepted[entry["id"]] = new.source
                for name in called:
                    uses[name] += 1
        if not accepted or max(uses.values()) < 2:
            return {"helpers_added": 0, "rewritten": 0}
        self.helpers = trial
        for entry_id, source in accepted.items():
            by_id[entry_id]["source"] = source
        self.save()
        return {"helpers_added": sum(v >= 1 for v in uses.values()), "rewritten": len(accepted)}

    def _helper_namespace(self):
        namespace = {}
        for source in self.helpers.values():
            namespace.update(compile_functions(source, namespace))
        return namespace
