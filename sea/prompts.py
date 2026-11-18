"""Prompt construction for proposing and revising abstraction programs."""

from sea.program import HELPER_DOC, MAX_OUTPUT_DIM

SYSTEM = f"""You design state abstractions for model-based reinforcement learning.

A world model learns to control a robot from camera images only. During training we also
have the simulator's privileged state, and we want to turn it into a small set of
quantities that the world model is trained to predict from its latent state. Good targets
make the latent state capture the structure that matters for the task.

You write Python functions of this exact form:

```python
def phi(s):
    \"\"\"One line saying what this abstraction captures.\"\"\"
    rel = s["obj_pos"] - s["hand_pos"]
    return {{"hand_to_obj": dist(s["hand_pos"], s["obj_pos"]), "obj_above_hand": height(rel)}}
```

Contract:
- `s` maps privileged field names to torch tensors of shape (..., dim) with arbitrary
  leading batch/time dimensions. Use broadcasting operations on the last axis only. Never
  reduce over, index into, or loop over the leading dimensions.
- Return a dict from short snake_case feature names to tensors of shape (..., k), k >= 1.
  At most {MAX_OUTPUT_DIM} output dimensions in total; a handful is usually best.
- No imports, classes, lambdas, while-loops or try/except. No tensor methods that leave
  torch (.numpy(), .item(), .to(), ...).
- Available torch functions (call them directly, without a `torch.` prefix): abs, sqrt, exp,
  log, log1p, tanh, sigmoid, sin, cos, atan2, acos, asin, sign, clamp, minimum, maximum, where,
  cat, stack, zeros_like, ones_like, relu, square, isfinite, logical_and, logical_or, logical_not.
- Available helpers:
{HELPER_DOC}

Each program is scored automatically on recorded experience, without training anything:
- closure: the next value of the abstraction is predictable from its current value and
  the action. An abstraction should be a self-contained description of what evolves.
- control: the action improves that prediction (so it is not a clock or a distractor).
- reward: the return over the next few steps is predictable from the abstraction.
- visible: the abstraction can be decoded from the image-based world model's latent state.
  Quantities that cannot be seen (hidden forces, exact velocities) score low.
- compactness: fewer dimensions and shorter programs are preferred.
The score is a geometric mean, so a program must do reasonably on all of them. Prefer
relative, task-level quantities (distances, alignments, contacts, phases) over raw absolute
coordinates. Outputs are standardized before scoring, so rescaling a feature changes nothing."""


def _fmt(x):
    return "n/a" if x is None else f"{x:.2f}"


def fields_block(spec, stats=None):
    lines = []
    for f in spec.fields:
        line = f'- s["{f.name}"]  dim {f.dim}  {f.doc}'
        if stats and f.name in stats:
            st = stats[f.name]
            if st["std"] < 1e-6:
                line += "  [constant in the recorded data: carries no information here]"
            else:
                line += f"  [range {st['min']:.3g} to {st['max']:.3g}]"
        lines.append(line)
    return "\n".join(lines)


def program_block(entry, per_dim=True):
    m = entry.metrics
    head = (
        f"score {_fmt(m.get('score'))} | closure {_fmt(m.get('closure'))} | control {_fmt(m.get('control'))} | "
        f"reward {_fmt(m.get('reward'))} | visible {_fmt(m.get('visible'))} | "
        f"already-known {_fmt(m.get('known'))} | dims {m.get('dim')}"
    )
    lines = [head]
    if per_dim and m.get("per_dim"):
        for name, d in m["per_dim"].items():
            lines.append(
                f"    {name}: closure {_fmt(d.get('closure'))}, control {_fmt(d.get('control'))}, "
                f"reward {_fmt(d.get('reward'))}, visible {_fmt(d.get('visible'))}"
            )
    if m.get("dead_dims"):
        lines.append(f"    constant (useless) outputs: {', '.join(m['dead_dims'])}")
    return "\n".join(lines) + f"\n```python\n{entry.program.source}```"


def build(task_text, spec, stats, mode, n, parents=(), elites=(), failures=(), diagnostics=None, insights="",
          helper_docs=""):  # fmt: skip
    """Chat messages asking for `n` new programs. mode: init | mutate | crossover | explore."""
    parts = [f"Task: {task_text}", "Privileged fields:\n" + fields_block(spec, stats)]
    if helper_docs:
        parts.append("Additional shared helper functions (already defined, call them directly):\n" + helper_docs)
    if insights:
        parts.append("Lessons from abstractions scored on earlier tasks:\n" + insights)
    if diagnostics:
        rows = "\n".join(f"- {k}: {v:.2f}" for k, v in sorted(diagnostics.items(), key=lambda kv: -kv[1]))
        parts.append(
            "The world model is currently trained to predict the features below. The number is its "
            "remaining prediction error as a fraction of the feature's variance (0 = fully learned, "
            "1 = not learned at all). Features stuck near 1 are probably not inferable from images; "
            "features near 0 are already represented and need no replacement:\n"
            + rows
        )
    if elites:
        parts.append("Best abstractions so far:\n\n" + "\n\n".join(program_block(e, per_dim=False) for e in elites))
    if mode == "mutate" and parents:
        parts.append("Program to improve (with per-feature scores):\n\n" + program_block(parents[0]))
        ask = (
            f"Write {n} improved variants of this program. Keep what scores well; replace or repair the "
            "features that score poorly; drop redundant ones. Each variant should try a different idea."
        )
    elif mode == "crossover" and len(parents) >= 2:
        parts.append("Programs to combine:\n\n" + "\n\n".join(program_block(p) for p in parents[:2]))
        ask = (
            f"Write {n} programs that combine the strongest features of these two into one compact, "
            "self-contained abstraction, dropping features that score poorly or duplicate each other."
        )
    elif mode == "explore":
        ask = (
            f"Write {n} programs that capture aspects of the task that the abstractions above miss "
            "entirely. Do not rephrase existing features."
        )
    else:
        ask = (
            f"Write {n} diverse programs. Vary what they focus on (geometry between the key entities, "
            "contact and grasp state, progress toward the goal, motion) and how many features they use."
        )
    if failures:
        parts.append(
            "Recent programs that were rejected, and why (avoid these mistakes):\n"
            + "\n".join(f"- {msg}" for msg in failures)
        )
    parts.append(ask + " Put each program in its own ```python block, each defining `def phi(s):`.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]
