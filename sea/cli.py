"""Offline tools for the abstraction search.

    python -m sea.cli baselines --data D.pt --out progs/            raw-state and random-mixture controls
    python -m sea.cli sample    --data D.pt --task T --n 30 --out progs/   one-shot LLM programs (no scoring)
    python -m sea.cli score     --data D.pt --programs progs/ --out scores.json
    python -m sea.cli evolve    --data D.pt --task T --out run/ [--library lib/]
    python -m sea.cli rankcorr  --scores scores.json --runs 'logdir/pilot_*'
    python -m sea.cli library   --path lib/ [--insights] [--refactor --data D.pt]

D.pt is a scoring dataset written during training (`sea.dump_data_at=[...]`).
"""

import argparse
import glob
import json
import pathlib
import random

from omegaconf import OmegaConf

from sea import fitness, tasks
from sea.archive import load_programs
from sea.evolve import Evolver
from sea.fitness import Dataset
from sea.library import Library
from sea.llm import make_llm
from sea.program import ProgramError, identity_source, random_projection_source

ROOT = pathlib.Path(__file__).resolve().parent.parent


def sea_config(overrides):
    cfg = OmegaConf.load(ROOT / "configs" / "configs.yaml").sea
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides or [])))


def task_data(args):
    data = Dataset.load(args.data)
    names = data.meta.get("tasks") or [args.task]
    if getattr(args, "task", None) and len(names) > 1:
        data = data.for_task(names.index(args.task))
    return data


def program_files(path):
    path = pathlib.Path(path)
    if path.is_dir():
        return sorted(path.glob("*.py"))
    return [path]


def cmd_baselines(args):
    data = Dataset.load(args.data)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = data.field_stats()
    live = [n for n in data.spec.names if stats[n]["std"] > 1e-6]
    (out / "raw.py").write_text(identity_source(data.spec, live))
    for i in range(args.random):
        (out / f"random_{i:02d}.py").write_text(random_projection_source(data.spec, args.dim, seed=i, fields=live))
    raw_dim = sum(data.spec.slices[n].stop - data.spec.slices[n].start for n in live)
    print(f"wrote raw.py ({raw_dim} dims; needs sea.max_dim >= {raw_dim}) and {args.random} random programs to {out}")


def cmd_sample(args):
    cfg = sea_config(args.set)
    data = task_data(args)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    evolver = Evolver(data.spec, tasks.describe(args.task), make_llm(cfg.llm, out), cfg.evolver, name=args.task)
    programs = evolver.one_shot(args.n, data.field_stats())
    for i, program in enumerate(programs):
        (out / f"oneshot_{i:02d}.py").write_text(program.source)
    print(f"wrote {len(programs)} one-shot programs to {out} ({evolver.llm.calls} LLM calls)")


def cmd_score(args):
    cfg = sea_config(args.set)
    data = task_data(args)
    rows = {}
    for path in program_files(args.programs):
        try:
            program = load_programs(path, data.spec, max_dim=int(cfg.max_dim))[0]
            rows[path.name] = fitness.score_program(program, data, weights=dict(cfg.evolver.weights))
        except ProgramError as e:
            rows[path.name] = {"score": 0.0, "reject": str(e)}
        m = rows[path.name]
        print(
            f"{path.name:28s} score {m['score']:.3f}  "
            + "  ".join(
                f"{k} {m[k]:.2f}" for k in ("closure", "control", "reward", "visible", "known") if m.get(k) is not None
            )
            + (f"  REJECTED: {m['reject']}" if m.get("reject") else "")
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


def cmd_evolve(args):
    cfg = sea_config(args.set)
    data = task_data(args)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    library = Library(args.library, dup_r2=float(cfg.library_dup_r2)) if args.library else None
    evolver = Evolver(
        data.spec, tasks.describe(args.task), make_llm(cfg.llm, out), cfg.evolver,
        logdir=out, library=library, seed=args.seed, name=args.task,
    )  # fmt: skip
    archive = evolver.run(data, generations=args.generations)
    picked = archive.select_active(data, int(cfg.budget_dims), float(cfg.max_redundancy))
    with (out / "selected.json").open("w") as f:
        json.dump([e.to_json() for e in picked], f, indent=1)
    if library is not None:
        print("library:", library.absorb(args.task, archive, data, picked))
    print(f"best score {archive.ranked()[0].score:.3f}; selected {len(picked)} programs, "
          f"{sum(e.program.dim for e in picked)} dims -> {out / 'selected.json'}")  # fmt: skip


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs, ys):
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx, vy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else float("nan")


def run_outcome(run, metric, last):
    """(program file name, mean of the last `last` evaluations) for one training run."""
    run = pathlib.Path(run)
    config = OmegaConf.load(run / ".hydra" / "config.yaml")
    values = []
    with (run / "metrics.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if metric in row:
                values.append(row[metric])
    if not values or not config.sea.programs:
        return None
    return pathlib.Path(str(config.sea.programs)).name, sum(values[-last:]) / len(values[-last:])


def cmd_rankcorr(args):
    with open(args.scores) as f:
        scores = json.load(f)
    outcomes = {}
    for run in sorted(glob.glob(args.runs)):
        res = run_outcome(run, args.metric, args.last)
        if res is not None:
            outcomes.setdefault(res[0], []).append(res[1])
    names = [n for n in outcomes if n in scores and not scores[n].get("reject")]
    if len(names) < 4:
        raise SystemExit(f"need at least 4 scored programs with finished runs, found {len(names)}")
    ys = [sum(outcomes[n]) / len(outcomes[n]) for n in names]
    rng = random.Random(0)
    print(f"{len(names)} programs, {sum(len(outcomes[n]) for n in names)} runs, outcome = {args.metric}")
    report = {}
    for key in ("score", "closure", "control", "reward", "visible", "known", "gap", "dim"):
        xs = [scores[n].get(key) for n in names]
        if any(x is None for x in xs):
            continue
        rho = spearman(xs, ys)
        boots = []
        for _ in range(2000):
            idx = [rng.randrange(len(names)) for _ in names]
            b = spearman([xs[i] for i in idx], [ys[i] for i in idx])
            if b == b:
                boots.append(b)
        boots.sort()
        lo, hi = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]) if boots else (float("nan"),) * 2
        report[key] = {"spearman": rho, "ci95": [lo, hi]}
        print(f"  {key:8s} rho = {rho:+.3f}   95% bootstrap CI [{lo:+.3f}, {hi:+.3f}]")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {"n": len(names), "metric": args.metric, "programs": dict(zip(names, ys)), "corr": report}, f, indent=1
            )


def cmd_library(args):
    cfg = sea_config(args.set)
    library = Library(args.path)
    if args.insights:
        print(library.update_insights(make_llm(cfg.llm, args.path)))
    if args.refactor:
        print(library.refactor(make_llm(cfg.llm, args.path), Dataset.load(args.data)))
    print(json.dumps(library.reuse_stats(), indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, fn):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--set", nargs="*", default=[], help="overrides of the `sea` config, e.g. llm.model=gpt-4o")
        return p

    p = add("baselines", cmd_baselines)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--random", type=int, default=3)
    p.add_argument("--dim", type=int, default=8)
    p = add("sample", cmd_sample)
    p.add_argument("--data", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--out", required=True)
    p = add("score", cmd_score)
    p.add_argument("--data", required=True)
    p.add_argument("--task", default=None)
    p.add_argument("--programs", required=True)
    p.add_argument("--out", default=None)
    p = add("evolve", cmd_evolve)
    p.add_argument("--data", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--library", default=None)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p = add("rankcorr", cmd_rankcorr)
    p.add_argument("--scores", required=True)
    p.add_argument("--runs", required=True, help="glob of training logdirs, one per (program, seed)")
    p.add_argument("--metric", default="episode/eval_score")
    p.add_argument("--last", type=int, default=5, help="average over the last N evaluations")
    p.add_argument("--out", default=None)
    p = add("library", cmd_library)
    p.add_argument("--path", required=True)
    p.add_argument("--insights", action="store_true")
    p.add_argument("--refactor", action="store_true")
    p.add_argument("--data", default=None)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
