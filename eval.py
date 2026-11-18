"""Zero-shot evaluation of a trained agent under configuration or dynamics shift.

    python eval.py --logdir logdir/run --episodes 50 --shifts "motor=0.5" "friction=1.5" "motor=0.5,friction=0.5"
    python eval.py --logdir logdir/run --episodes 50 --configs test          # Meta-World held-out layouts

Results are written to <logdir>/eval_<name>.json. The policy acts from pixels only; the
privileged state and the semantic head are not used at evaluation time.
"""

import argparse
import json
import pathlib

import torch
from omegaconf import OmegaConf

import tools
from dreamer import Dreamer
from envs import make_envs


@torch.no_grad()
def run_episodes(agent, envs, episodes):
    """Returns per-episode returns, success flags (nan if the suite has none) and task ids."""
    returns, successes, tasks = [], [], []
    while len(returns) < episodes:
        n = envs.env_num
        done = torch.ones(n, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(n, dtype=torch.bool, device=agent.device)
        ret = torch.zeros(n, device=agent.device)
        success = torch.full((n,), float("nan"), device=agent.device)
        task = torch.zeros(n, dtype=torch.long, device=agent.device)
        state = agent.get_initial_state(n)
        act = state["prev_action"].clone()
        while not once_done.all():
            trans, step_done = envs.step(act.detach(), done)
            trans = trans.to(agent.device)
            done = step_done.to(agent.device)
            act, state = agent.act(trans, state, eval=True)
            ret += trans["reward"][:, 0] * ~once_done
            if "log_success" in trans.keys():
                hit = trans["log_success"][:, 0].float() * ~once_done
                success = torch.maximum(torch.nan_to_num(success, nan=0.0), hit)
            if "task_id" in trans.keys():
                task = trans["task_id"][:, 0].long()
            once_done |= done
        returns += ret.tolist()
        successes += success.tolist()
        tasks += task.tolist()
    return returns[:episodes], successes[:episodes], tasks[:episodes]


def parse_shift(text):
    return {k: float(v) for k, v in (item.split("=") for item in text.split(",") if item)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--checkpoint", default="latest.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--envs", type=int, default=10)
    parser.add_argument("--shifts", nargs="*", default=None, help="DMC: 'motor=0.5,friction=1.25' per condition")
    parser.add_argument("--configs", default=None, help="Meta-World: random | train | test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir).expanduser()
    config = OmegaConf.load(logdir / ".hydra" / "config.yaml")
    if args.device:
        config.device = args.device
    config.seed = args.seed
    config.model.compile = False
    config.env.env_num = 1  # only the evaluation workers are used
    config.env.eval_episode_num = args.envs
    tools.set_seed_everywhere(args.seed)

    conditions = (
        [{"configs": args.configs}] if args.shifts is None else [{"shift": parse_shift(s)} for s in args.shifts]
    )
    agent, results = None, []
    for cond in conditions:
        if "shift" in cond:
            config.env.eval_shift = cond["shift"]
        elif cond["configs"]:
            config.env.eval_configs = cond["configs"]
        train_envs, eval_envs, obs_space, act_space, info = make_envs(config.env)
        if agent is None:
            agent = Dreamer(config.model, obs_space, act_space, info["priv_spec"], len(info["tasks"])).to(config.device)
            state = torch.load(logdir / args.checkpoint, map_location=config.device, weights_only=False)
            agent.load_state_dict(state["agent_state_dict"])
            agent.clone_and_freeze()
            agent.eval()
        returns, successes, tasks = run_episodes(agent, eval_envs, args.episodes)
        ret = torch.tensor(returns)
        suc = torch.tensor(successes)
        row = {
            "condition": cond,
            "episodes": len(returns),
            "return_mean": float(ret.mean()),
            "return_std": float(ret.std()) if len(returns) > 1 else 0.0,
            "success_rate": None if bool(torch.isnan(suc).all()) else float(torch.nan_to_num(suc).mean()),
        }
        if len(info["tasks"]) > 1:
            task_ids = torch.tensor(tasks)
            row["per_task"] = {
                name: {
                    "return_mean": float(ret[task_ids == i].mean()),
                    "success_rate": float(suc[task_ids == i].mean()),
                }
                for i, name in enumerate(info["tasks"])
                if bool((task_ids == i).any())
            }
        results.append(row)
        print(json.dumps(row))
        for envs in (train_envs, eval_envs):
            for env in envs.envs:
                env.close()
    name = args.name or ("shift" if args.shifts is not None else f"configs_{args.configs or 'default'}")
    out = logdir / f"eval_{name}.json"
    with out.open("w") as f:
        json.dump({"checkpoint": args.checkpoint, "seed": args.seed, "results": results}, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
